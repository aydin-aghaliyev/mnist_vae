import tensorflow as tf
import keras
from keras import layers, Model
import numpy as np
from vae.vae_model import Sampling

MAX_LEN = 12
VOCAB = "abcdefghijklmnopqrstuvwxyz "  # 26 letters + space used as padding
VOCAB_SIZE = len(VOCAB)  # 27
INPUT_DIM = MAX_LEN * VOCAB_SIZE  # 324

char_to_idx = {c: i for i, c in enumerate(VOCAB)}


def encode_word(word: str) -> list:
    """Pad/truncate word to MAX_LEN and return a flat one-hot vector."""
    word = word.lower().ljust(MAX_LEN)[:MAX_LEN]
    one_hot = [[0.0] * VOCAB_SIZE for _ in range(MAX_LEN)]
    for i, c in enumerate(word):
        one_hot[i][char_to_idx.get(c, VOCAB_SIZE - 1)] = 1.0
    flat = [v for row in one_hot for v in row]
    return flat


def decode_output(vec) -> str:
    """Convert a flat probability vector back to a word string."""
    matrix = vec.reshape(MAX_LEN, VOCAB_SIZE)
    indices = matrix.argmax(axis=1)
    return "".join(VOCAB[i] for i in indices).rstrip(" ")


@keras.saving.register_keras_serializable()
class WordVAE(Model):
    """VAE for generating plausible English-looking words.

    The decoder uses per-character softmax so each output position is a
    probability distribution over the vocabulary, enabling categorical
    cross-entropy reconstruction loss.
    """

    def __init__(self, latent_dim: int = 16, max_len: int = MAX_LEN,
                 vocab_size: int = VOCAB_SIZE, free_bits: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.free_bits = free_bits
        self.input_dim = max_len * vocab_size
        # beta is updated externally by KLAnnealingCallback each epoch
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="beta")
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    # ------------------------------------------------------------------
    # Sub-networks
    # ------------------------------------------------------------------

    def _build_encoder(self):
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(512)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(64)(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dense(self.input_dim)(x)
        # Reshape to (max_len, vocab_size) then softmax over character dim
        x = layers.Reshape((self.max_len, self.vocab_size))(x)
        outputs = layers.Softmax(axis=-1, name="char_probs")(x)
        return Model(latent_inputs, outputs, name="decoder")

    # ------------------------------------------------------------------
    # Keras serialisation
    # ------------------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "max_len": self.max_len,
            "vocab_size": self.vocab_size,
            "free_bits": self.free_bits,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(
            latent_dim=config.get("latent_dim", 16),
            max_len=config.get("max_len", MAX_LEN),
            vocab_size=config.get("vocab_size", VOCAB_SIZE),
            free_bits=config.get("free_bits", 0.5),
        )

    # ------------------------------------------------------------------
    # Forward / losses
    # ------------------------------------------------------------------

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)

    def _compute_losses(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)  # (batch, max_len, vocab_size)

        # Reshape flat input back to (batch, max_len, vocab_size)
        data_reshaped = tf.reshape(data, (-1, self.max_len, self.vocab_size))

        # Categorical cross-entropy: − Σ_t Σ_c y*log(p)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                -tf.reduce_sum(data_reshaped * tf.math.log(reconstruction + 1e-8), axis=-1),
                axis=-1,
            )
        )

        # Free bits: clamp per-dimension KL to at least free_bits nats,
        # preventing trivial collapse of individual latent dimensions.
        kl_per_dim = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_per_dim = tf.maximum(kl_per_dim, self.free_bits)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=1))

        # KL annealing: beta ramps from 0 → 1 over the warmup period
        total_loss = reconstruction_loss + self.beta * kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "beta": self.beta,
        }

    def train_step(self, data):
        with tf.GradientTape() as tape:
            losses = self._compute_losses(data)
        grads = tape.gradient(losses["loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return losses

    def test_step(self, data):
        if isinstance(data, (tuple, list)):
            data = data[0]
        return self._compute_losses(data)


class KLAnnealingCallback(keras.callbacks.Callback):
    """Linearly ramps the VAE's beta from 0 to 1 over `warmup_epochs` epochs.

    Usage::

        vae = WordVAE(latent_dim=16)
        vae.compile(optimizer=Adam(learning_rate=0.001))
        vae.fit(x_train, epochs=30, callbacks=[KLAnnealingCallback(warmup_epochs=15)])
    """

    def __init__(self, warmup_epochs: int = 15):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        new_beta = min(1.0, (epoch + 1) / self.warmup_epochs)
        self.model.beta.assign(new_beta)
