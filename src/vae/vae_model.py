import tensorflow as tf
import keras
from keras import layers, Model

@keras.saving.register_keras_serializable()
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


@keras.saving.register_keras_serializable()
class VAE(Model):
    def __init__(self, latent_dim=8, input_dim=784, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(128, activation="relu")(latent_inputs)
        x = layers.Dense(256, activation="relu")(x)
        outputs = layers.Dense(self.input_dim, activation="sigmoid")(x)
        return Model(latent_inputs, outputs, name="decoder")

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "input_dim": self.input_dim,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(
            latent_dim=config.get("latent_dim", 8),
            input_dim=config.get("input_dim", 784),
        )

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def _compute_losses(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        # Per-pixel BCE, summed over pixels, averaged over the batch
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.ops.binary_crossentropy(data, reconstruction), axis=-1
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
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
