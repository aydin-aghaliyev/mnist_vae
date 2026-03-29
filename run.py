import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.optimizers import Adam
from vae_model import VAE

latent_dim = 8


# +
# Load the data (we don't need the labels 'y' for training the VAE)
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize and Flatten
# 28x28 pixels -> 784 
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

# +
vae = VAE(latent_dim=latent_dim)

# Compile with an optimizer (the loss is handled inside the VAE class)
vae.compile(optimizer=Adam(learning_rate=0.001))

# Train it!
history = vae.fit(x_train, epochs=30, batch_size=128, validation_data=(x_test, x_test))


# +
import matplotlib.pyplot as plt

epochs = range(1, len(history.history["loss"]) + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, key, title in zip(axes, ["loss", "reconstruction_loss", "kl_loss"], ["Total Loss", "Reconstruction Loss", "KL Loss"]):
    ax.plot(epochs, history.history[key], label="Train")
    ax.plot(epochs, history.history[f"val_{key}"], label="Validation", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.legend()

plt.tight_layout()
plt.show()


# +
import matplotlib.pyplot as plt
import umap

# Load labels so we can colour-code the points
(_, y_train), (_, y_test) = mnist.load_data()

# Sample 50 random test images
n = 50
indices = np.random.choice(len(x_test), n, replace=False)
x_sample = x_test[indices]
y_sample = y_test[indices]

# Get latent representations (z_mean is the deterministic projection)
z_mean, _, _ = vae.encoder.predict(x_sample, verbose=0)

# Reduce 8D latent space -> 2D with UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
z_2d = reducer.fit_transform(z_mean)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y_sample, cmap="tab10", s=80, alpha=0.85)
plt.colorbar(scatter, ax=ax, label="Digit class")
ax.set_title("UMAP projection of 8D latent space (50 MNIST samples)")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
plt.tight_layout()
plt.show()


# +
# Define an arbitrary latent vector (latent_dim = 8)
z_custom = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

# Decode it
decoded = vae.decoder.predict(z_custom, verbose=0)

# Reshape from 784 -> 28x28 and plot
img = decoded[0].reshape(28, 28)
plt.figure(figsize=(3, 3))
plt.imshow(img, cmap="gray")
plt.title("Decoded image from custom latent vector")
plt.axis("off")
plt.tight_layout()
plt.show()

# -

vae.save("vae.keras")
print("Model saved: vae.keras")

