import numpy as np
import streamlit as st
import keras
import matplotlib.pyplot as plt
from vae_model import VAE

st.set_page_config(page_title="VAE Latent Space Explorer", layout="centered")
st.title("VAE Latent Space Explorer")
st.write("Adjust the 8 latent dimensions and see the decoded image.")

@st.cache_resource
def load_vae():
    return keras.models.load_model("vae.keras")

vae = load_vae()
decoder = vae.decoder
latent_dim = vae.latent_dim

# Initialise session state for each latent dimension
for i in range(latent_dim):
    if f"z_{i}" not in st.session_state:
        st.session_state[f"z_{i}"] = 0.0

st.sidebar.header(f"Latent Vector ({latent_dim}D)")

if st.sidebar.button("Randomize"):
    random_vals = np.random.uniform(-4.0, 4.0, size=latent_dim)
    for i in range(latent_dim):
        st.session_state[f"z_{i}"] = float(round(random_vals[i], 2))

z = np.array([[
    st.sidebar.slider(f"z[{i}]", min_value=-4.0, max_value=4.0, step=0.05, key=f"z_{i}")
    for i in range(latent_dim)
]])

decoded = decoder.predict(z, verbose=0)
img = decoded[0].reshape(28, 28)

fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(img, cmap="gray", vmin=0, vmax=1)
ax.axis("off")
st.pyplot(fig)

st.write("**Latent vector:**", z[0].tolist())
