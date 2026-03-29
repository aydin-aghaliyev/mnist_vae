import numpy as np
import streamlit as st
import keras
from vae.vae_words_model import WordVAE, decode_output

st.set_page_config(page_title="Word VAE – Name Generator", layout="centered")
st.title("Word VAE – Plausible Name Generator")
st.write(
    "Adjust the 16 latent dimensions to navigate the learned word space "
    "and generate plausible-looking English words."
)

MODEL_PATH = "vae_words.keras"


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)


try:
    vae = load_model()
except Exception:
    st.error(
        f"Could not load `{MODEL_PATH}`. "
        "Please run **train_words.py** first to generate the model file."
    )
    st.stop()

decoder = vae.decoder
latent_dim = vae.latent_dim

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for i in range(latent_dim):
    if f"wz_{i}" not in st.session_state:
        st.session_state[f"wz_{i}"] = 0.0

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header(f"Latent Vector ({latent_dim}D)")

col1, col2 = st.sidebar.columns(2)

if col1.button("Randomize"):
    vals = np.random.normal(0, 1.5, size=latent_dim)
    for i in range(latent_dim):
        st.session_state[f"wz_{i}"] = float(round(vals[i], 2))

if col2.button("Reset"):
    for i in range(latent_dim):
        st.session_state[f"wz_{i}"] = 0.0

st.sidebar.markdown("---")

z = np.array([[
    st.sidebar.slider(
        f"z[{i}]",
        min_value=-4.0,
        max_value=4.0,
        step=0.05,
        key=f"wz_{i}",
    )
    for i in range(latent_dim)
]])

# ---------------------------------------------------------------------------
# Decode & display
# ---------------------------------------------------------------------------
output = decoder.predict(z, verbose=0)[0]  # (MAX_LEN, VOCAB_SIZE)
word = decode_output(output)

st.markdown("## Generated word")
st.markdown(f"<h1 style='text-align:center; letter-spacing:0.15em'>{word or '(empty)'}</h1>",
            unsafe_allow_html=True)

# Per-character probability bars
st.markdown("### Character probabilities")
import pandas as pd
from src.vae.vae_words_model import VOCAB, MAX_LEN, VOCAB_SIZE

rows = []
for pos in range(MAX_LEN):
    probs = output[pos]
    top_idx = int(probs.argmax())
    rows.append({
        "Position": pos + 1,
        "Chosen char": VOCAB[top_idx] if VOCAB[top_idx] != " " else "⎵",
        "Confidence": float(probs[top_idx]),
    })

df = pd.DataFrame(rows)
st.dataframe(df.style.format({"Confidence": "{:.2%}"}), use_container_width=True)

# ---------------------------------------------------------------------------
# Batch random generation
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### Batch random sampling")
n_samples = st.slider("Number of words to generate", 5, 50, 20)

if st.button("Generate batch"):
    z_batch = np.random.normal(0, 1.5, size=(n_samples, latent_dim)).astype("float32")
    outputs = decoder.predict(z_batch, verbose=0)
    generated_words = [decode_output(o) for o in outputs]
    # Display in a tidy grid
    cols = st.columns(4)
    for idx, gw in enumerate(generated_words):
        cols[idx % 4].write(f"**{gw}**" if gw else "*(empty)*")

st.markdown("---")
st.caption("Latent vector: " + str(z[0].tolist()))
