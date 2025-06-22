import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import load_generator

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATOR_PATH = "generator.pth"
Z_DIM = 100
NUM_SAMPLES = 5

# Load generator once
generator = load_generator(GENERATOR_PATH, device=DEVICE)

# Streamlit UI
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9)", list(range(10)))

if st.button("Generate images"):
    # Prepare noise and label tensors
    z = torch.randn(NUM_SAMPLES, Z_DIM, device=DEVICE)
    labels = torch.full((NUM_SAMPLES,), fill_value=digit, dtype=torch.long, device=DEVICE)

    # Generate images
    with torch.no_grad():
        gen_imgs = generator(z, labels).cpu()

    # Convert to displayable format
    cols = st.columns(NUM_SAMPLES)
    for i in range(NUM_SAMPLES):
        img_arr = gen_imgs[i].squeeze().numpy()  # shape (28,28)
        # Denormalize from [-1,1] to [0,255]
        img_uint8 = ((img_arr + 1) * 127.5).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode='L')

        with cols[i]:
            st.image(img_pil, caption=f"Sample {i+1}", use_column_width=True)