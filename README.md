# 🌙 Low-Light Enhancement — GAN vs U-Net

A comparative deep learning study evaluating:

- 🔥 GAN-based low-light enhancement
- 🧠 U-Net supervised enhancement

This project analyzes visual quality, convergence behavior, and model stability when restoring low-light images.

---

# 📌 Project Overview

Low-light images typically suffer from:
- Poor visibility
- Loss of contrast
- Noise amplification
- Color distortion

This project compares two approaches:

| Model | Type | Learning Strategy |
|-------|------|-------------------|
| GAN   | Adversarial | Generator vs Discriminator |
| U-Net | Supervised  | Encoder–Decoder with skip connections |

---

# 🔥 GAN Results

## 🖼 Visual Comparison

<img src="assets/gan_results.png" width="100%">

- **Result** → Generated enhanced image  
- **High Light** → Ground truth  
- **Low Light** → Input  
- **High - Result** → Difference map  

---

## 📈 GAN Training Curves

<img src="assets/gan_training.png" width="100%">

### Observations:

- Generator loss stabilizes quickly
- Discriminator accuracy approaches ~100%
- Precision & recall converge strongly
- Balanced adversarial training achieved

This indicates stable GAN convergence without collapse.

---

# 🧠 U-Net Results

## 🖼 Visual Comparison

<img src="assets/unet_results.png" width="100%">

U-Net produces:
- Strong brightness recovery
- Stable structural preservation
- Less adversarial texture hallucination

---

## 📈 U-Net Training Curves

<img src="assets/unet_training.png" width="100%">

### Observations:

- Smooth training loss convergence
- Validation accuracy fluctuates slightly
- Recall remains stable
- Precision trends downward slightly (possible mild overfitting)

---

# ⚖️ GAN vs U-Net Comparison

| Criteria | GAN | U-Net |
|-----------|------|-------|
| Visual realism | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Structural preservation | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Training stability | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Sharpness | High | Moderate |
| Overfitting risk | Medium | Low |

---

# 🧪 How To Run

```bash
git clone https://github.com/El7agAdel/Low-Light-Enhancement-GAN-vs-UNet.git
cd Low-Light-Enhancement-GAN-vs-UNet
