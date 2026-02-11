# Low-Light Enhancement — GAN vs. U-Net

A small research/engineering project comparing **two deep-learning approaches for low-light image enhancement**:

- **GAN-based enhancement** (adversarial training to push results toward realistic-looking outputs)
- **U-Net-based enhancement** (strong supervised baseline with skip connections to preserve structure)

This repository was created as a **CSE 681 project container** and includes both implementations, plus the project report and presentation. :contentReference[oaicite:1]{index=1}

---

## Contents

- `GAN/` — GAN-based low-light enhancement implementation :contentReference[oaicite:2]{index=2}  
- `Unet/` — U-Net enhancement implementation :contentReference[oaicite:3]{index=3}  
- `Report.pdf` — full report (methodology, experiments, results) :contentReference[oaicite:4]{index=4}  
- `Presentation.pdf` — slides :contentReference[oaicite:5]{index=5}  
- `README.md` — this file :contentReference[oaicite:6]{index=6}  
- **Presentation video** — linked in the repo landing page (YouTube) :contentReference[oaicite:7]{index=7}  

> Note: Most of the code is in notebooks (repo language shows Jupyter Notebook). :contentReference[oaicite:8]{index=8}

---

## Project Goal

Low-light images often suffer from:
- low visibility / crushed shadows
- noise amplification when brightness is increased
- color shifts and loss of detail

This project evaluates how a **GAN** compares to a **U-Net** baseline in:
- brightness/contrast recovery
- detail preservation
- artifact/noise behavior
- overall perceptual quality

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/El7agAdel/Low-Light-Enhancement-GAN-vs-UNet.git
cd Low-Light-Enhancement-GAN-vs-UNet
