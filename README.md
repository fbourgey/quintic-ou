# Quintic Ornstein-Uhlenbeck (OU) Models

This repository reproduces the results from the following papers on the quintic Ornstein-Uhlenbeck volatility model:

- Abi Jaber, Illand, and Li (2023). *The quintic Ornstein-Uhlenbeck volatility model that jointly
calibrates SPX & VIX smiles*. [arXiv](https://arxiv.org/pdf/2212.10917)

## Structure

- [quintic.ipynb](./quintic.ipynb): Jupyter notebook to compute and plot SPX and VIX implied volatilities.
- [checks.ipynb](./checks.ipynb): Jupyter notebook with various checks and validations from original code.

## Python Installation Guide

### Option 1: Standard Virtual Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/fbourgey/quintic-ou.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd quintic-ou
   ```

3. **Create a virtual environment:**

   ```bash
   python3 -m venv .venv
   ```

4. **Activate the environment:**

   ```bash
   source .venv/bin/activate
   ```

5. **Install dependencies:**

   ```bash
   pip install .
   ```

6. **Launch Jupyter Lab (optional):**

   ```bash
   jupyter lab
   ```

---

### Option 2: Using `uv` (Recommended)

If you have [`uv`](https://docs.astral.sh/uv/) installed, setup is simpler and faster.  
After cloning the repository (steps 1–2 above), run:

```bash
uv sync
```

This will automatically create a virtual environment and install all dependencies.
