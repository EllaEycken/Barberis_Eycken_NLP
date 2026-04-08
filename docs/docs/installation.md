# Installation

## Recommended: Conda

This project was developed and tested using **Conda**.
Using Conda is strongly recommended due to complex NLP and audio dependencies.

---

### 1. Create the environment

From the repository root:

```bash
conda env create -f environment.yml
```

This creates a Conda environment called:
Ella_PhD_NLP

### 2. Activate the environment
```bash
conda activate Ella_PhD_NLP
```

Verify that the correct Python interpreter is used:
```bash
which python
```
Expected output contains:
.../anaconda3/envs/Ella_PhD_NLP/python


## Alternative: Pip
For users who do not use Conda:
```bash
pip install -r requirements.txt
```
⚠️ This may be less reliable on Windows due to compiled dependencies
(e.g. PyTorch, audio libraries).


---



## Next step: 
**data_setup**
