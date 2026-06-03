Sparse Bayesian Image Expansion Filters (kanemura2009)

This small project implements the variational sparse Bayesian learning approach for image expansion filters described in Kanemura et al., "Learning Color Image Expansion Filters" (ICIP 2009). It provides:

- `sbef.py`: an implementation of the variational ARD learning algorithm for one-channel image expansion (models x = W y + eps).
- `demo.py`: a runnable demo that trains per-channel expanders on a test image and reconstructs an expanded image, reporting PSNR.
- `requirements.txt`: required Python packages.

How it maps to the paper

- The model and update equations follow the paper's Sec. 2-4. The implementation uses variational-factor updates for q(A), q(W), q(beta).
- The demo shows end-to-end usage: extract low/high resolution patches, train, and apply the learned filters.

Run the demo (Windows PowerShell):

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" demo.py
```

Notes and limitations

- This is a compact educational implementation. It focuses on clarity rather than performance.
- The training hyperparameters (a_alpha0, iterations) are simplified; for production experiments use the USC-SIPI dataset and follow the paper's preprocessing steps (anti-aliasing kernel, patch extraction, color space conversions).
