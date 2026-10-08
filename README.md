# DataLabAssignement1

## generate.py
Use the file *generate.py* to complete your ratings table. 
It takes in argument *--name* the name of the files you want to use and it saves the complete matrix as *output.npy*.
DO NOT CHANGE THE LINES TO LOAD AND SAVE THE TABLE. Between those to you are free to use any method for matrix completion. 
Example:
  > python3 generate.py --name ratings_train.npy

## requirements.txt
Among the good pratice of datascience, we encourage you to use conda or virtualenv to create python environment. 
To test your code on our platform, you are required to update the *requirements.txt*, with the different librairies you might use. 
When your code will be test, we will execute: 
  > pip install -r requirements.txt


---

# Method

## The four rules

1. Load data with `cf.data.load_ratings()`: it checks the checksum, the shape (users × items) and the rating scale, and returns read-only arrays.
2. Tune on `cf.protocol.kfold_splits()`, the same 5 folds for everyone. Use `final_split()` once per method, to report, never to choose.
3. A model is a frozen dataclass with `complete(R, rng)`, in `cf/models/<method>.py`. Inside the model's implementation, any source of randomness should use the rng random generator for reproducibility.
4. Score with `cf.protocol.evaluate()` / `cross_validate()`, and record `PROTOCOL_VERSION` and the seed with every result.

Everything else is yours: scripts, logging format, tracking.
Keep the out-of-fold predictions (`cross_validate(...).oof`) of your final configurations: the blend is learned on them.

## Weights & Biases, optional

Team project `LeGroupe/Project A1`.
Never import `wandb` from `cf/` or `generate.py`: the grader installs `requirements.txt` only.

## Git

- One branch per person. Your files: `cf/models/<method>.py`, your scripts, `report/sections/<section>.tex`, `report/generated/<method>/`.
- A change to `cf/protocol.py` bumps `PROTOCOL_VERSION` and is announced to everyone.
- What `generate.py` submits is changed on `main` only.

## Report

`report/main.tex` inputs one file per section from `report/sections/`, each with one owner; write one sentence per line.
Build it with [Tectonic](https://tectonic-typesetting.github.io/): `tectonic report/main.tex`.
`report.pdf` and `slides.pdf` are committed at the repository root, on `main` only.
