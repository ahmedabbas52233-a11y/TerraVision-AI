"""
train.py — root-level entry point shim.
Delegates to terravision/train.py so you can run:
    python train.py [--epochs N] [--eval-only]
from the project root without cd-ing into the package.
"""
from terravision.train import main

if __name__ == "__main__":
    main()
