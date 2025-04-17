Torch based deep learning framework implementing semiring backpropagation

It uses PyTorch for elemental operations but implements its own computational
graph and (semiring) backpropagation.

# Setup
Clone this repo and do:

- `pip install -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `pip install -e .`

This creates an editable install of the package. You can now use it inside the
`.venv` virtual enfironment by using `from semiflow import <module>`.
