# Graph Neural Networks for Crystals

Hands-on workshop, **DGM AI School 26 · Aachen**. You build a GNN for crystals from scratch —
the neuron, the graph, message passing, pooling, training and evaluation.

| notebook | |
|---|---|
| `workshop_starter.ipynb` — the exercises | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulopires0/aachen-school/blob/main/workshop_starter.ipynb) |
| `workshop_solutions.ipynb` — every answer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulopires0/aachen-school/blob/main/workshop_solutions.ipynb) |

**Run the first cell before anything else.** On Colab it clones this repository and installs
`pymatgen`; locally it does nothing. Everything after it assumes the working folder is the repo.

### Two things not to do on Colab

- **Do not run `pip install -r requirements.txt`.** That file pins a CPU build of PyTorch and
  would reinstall it over the one Colab already ships. It is for local installs only.
- **Do not bother with a GPU runtime.** Nothing in the notebook moves tensors to CUDA, so a GPU
  changes nothing. The CPU runtime trains the model in about a minute.

### Running locally instead

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab workshop_starter.ipynb
```

### What else is here

`workshop.py` holds the plumbing you never have to write — reading the crystals, trimming the
neighbour list, the training loop. `structures.tar.gz` is 6000 crystal structures as cif files,
read straight from the archive, and `targets.csv` holds the measured property for each one.

The research model shown at the end of the session is TUGA-SP:
<https://github.com/ppdebreuck/tuga-sp>. It is not needed for the workshop.
