"""Helpers for the GNN workshop (DGM AI School 26, Aachen).

Only the plumbing lives here: reading the crystals, trimming the neighbour list, stacking graphs
into batches and the training loop. Everything that teaches something you write yourself.

    import workshop as wk

    data = wk.load_data()                       # crystals + measured Theta
    hist = wk.train(model, loss_fn, graphs, data.theta, i_tr, i_va)
    pred = wk.predict(model, graphs, i_te)
"""
from collections import namedtuple
import os
import time
import warnings

import numpy as np
import torch
from pymatgen.core import Structure

SEED = 42

Data = namedtuple("Data", "structures ids formulas theta lam omega")


# --------------------------------------------------------------------------- crystals
def load_data(folder=".", n=None):
    """Read the crystals from `folder/structures.tar.gz` and the targets from `folder/targets.csv`.

    The 6000 cif files travel as one compressed archive, so nothing has to be unpacked first.
    theta is the *measured* value (it carries ~4 % noise); lam and omega are the two computed
    ingredients behind it.
    """
    import csv
    import tarfile
    t0 = time.time()
    rows = list(csv.DictReader(open(os.path.join(folder, "targets.csv"))))
    if n:
        rows = rows[:n]

    text = {}                                            # id -> the contents of its cif file
    with tarfile.open(os.path.join(folder, "structures.tar.gz")) as tar:
        for member in tar:
            name = os.path.basename(member.name)
            if name.endswith(".cif"):
                text[name[:-4]] = tar.extractfile(member).read().decode()

    with warnings.catch_warnings():                      # cif rounding notices, harmless
        warnings.simplefilter("ignore")
        structures = [Structure.from_str(text[r["id"]], fmt="cif") for r in rows]
    print(f"{len(rows)} crystals read in {time.time() - t0:.0f} s")

    def column(key):
        return np.array([float(r[key]) for r in rows])

    return Data(structures, [r["id"] for r in rows], [r["formula"] for r in rows],
                column("theta"), column("lam"), column("omega"))



def keep_nearest(centre, neighbour, image, distance, k=12):
    """Of all the neighbours of each atom, keep only the k nearest."""
    order = np.lexsort((distance, centre))
    centre, neighbour = centre[order], neighbour[order]
    image, distance = image[order], distance[order]
    rank = np.arange(len(centre)) - np.searchsorted(centre, centre)
    keep = rank < k
    return centre[keep], neighbour[keep], image[keep], distance[keep] 


# --------------------------------------------------------------------------- batching
def batch(graphs, ids):
    """Stack several crystals into one big graph.

    Returns (x, c, n, d, b, n_graphs); `b` says which crystal each atom belongs to.
    """
    xs, cs, ns, ds, bs, off = [], [], [], [], [], 0
    for g, i in enumerate(ids):
        x, c, n, d = graphs[i]
        xs.append(x); cs.append(c + off); ns.append(n + off); ds.append(d)
        bs.append(torch.full((len(x),), g)); off += len(x)
    return torch.cat(xs), torch.cat(cs), torch.cat(ns), torch.cat(ds), torch.cat(bs), len(ids)


# --------------------------------------------------------------------------- training
def train(model, loss_fn, graphs, target, i_tr, i_va, epochs=20, lr=1e-3, batch_size=64,
          seed=SEED, quiet=False):
    
    torch.manual_seed(seed)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y = torch.tensor(np.asarray(target), dtype=torch.float32)

    def run(ids):
        return model(*batch(graphs, ids)).squeeze(-1)

    history, t0 = [], time.time()

    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(i_tr)

        for k in range(0, len(order), batch_size):
            ids = order[k:k + batch_size]
            loss = loss_fn(run(ids), y[ids])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()

        with torch.no_grad():
            train_loss = float(loss_fn(run(i_tr[:1000]), y[i_tr[:1000]]))
            val_loss = float(loss_fn(run(i_va), y[i_va]))
        history.append((train_loss, val_loss))

        if not quiet:
            print(f"epoch {epoch:3d}   train {train_loss:.3f}   validation {val_loss:.3f}")

    if not quiet:
        print(f"done in {time.time() - t0:.0f} s")

    return np.array(history)


def predict(model, graphs, ids):
    model.eval()
    with torch.no_grad():
        p = model(*batch(graphs, ids))
    return p.squeeze(-1).numpy() if p.shape[-1] == 1 else p.numpy()
