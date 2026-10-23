"""Export the trained `no_llm` checkpoint (torch.save zip) to a plain .npz, without importing torch.

    python tools/export_weights.py checkpoints/no_llm_s1.pt data/no_llm_s1_weights.npz

The checkpoint is checkpoints/no_llm_s1.pt of the project (seed 1, split 1, best epoch 14, SHA-256
e424f216754909b20c580f67be4cf27f74af4351293317c36973bf77a4af022f). Keys keep their PyTorch names.
"""
import collections
import hashlib
import pickle
import sys
import zipfile

import numpy as np

DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64}


class _Storage:
    def __init__(self, arr):
        self.arr = arr


def _rebuild(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    n = int(np.prod(size)) if len(size) else 1
    return storage.arr[offset:offset + n].reshape(tuple(size)).copy()  # all tensors of the checkpoint are contiguous


def load_checkpoint(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild
            if module == "torch" and name in DTYPES:
                return name
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, storage_type, key, _location, _numel = pid
            return _Storage(np.frombuffer(z.read(f"{root}/data/{key}"), dtype=DTYPES[storage_type]))

    with z.open(f"{root}/data.pkl") as f:
        return Unpickler(f).load()


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = load_checkpoint(src)
    state = {k: v.astype(np.float32) for k, v in ckpt["state"].items()}
    np.savez(dst, **state)
    print("source sha256", hashlib.sha256(open(src, "rb").read()).hexdigest())
    print("meta", ckpt["meta"])
    print("tensors", len(state), "parameters", sum(v.size for v in state.values()), "->", dst)
