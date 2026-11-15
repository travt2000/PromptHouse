"""Krea 2 Turbo standalone image generator."""

import os

# PyTorch's Windows builds allocate CPU memory with mimalloc, which by default keeps freed blocks below its
# "huge" threshold (~32 MB - i.e. nearly every model weight) committed for reuse. Without these options a
# model released from system RAM stays in the process's private bytes, and the app would misreport RAM as
# freed. Must be set before torch is imported (mimalloc reads options when c10.dll loads).
for _k, _v in (("MIMALLOC_PURGE_DECOMMITS", "1"), ("MIMALLOC_RESET_DECOMMITS", "1"),
               ("MIMALLOC_PURGE_DELAY", "0"), ("MIMALLOC_RESET_DELAY", "0")):
    os.environ.setdefault(_k, _v)
