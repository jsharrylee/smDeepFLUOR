# -*- coding: utf-8 -*-
"""
TIFF I/O utilities for smDeepFLUOR.

Goal: standardize diverse TIFF inputs into a (T, Y, X) numpy array.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import tifffile as tiff


def read_tiff_stack_TYX(path: str) -> np.ndarray:
    """
    Read a TIFF stack and return a numpy array with shape (T, Y, X).

    Notes
    -----
    - Selects the largest TIFF series (more robust than using the first series).
    - Attempts to interpret common axis conventions (TYX, ZYX, YXT, YX).
    - Falls back to a generic reshape for unusual TIFF layouts.
    """
    with tiff.TiffFile(path) as tf:
        s = max(tf.series, key=lambda ss: int(np.prod(ss.shape)))
        arr = s.asarray()
        axes = getattr(s, "axes", "")

        if axes == "TYX":
            stack = arr
        elif axes == "ZYX":
            # Treat Z as time (common for single stack acquisitions)
            stack = arr
        elif axes == "YXT":
            stack = np.moveaxis(arr, -1, 0)  # (Y,X,T) -> (T,Y,X)
        elif axes == "YX":
            # Multi-page 2D TIFF -> treat pages as time
            if len(tf.pages) > 1:
                stack = np.stack([p.asarray() for p in tf.pages], axis=0)
            else:
                stack = arr[np.newaxis, ...]
        else:
            # Generic fallback: attempt to flatten leading dims into time
            if arr.ndim >= 3:
                tdim = int(np.prod(arr.shape[:-2]))
                stack = arr.reshape((tdim, arr.shape[-2], arr.shape[-1]))
            elif arr.ndim == 2:
                stack = arr[np.newaxis, ...]
            else:
                stack = np.squeeze(arr)
                if stack.ndim == 2:
                    stack = stack[np.newaxis, ...]

    return np.asarray(stack)
