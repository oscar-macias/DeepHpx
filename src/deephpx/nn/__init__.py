"""Neural network modules for DeepHpx.

Milestone 3 introduces the core building block needed to replace DeepSphere's
graph convolution stack without PyGSP / PyTorch Geometric dependencies:

- Chebyshev (ChebNet-style) graph convolution on HEALPix graphs.

Milestone 4 adds HEALPix sampling pooling / unpooling layers.

The intended workflow is:
1) Build a (scaled) Laplacian with :mod:`deephpx.graph.laplacian`.
2) Convert it to ``torch.sparse`` COO.
3) Use :class:`deephpx.nn.chebyshev.SphericalChebConv` inside your encoder.
"""

from __future__ import annotations

from .chebyshev import ChebConv, SphericalChebConv, cheb_conv
from .healpix_pooling import (
    Healpix,
    HealpixAvgPool,
    HealpixAvgUnpool,
    HealpixMaxPool,
    HealpixMaxUnpool,
)

__all__ = [
    "cheb_conv",
    "ChebConv",
    "SphericalChebConv",
    "HealpixAvgPool",
    "HealpixAvgUnpool",
    "HealpixMaxPool",
    "HealpixMaxUnpool",
    "Healpix",
]
