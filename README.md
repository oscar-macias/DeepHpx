# DeepHpx (Milestone 5)

This is the **Milestone 5** cut of the DeepHpx project.

- **Milestone 1:** HEALPix map ingestion + ordering normalization (RING <-> NEST)
- **Milestone 2:** PyGSP-free HEALPix graph backend (neighbours -> adjacency -> Laplacian)
- **Milestone 3:** Chebyshev (ChebNet-style) graph convolution in pure PyTorch
- **Milestone 4:** HEALPix pooling/unpooling layers (NSIDE/2 <-> pixels/4)
- **Milestone 5:** Multi-resolution HEALPix encoder (maps -> fixed-length embeddings)

## Install (editable)

Base (NumPy only):

```bash
pip install -e .
```

Enable FITS I/O + ordering conversions + neighbour queries:

```bash
pip install -e '.[healpix]'
```

Enable sparse graph utilities (adjacency/Laplacian):

```bash
pip install -e '.[graph]'
```

(Optional) enable torch (needed for Milestone 3 NN layers):

```bash
pip install -e '.[torch]'
```

For development (pytest):

```bash
pip install -e '.[dev,healpix,graph,torch]'
pytest
```

## Milestone 5 features

- **Neighbours:** build 8-neighbour connectivity via `healpy.pixelfunc.get_all_neighbours`
- **Adjacency:** build a SciPy sparse CSR adjacency matrix from neighbour lists
- **Laplacian:** build combinatorial or normalized Laplacians and scale them for Chebyshev convolutions
- **Torch:** convert SciPy sparse matrices to `torch.sparse` COO tensors
- **ChebConv:** Chebyshev graph convolution (`deephpx.nn.ChebConv` / `deephpx.nn.SphericalChebConv`)
- **Pooling:** HEALPix AvgPool/MaxPool (+ corresponding unpool layers) for hierarchical downsampling
- **Encoder:** `deephpx.nn.HealpixEncoder` that stacks ChebConv + pooling and returns an embedding vector

## Examples

Build a graph for a given NSIDE and print stats:

```bash
python examples/01_build_graph.py --nside 32 --nest true --kind normalized
```

Run a Chebyshev convolution forward pass on a HEALPix graph:

```bash
python examples/02_chebconv_forward.py --nside 16 --K 3
```

Smoke test the HEALPix pooling/unpooling layers:

```bash
python examples/03_healpix_pooling_smoke.py --npix 12288 --mode average
python examples/03_healpix_pooling_smoke.py --npix 12288 --mode max
```

Smoke test the multi-resolution encoder:

```bash
python examples/04_encoder_forward.py --nside 16 --levels 3 --channels 8,16,32 --embedding-dim 64
```

