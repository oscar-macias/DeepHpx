# DeepHpx (Milestone 1)

This is the **Milestone 1** cut of the DeepHpx project: **HEALPix map ingestion** and **ordering normalization**.

## Install (editable)

```bash
pip install -e .
```

To enable FITS I/O and ordering conversions, install the HEALPix extra:

```bash
pip install -e '.[healpix]'
```

## Milestone 1 features

- Load HEALPix maps from:
  - `.fits` / `.fits.gz` via `healpy.fitsfunc.read_map`
  - `.npy`, `.npz` via NumPy
- Convert ordering between **RING** and **NESTED/NEST** via `healpy.pixelfunc.reorder`
- Minimal smoke example in `examples/00_smoke_read_map.py`

## Example

```bash
python examples/00_smoke_read_map.py /path/to/map.fits --assume-input-order RING --to NEST
```

