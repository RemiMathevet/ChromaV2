***English** | [Français](README.fr.md)*

# CHROMA V2

**C**alibrated **H**istological **R**econstruction **O**f **M**ultiplexed **A**bsorbances

A stain normalizer for **HES** slides (Hematoxylin–Eosin–Saffron) that works in
**optical density** and separates brightness from chromaticity **before** decomposition.
Single file, `numpy` + `scikit-learn`, no deep learning.

![Native / CHROMA V2 / Macenko comparison at x20](assets/comparison_natif_chroma_macenko_x20.png)

*Placenta patches at x20. Left column: native image (inter-slide staining variability).
Middle: CHROMA V2. Right: Macenko. CHROMA harmonizes color while staying close to the
native image (DINOv2 cosine distance ≈ 0.02–0.03 vs native, against ≈ 0.05–0.09 for Macenko
on the same patches) and preserves the pigments and red blood cells that Macenko erases.*

## Why

HES staining varies from slide to slide (reagents, bath age, scanner). This color drift
disturbs both the pathologist's eye and encoders (DINOv2, CTransPath…). Existing normalizers
each have a weakness:

| Method | Limitation |
|---|---|
| **Macenko** (SVD, 2 components H&E) | Ignores saffron, unstable per slide, destroys pigments (meconium, hemosiderin) and red blood cells |
| **Reinhard** (Lab) | Separates brightness/chromaticity but in a **perceptual** space, not a physical one |

**CHROMA** is the synthesis: OD space (physical, additive via Beer-Lambert) + brightness/
chromaticity separation (Reinhard's idea) + a biological 3-stain decomposition.

### The physical principle

The 3 HES stains behave like the 3 subtractive **CMYK** primaries: H ≈ Cyan, E ≈ Magenta,
S ≈ Yellow, plus a residual = Key (pigments). Moving to optical density makes the mixture
**additive**:

```
OD = -log10(I / I0)          (Beer-Lambert law, I0 = bright background per slide)
```

### The V2 novelty: remove the gray axis before decomposing

The gray axis `ĝ = [1,1,1]/√3` in OD encodes **thickness/brightness variation** without color.
In V1 it contaminated the saffron channel (saffron would "light up" on gray tissue). V2 removes
it **before** decomposition:

```
T   (brightness)     = OD · ĝ                  tissue thickness, scalar per pixel
Sat (saturation)     = ‖OD − T·ĝ‖              chromatic intensity
c   (concentrations) = NNLS(S⊥, OD − T·ĝ) ≥ 0   on chromaticity only
K   (residual)       = OD − (T·ĝ + S⊥·c)       unexplained pigments
```

where `S⊥ = S − ĝ(ĝᵀS)` is the stain matrix projected out of the gray axis. Result: brightness
is carried by `T`, never by the stains → saffron no longer captures gray. This rank-2 path is
used by `normalize()` only.

`decompose()` (since 2.2) works at **full rank 3**, gray axis kept: `c = NNLS(S, OD)`,
`K = ‖OD − S·c‖` = true out-of-cone residual. In rank 2, `S⊥` spans the whole chromatic plane,
so H/E/S are not identifiable and K is only a clipping residual. Thickness invariance in rank 3
comes from the ratios (`Cellularite = H/E`, `Fibrose = S/(H+E)`).

Since 2.2, H/E/S are assigned to the NMF vectors by the permutation maximizing cosine similarity
to the canonical basis (2.0 used a single-channel argmax that swapped H and S on RGB input).

### 7 interpretable channels

```
RGB → OD ─┬─ T   [tissue thickness]
          ├─ Sat [chromatic intensity]
          └─ chromaticity ─ decomposition ─┬─ H [hematoxylin / nuclei]
                                           ├─ E [eosin / cytoplasm]
                                           ├─ S [saffron / collagen]
                                           └─ K [residual: meconium, hemosiderin, formalin]

Total: H, E, S, K, T, Sat + normalized image
```

The **K channel is left untouched** during normalization: this is the key advantage over
Macenko, which wipes out these diagnostic pigments. The gray-decontaminated **H channel**
isolates pure basophilia and turns out to be a strong descriptor of necrosis and inflammation.

## Automatic calibration

No annotation, no human intervention. We pool tissue pixels (Otsu mask on saturation) from
~10 HES slides, run a 3-component NMF **anchored** on a canonical HES basis (otherwise NMF is
non-identifiable on the HES cone), and assign H/E/S by the permutation that best matches that
basis (cosine). A single site-specific artifact: a JSON calibration file.

**Site-median target (2.3, recommended).** Rather than a target calibrated on a small pool,
calibrate every slide (one source JSON each), then take the median: `chroma.site_target(calibs)`
(median stain vectors, I0 and stats; no slide is re-read). On our site, a target calibrated on
10 slides had a hematoxylin vector pulled towards red; after transfer, nuclei turned blue-green.

## Usage

```python
import numpy as np
from PIL import Image
import chroma

# 1) calibrate a TARGET on ~10 RGB thumbnails of HES slides (once per site/scanner)
target = chroma.auto_calibrate([np.asarray(Image.open(p).convert("RGB")) for p in thumbs])
norm = chroma.CHROMA(target)

# 2) for each slide to normalize: calibrate its SOURCE and build the PIL→PIL function
src = chroma.auto_calibrate([np.asarray(Image.open("slide.png").convert("RGB"))])
stain_fn = chroma.make_stain_fn(src, norm)

normalized = stain_fn(Image.open("patch.png"))     # normalized RGB PIL.Image

# 2bis) rank-3 transfer (2.3): needs `stats_full` in source and target
#       (written by auto_calibrate ≥ 2.3; site_target computes it too)
stain_fn3 = chroma.make_stain_fn(src, norm, rank3=True)

# 3) decomposition into 7 channels (2D maps)
channels = norm.decompose(np.asarray(Image.open("patch.png").convert("RGB")))
H, E, S, K, T, Sat = (channels[k] for k in ("H", "E", "S", "K", "T", "Sat"))
```

`stain_fn` plugs directly in as a `torchvision.transforms.Lambda` in an embedding pipeline.
Cost < 1 s per 224×224 patch on CPU.

### `normalize()` or `transfer()`?

`normalize()` (default) works on chromaticity, gray removed (rank 2). There, H⊥ and S⊥ are
almost opposite: on very dark pixels their concentrations cancel out, and per-stain gains
break the cancellation → green/cyan nuclei. `transfer()` (`rank3=True`) decomposes at full
rank 3 (`c = NNLS(S_source, OD)`), applies the gains `stats_full` target/source, recomposes
on the target stains, keeps K, and fades back to the native pixel where the sensor saturates
(max OD 1.2 → 1.8).

Benchmark (400 patches, 40 slides, towards the site-median target) — share of patches
with > 0.5 % new cyan pixels / inter-slide spread of the mean tissue color in Lab:

| | cyan | spread L / a* / b* |
|---|---|---|
| native | — | 6.8 / 5.1 / 6.1 |
| `normalize()` | 1.0 % | 6.0 / 9.4 / 5.3 |
| `transfer()` | 0.2 % | 5.0 / 6.7 / 3.9 |

Both increase the inter-slide spread on a*: per-slide gains mix the intensity of the stain
bath with the tissue composition (an eosin-rich slide gets its E lowered). `normalize()`
stays the default so that existing embeddings remain comparable.

Built-in self-check:

```bash
python chroma.py     # pure stains, out-of-cone K, thickness ratios, identity, rank-3 transfer
```

## Calibration format (JSON)

```json
{
  "method": "NMF_auto_chroma_separated",
  "chroma_version": "2.3",
  "I0": [253.7, 253.6, 253.7],
  "stain_vectors": { "H": [...], "E": [...], "S": [...] },
  "stats": {
    "mu": [...], "sigma": [...],
    "brightness_mu": 0.092, "brightness_sigma": 0.05,
    "sat_mu": 0.053, "sat_sigma": 0.03
  },
  "stats_full": { "mu": [...], "sigma": [...], "frac_3": 0.98, "K_p95": 0.05 },
  "calib_id": "4d75cb66"
}
```

## Dependencies

`numpy`, `scikit-learn`, `opencv-python`, `Pillow` (see `requirements.txt`). No framework,
no ORM, no GPU required.

## Authors

- **Rémi Mathevet** — fetal pathologist — design, specification, clinical validation
- **Claude** (Anthropic) — implementation, gray-separated decomposition, QC tooling

## License

Apache-2.0
