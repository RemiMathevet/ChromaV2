*[English](README.md) | **Français***

# CHROMA V2

**C**alibrated **H**istological **R**econstruction **O**f **M**ultiplexed **A**bsorbances

Normaliseur colorimétrique pour lames **HES** (Hématoxyline–Éosine–Safran) qui
travaille en **densité optique** et sépare la brightness de la chromaticité
**avant** décomposition. Un seul fichier, `numpy` + `scikit-learn`, sans deep learning.

![Comparatif natif / CHROMA V2 / Macenko en x20](assets/comparison_natif_chroma_macenko_x20.png)

*Patchs x20 de placenta. Colonne de gauche : image native (variabilité de coloration inter-lame).
Milieu : CHROMA V2. Droite : Macenko. CHROMA harmonise la couleur en restant proche
de l'image native (distance cosinus DINOv2 ≈ 0.02–0.03 vs natif, contre ≈ 0.05–0.09 pour
Macenko sur les mêmes patchs) et préserve les pigments et les érythrocytes que Macenko efface.*

## Pourquoi

La coloration HES varie d'une lame à l'autre (réactifs, âge du bain, scanner). Cette
dérive colorimétrique perturbe autant l'œil du pathologiste que les encodeurs (DINOv2,
CTransPath…). Les normaliseurs existants ont chacun un défaut :

| Méthode | Limite |
|---|---|
| **Macenko** (SVD, 2 composantes H&E) | Ignore le safran, instable par lame, détruit les pigments (méconium, hémosidérine) et les globules rouges |
| **Reinhard** (Lab) | Sépare brightness/chromaticité mais dans un espace **perceptuel**, pas physique |

**CHROMA** est la synthèse : espace OD (physique, additif par Beer-Lambert) + séparation
brightness/chromaticité (l'idée de Reinhard) + décomposition biologique à 3 colorants.

### Le principe physique

Les 3 colorants HES se comportent comme les 3 primaires soustractives **CMYK** :
H ≈ Cyan, E ≈ Magenta, S ≈ Yellow, et un résiduel = Key (pigments). Le passage en densité
optique rend le mélange **additif** :

```
OD = -log10(I / I0)          (loi de Beer-Lambert, I0 = fond clair par lame)
```

### La nouveauté V2 : retirer l'axe gris avant de décomposer

L'axe gris `ĝ = [1,1,1]/√3` en OD encode la **variation d'épaisseur/brightness** sans
couleur. En V1, il contaminait le canal safran (le safran « s'allumait » sur du tissu gris).
V2 le retire **avant** la décomposition :

```
T   (brightness)  = OD · ĝ                  épaisseur tissulaire, scalaire par pixel
Sat (saturation)  = ‖OD − T·ĝ‖              intensité chromatique
c   (concentrations) = NNLS(S⊥, OD − T·ĝ) ≥ 0   sur la chromaticité seule
K   (résiduel)    = OD − (T·ĝ + S⊥·c)       pigments non expliqués
```

où `S⊥ = S − ĝ(ĝᵀS)` est la matrice de stain projetée hors de l'axe gris. Résultat :
la brightness est portée par `T`, jamais par les colorants → le safran ne capte plus le gris.
Ce chemin de rang 2 sert uniquement à `normalize()`.

`decompose()` (depuis 2.2) travaille en **rang 3 plein**, axe gris conservé : `c = NNLS(S, OD)`,
`K = ‖OD − S·c‖` = vrai résidu hors-cône. En rang 2, `S⊥` engendre tout le plan chromatique :
H/E/S n'y sont pas identifiables et K n'est qu'un résidu d'écrêtage. L'invariance d'épaisseur
en rang 3 passe par les ratios (`Cellularite = H/E`, `Fibrose = S/(H+E)`).

Depuis 2.2, H/E/S sont assignés aux vecteurs NMF par la permutation qui maximise le cosinus à
la base canonique (la 2.0 utilisait un argmax mono-canal qui inversait H et S sur une entrée RGB).

### 7 canaux interprétables

```
RGB → OD ─┬─ T   [épaisseur tissulaire]
          ├─ Sat [intensité chromatique]
          └─ chromaticité ─ décomposition ─┬─ H [hématoxyline / noyaux]
                                           ├─ E [éosine / cytoplasme]
                                           ├─ S [safran / collagène]
                                           └─ K [résiduel : méconium, hémosidérine, formol]

Total : H, E, S, K, T, Sat + image normalisée
```

Le canal **K est laissé intact** pendant la normalisation : c'est l'avantage clé sur Macenko,
qui écrase ces pigments diagnostiques. Le canal **H** décontaminé du gris isole la basophilie
pure et se révèle un fort descripteur de la nécrose et de l'inflammation.

## Calibration automatique

Aucune annotation, aucune intervention humaine. On pool des pixels tissulaires (masque Otsu
sur la saturation) de ~10 lames HES, on fait une NMF 3 composantes **ancrée** sur une base HES
canonique (sinon la NMF est non-identifiable sur le cône HES), et on assigne H/E/S par la
permutation la plus proche de cette base (cosinus). Un seul artefact site-spécifique : un
fichier JSON de calibration.

**Cible médiane de site (2.3, recommandée).** Plutôt qu'une cible calibrée sur un petit pool,
calibrer chaque lame (un JSON source par lame) puis prendre la médiane : `chroma.site_target(calibs)`
(vecteurs, I0 et stats médians ; aucune lame relue). Sur notre site, une cible calibrée sur 10
lames avait un vecteur hématoxyline tiré vers le rouge : après transfert, noyaux bleu-vert.

## Utilisation

```python
import numpy as np
from PIL import Image
import chroma

# 1) calibrer une CIBLE sur ~10 vignettes RGB de lames HES (une fois par site/scanner)
target = chroma.auto_calibrate([np.asarray(Image.open(p).convert("RGB")) for p in thumbs])
norm = chroma.CHROMA(target)

# 2) pour chaque lame à normaliser : calibrer sa SOURCE et fabriquer la fonction PIL→PIL
src = chroma.auto_calibrate([np.asarray(Image.open("lame.png").convert("RGB"))])
stain_fn = chroma.make_stain_fn(src, norm)

normalisee = stain_fn(Image.open("patch.png"))     # PIL.Image RGB normalisée

# 2bis) transfert rang 3 (2.3) : exige `stats_full` dans la source et la cible
#       (écrit par auto_calibrate ≥ 2.3 ; site_target le calcule aussi)
stain_fn3 = chroma.make_stain_fn(src, norm, rank3=True)

# 3) décomposition en 7 canaux (maps 2D)
canaux = norm.decompose(np.asarray(Image.open("patch.png").convert("RGB")))
H, E, S, K, T, Sat = (canaux[k] for k in ("H", "E", "S", "K", "T", "Sat"))
```

`stain_fn` se branche directement comme `torchvision.transforms.Lambda` dans un pipeline
d'embeddings. Coût < 1 s par patch 224×224 sur CPU.

### `normalize()` ou `transfer()` ?

`normalize()` (défaut) travaille sur la chromaticité, gris retiré (rang 2). H⊥ et S⊥ y sont
presque opposés : sur les pixels très sombres leurs concentrations s'annulent, et des gains
différents par colorant cassent l'annulation → noyaux verts ou cyan. `transfer()`
(`rank3=True`) décompose en rang 3 plein (`c = NNLS(S_source, OD)`), applique les gains
`stats_full` cible/source, recompose sur les colorants cible, garde K, et revient au pixel
natif en fondu là où le capteur sature (OD max de 1,2 à 1,8).

Banc (400 patchs, 40 lames, vers la cible médiane de site) — part des patchs avec plus de
0,5 % de pixels cyan nouveaux / dispersion inter-lames de la couleur moyenne du tissu en Lab :

| | cyan | dispersion L / a* / b* |
|---|---|---|
| natif | — | 6,8 / 5,1 / 6,1 |
| `normalize()` | 1,0 % | 6,0 / 9,4 / 5,3 |
| `transfer()` | 0,2 % | 5,0 / 6,7 / 3,9 |

Les deux augmentent la dispersion inter-lames sur a* : les gains par lame mêlent l'intensité
du bain de coloration et la composition du tissu (une lame riche en éosine voit son E
abaissé). `normalize()` reste le défaut pour que les embeddings existants restent comparables.

Self-check intégré :

```bash
python chroma.py     # colorants purs, K hors-cône, ratios d'épaisseur, identité, transfert rang 3
```

## Format de calibration (JSON)

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

## Dépendances

`numpy`, `scikit-learn`, `opencv-python`, `Pillow` (voir `requirements.txt`). Pas de framework,
pas d'ORM, pas de GPU requis.

## Auteurs

- **Rémi Mathevet** — fœtopathologiste — conception, spécification, validation clinique
- **Claude** (Anthropic) — implémentation, décomposition gris-séparée, outillage QC

## Licence

Apache-2.0
