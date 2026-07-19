#!/usr/bin/env python3
"""CHROMA — normaliseur colorimétrique HES par décomposition CMYK en densité optique.

Calibrated Histological Reconstruction Of Multiplexed Absorbances. Les 3 colorants HES
(Hématoxyline/Éosine/Safran) = 3 primaires soustractives (C/M/Y) ; le passage en densité
optique (OD = -log10(I/I0), loi de Beer-Lambert) rend le mélange additif → déconvolution
linéaire par matrice de stain S (3x3, inversible). Canal K = résiduel non normalisé
(préserve méconium/hémosidérine, avantage clé sur Macenko).

Calibration AUTOMATIQUE par NMF (aucune annotation, décision Prefect chroma 2026-07-18) :
pool de pixels tissulaires OD sur ~10 lames HES → NMF 3 comp → vecteurs H/E/S assignés par
canal d'absorption dominant. Un fichier JSON de calib par site.

API :
    calib = auto_calibrate([thumb_rgb, ...])          # dict cible (sérialisable JSON)
    src   = auto_calibrate([thumb_rgb_de_la_lame])    # calib source d'UNE lame
    norm  = CHROMA(calib)
    fn    = make_stain_fn(src, norm)                  # PIL->PIL, branché dans embed_multimag
"""
import hashlib
import json

import numpy as np

__version__ = "2.0"

# Axe gris unitaire : direction [1,1,1] en OD = variation d'épaisseur/brightness sans
# couleur. On la retire de l'OD avant décomposition pour que le safran ne capte pas le
# gris (Test 2 spec : séparation brightness/safran).
GRAY = np.ones(3) / np.sqrt(3.0)


def _gray_removed(S):
    """Matrice de stain (colonnes H/E/S) projetée hors de l'axe gris : S⊥ = S − ĝ(ĝᵀS).
    Chaque colonne devient purement chromatique (somme des composantes = 0)."""
    return S - np.outer(GRAY, GRAY @ S)


def calib_id(calib: dict) -> str:
    """Empreinte courte (8 hex) de la matrice de calib — identifie sa version."""
    sv = calib["stain_vectors"]
    key = json.dumps([calib["I0"], sv["H"], sv["E"], sv["S"]], sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
#  Calibration automatique (NMF) — pas d'annotation
# ---------------------------------------------------------------------------

def _otsu_tissue(thumb):
    """Masque tissu (True=tissu) sur la saturation HSV, seuil Otsu assoupli."""
    import cv2
    sat = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[:, :, 1]
    t, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return sat >= max(1, int(t * 0.4))


def auto_calibrate(thumbs, n_pixels_per=500_000, n_nmf_runs=3, seed=0):
    """Calibration NMF sur une liste de vignettes RGB uint8 (HES routine uniquement).

    Retourne dict {I0, stain_vectors{H,E,S}, stats{mu,sigma}, meta}. `thumbs` = liste de
    ndarray (H,W,3) uint8 déjà chargés (découplé de l'I/O lame).
    """
    rng = np.random.default_rng(seed)

    # --- I0 = verre (fond clair) : non-tissu ET brillant. Le >180 écarte le padding
    # noir des vignettes OMNISSIAH (tuiles manquantes = zéros), sinon I0 s'effondre. ---
    bg = []
    for t in thumbs:
        m = _otsu_tissue(t)
        b = t[(~m) & (t.max(2) > 180)]
        if len(b):
            bg.append(b.astype(np.float64))
    I0 = np.concatenate(bg).mean(0) if bg else np.array([240., 240., 240.])

    # --- pool de pixels tissulaires en OD (+ brightness/saturation OD gris-axe) ---
    # ĝ = axe gris unitaire. T (brightness) = od·ĝ = épaisseur tissulaire ;
    # Sat = ‖od − T·ĝ‖ = intensité chromatique (perpendiculaire au gris). Spec CHROMA.
    gray_axis = np.ones(3) / np.sqrt(3.0)
    pool, bright, sat = [], [], []
    for t in thumbs:
        m = _otsu_tissue(t)
        px = t[m].astype(np.float64)
        if len(px) > n_pixels_per:
            px = px[rng.choice(len(px), n_pixels_per, replace=False)]
        od = -np.log10(np.clip(px, 1.0, None) / I0)
        T = od @ gray_axis
        bright.append(T)
        sat.append(np.linalg.norm(od - T[:, None] * gray_axis, axis=1))
        od = od[np.linalg.norm(od, axis=1) > 0.15]     # vire les quasi-blancs
        if len(od):
            pool.append(od)
    pool = np.concatenate(pool)
    bright, sat = np.concatenate(bright), np.concatenate(sat)

    # --- NMF ANCRÉE sur base HES canonique. Sans seed, la NMF est non identifiable
    # sur un cône HES : elle tombe sur des axes RGB (H=bleu pur), curseurs S/K morts
    # (dégénérescence constatée 2026-07-18, invariante à la diversité du pool). Le seed
    # custom la garde dans le bassin physique tout en s'ajustant au site. ---
    from sklearn.decomposition import non_negative_factorization
    H0 = np.array([[0.65, 0.70, 0.29],    # H hématoxyline (absorbe bleu)
                   [0.09, 0.99, 0.11],    # E éosine (absorbe vert)
                   [0.10, 0.21, 0.97]])   # S safran (absorbe rouge)
    H0 /= np.linalg.norm(H0, axis=1, keepdims=True)
    X = np.clip(pool, 0, None)
    W0 = np.clip(X @ np.linalg.pinv(H0), 0, None)
    _, best_H, _ = non_negative_factorization(
        X, n_components=3, init="custom", H=H0.copy(), W=W0,
        update_H=True, max_iter=500, random_state=seed)
    best_err = float(np.linalg.norm(X - W0 @ best_H))  # proxy erreur reconstruction
    vecs = [v / np.linalg.norm(v) for v in best_H]     # 3 vecteurs unitaires

    # --- assignation H/E/S par canal d'absorption relatif dominant ---
    # ratio[i,c] = poids du canal c du vecteur i / somme des 2 autres canaux
    ratios = np.array([[vecs[i][c] / max(sum(vecs[i][j] for j in range(3) if j != c), 1e-10)
                        for c in range(3)] for i in range(3)])
    rem = list(range(3)); assign = {}
    for name, ch in (("H", 2), ("E", 1), ("S", 0)):     # H~bleu(B), E~vert(G), S~rouge(R)
        k = int(np.argmax(ratios[rem, ch]))
        assign[name] = vecs[rem[k]]; rem.pop(k)

    # --- stats des concentrations (mu, sigma) sur la CHROMATICITÉ (gris retiré) ---
    S = np.column_stack([assign["H"], assign["E"], assign["S"]])
    S_chroma = _gray_removed(S)
    chroma_pool = pool - (pool @ GRAY)[:, None] * GRAY
    c = np.clip(chroma_pool @ np.linalg.pinv(S_chroma).T, 0, None)
    out = {
        "method": "NMF_auto_chroma_separated",
        "chroma_version": __version__,
        "I0": I0.tolist(),
        "stain_vectors": {k: v.tolist() for k, v in assign.items()},
        "stats": {"mu": c.mean(0).tolist(), "sigma": (c.std(0) + 1e-6).tolist(),
                  "brightness_mu": float(bright.mean()), "brightness_sigma": float(bright.std() + 1e-6),
                  "sat_mu": float(sat.mean()), "sat_sigma": float(sat.std() + 1e-6)},
        "n_slides": len(thumbs),
        "n_pixels": int(len(pool)),
        "nmf_reconstruction_error": float(best_err),
    }
    out["calib_id"] = calib_id(out)
    return out


# ---------------------------------------------------------------------------
#  Normaliseur
# ---------------------------------------------------------------------------

def _channels_from_od(od, S):
    """od (N,3) → (T, Sat, c, K). Sépare la brightness AVANT la décomposition :
      T   = od·ĝ                     brightness (épaisseur tissulaire), scalaire/pixel
      Sat = ‖od − T·ĝ‖               intensité chromatique
      c   = pinv(S⊥)·(od − T·ĝ) ≥ 0  concentrations sur la chromaticité (gris retiré)
      K   = od − (T·ĝ + S⊥·c)        résiduel non expliqué (pigments : méconium/sidérine)
    S⊥ = S gray-removed est de rang 2 : c est la solution min-norm 2D (brightness portée
    par T, pas par les colorants) → le safran ne s'allume plus sur le gris. Reconstruction
    T·ĝ + S⊥·c + K = od exacte."""
    S_chroma = _gray_removed(S)
    T = od @ GRAY
    chroma = od - T[:, None] * GRAY
    Sat = np.linalg.norm(chroma, axis=1)
    c = np.clip(chroma @ np.linalg.pinv(S_chroma).T, 0, None)
    K = od - (T[:, None] * GRAY + c @ S_chroma.T)
    return T, Sat, c, K


class CHROMA:
    def __init__(self, target_calib: dict):
        sv = target_calib["stain_vectors"]
        st = target_calib["stats"]
        self.I0_target = np.asarray(target_calib["I0"], float)
        self.S_target = np.column_stack([np.asarray(sv[k], float) for k in ("H", "E", "S")])
        self.S_target_chroma = _gray_removed(self.S_target)
        self.mu_target = np.asarray(st["mu"], float)
        self.sigma_target = np.asarray(st["sigma"], float)
        # stats brightness/saturation (optionnelles : absentes des calibs v1)
        self.bmu = st.get("brightness_mu")
        self.bsig = st.get("brightness_sigma")

    @staticmethod
    def _od(img, I0):
        return -np.log10(np.clip(img.astype(np.float64), 1.0, None) / I0)

    def decompose(self, img, I0=None):
        """img (H,W,3) uint8 → dict des 7 canaux (H,E,S,K,T,Sat + od), maps 2D."""
        I0 = self.I0_target if I0 is None else np.asarray(I0, float)
        h, w = img.shape[:2]
        od = self._od(img, I0).reshape(-1, 3)
        T, Sat, c, K = _channels_from_od(od, self.S_target)
        return {"H": c[:, 0].reshape(h, w), "E": c[:, 1].reshape(h, w),
                "S": c[:, 2].reshape(h, w), "K": np.linalg.norm(K, axis=1).reshape(h, w),
                "T": T.reshape(h, w), "Sat": Sat.reshape(h, w),
                "od": od.reshape(h, w, 3)}

    get_channels = decompose

    def normalize(self, img, I0_source, S_source, mu_source, sigma_source,
                  brightness_source=None, gains=None):
        """img RGB uint8 → RGB uint8 normalisé vers la cible.

        Brightness (T) et chromaticité (concentrations) normalisées SÉPARÉMENT :
          - concentrations : z-score src→cible (contrôle la saturation par colorant) ;
          - brightness T : match affine src→cible si stats dispo (sinon T inchangé) ;
          - résiduel K (pigments) : INCHANGÉ.
        gains = (gH,gE,gS,gK) multiplient concentrations + K avant recomposition (défaut 1)."""
        h, w = img.shape[:2]
        od = self._od(img, I0_source).reshape(-1, 3)
        T, _, c_pos, K = _channels_from_od(od, S_source)
        c_norm = np.clip((c_pos - mu_source) / sigma_source * self.sigma_target + self.mu_target,
                         0, None)
        if brightness_source is not None and self.bmu is not None:
            bmu_s, bsig_s = brightness_source
            T = (T - bmu_s) / bsig_s * self.bsig + self.bmu
        if gains is not None:
            c_norm = c_norm * np.asarray(gains[:3], float)
            K = K * float(gains[3])
        od_norm = T[:, None] * GRAY + c_norm @ self.S_target_chroma.T + K
        rgb = self.I0_target * (10.0 ** (-od_norm.reshape(h, w, 3)))
        return np.clip(rgb, 0, 255).astype(np.uint8)


def make_stain_fn(source_calib: dict, norm: CHROMA):
    """Fabrique un callable PIL→PIL bindé à la calib source d'une lame (branché comme
    transforms.Lambda dans embed_multimag._build_transform)."""
    from PIL import Image
    sv = source_calib["stain_vectors"]
    st = source_calib["stats"]
    I0_s = np.asarray(source_calib["I0"], float)
    S_s = np.column_stack([np.asarray(sv[k], float) for k in ("H", "E", "S")])
    mu_s = np.asarray(st["mu"], float)
    sig_s = np.asarray(st["sigma"], float)
    bright_s = (st["brightness_mu"], st["brightness_sigma"]) if "brightness_mu" in st else None
    return lambda im: Image.fromarray(norm.normalize(np.asarray(im.convert("RGB")),
                                                     I0_s, S_s, mu_s, sig_s,
                                                     brightness_source=bright_s))


def _demo():
    """Self-check V2 : reconstruction T·ĝ+S⊥·c+K exacte (normalize src==cible ≈ identité),
    gray tissu → concentrations ≈ 0 (décontamination), gains=0 → sortie grise."""
    rng = np.random.default_rng(0)
    S = np.column_stack([[0.18, 0.20, 0.90],   # H (abs. bleu)
                         [0.10, 0.90, 0.30],   # E (abs. vert)
                         [0.90, 0.35, 0.20]])   # S (abs. rouge)
    S /= np.linalg.norm(S, axis=0)
    c_true = rng.uniform(0, 1.2, size=(64, 64, 3))
    I0 = np.array([240., 240., 240.])
    od = c_true.reshape(-1, 3) @ S.T
    img = np.clip(I0 * 10.0 ** (-od.reshape(64, 64, 3)), 0, 255).astype(np.uint8)

    calib = {"I0": I0.tolist(),
             "stain_vectors": {"H": S[:, 0].tolist(), "E": S[:, 1].tolist(), "S": S[:, 2].tolist()},
             "stats": {"mu": [0., 0., 0.], "sigma": [1., 1., 1.]}}
    norm = CHROMA(calib)

    # 1) décontamination : un patch gris pur (od ∝ ĝ) → concentrations ≈ 0, brightness > 0
    gray_img = np.full((16, 16, 3), 150, np.uint8)
    ch = norm.decompose(gray_img, I0)
    assert max(abs(ch["H"]).max(), abs(ch["E"]).max(), abs(ch["S"]).max()) < 1e-6, "gris contamine HES"
    assert ch["T"].mean() > 0.1, "brightness du gris devrait être > 0"

    # 2) reconstruction : normalize(src==cible) ≈ identité (K rattrape le résiduel)
    out = norm.normalize(img, I0, S, np.zeros(3), np.ones(3))
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 2.0, "identité off"
    out_g1 = norm.normalize(img, I0, S, np.zeros(3), np.ones(3), gains=(1, 1, 1, 1))
    assert np.array_equal(out_g1, out), "gains=(1,1,1,1) doit être identité"

    # 3) gains=0 → plus de chroma ni pigment, il reste la brightness grise (R≈G≈B)
    out_g0 = norm.normalize(img, I0, S, np.zeros(3), np.ones(3), gains=(0, 0, 0, 0))
    spread = out_g0.astype(float).std(axis=2).mean()
    assert spread < 2.0, f"gains=0 doit donner du gris (spread={spread:.2f})"
    print("chroma _demo OK — décontamination gris, reconstruction ≈ identité, gains OK")


if __name__ == "__main__":
    _demo()
