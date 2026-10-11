#!/usr/bin/env python3
"""CHROMA — normaliseur colorimétrique HES par décomposition CMYK en densité optique.

Calibrated Histological Reconstruction Of Multiplexed Absorbances. Les 3 colorants HES
(Hématoxyline/Éosine/Safran) = 3 primaires soustractives (C/M/Y) ; le passage en densité
optique (OD = -log10(I/I0), loi de Beer-Lambert) rend le mélange additif → déconvolution
linéaire par matrice de stain S (3x3, inversible). Canal K = résiduel HORS-CÔNE, non
normalisé (à mesurer sur méconium natif avant d'en faire un argument : un brun est un
mélange POSITIF de H+E+S, donc dans le cône → K ne s'allume pas dessus).

Deux chemins distincts, à ne pas confondre :
  - `normalize()`  : gris séparé (S⊥, rang 2) — robuste à la dérive d'exposition/I0 ;
  - `decompose()`  : rang 3 plein — concentrations identifiables et K défini.

Calibration AUTOMATIQUE par NMF (aucune annotation, décision Prefect chroma 2026-07-18) :
pool de pixels tissulaires OD sur ~10 lames HES → NMF 3 comp → vecteurs H/E/S appariés par
cosinus à la base canonique H0. Un fichier JSON de calib par site.

API :
    calib = auto_calibrate([thumb_rgb, ...])          # dict cible (sérialisable JSON)
    src   = auto_calibrate([thumb_rgb_de_la_lame])    # calib source d'UNE lame
    norm  = CHROMA(calib)
    fn    = make_stain_fn(src, norm)                  # PIL->PIL, branché dans embed_multimag
"""
import hashlib
import json
from itertools import combinations, permutations

import numpy as np

__version__ = "2.2"

# Axe gris unitaire : direction [1,1,1] en OD = variation d'épaisseur/brightness sans
# couleur. On la retire de l'OD avant décomposition pour que le safran ne capte pas le
# gris (Test 2 spec : séparation brightness/safran).
GRAY = np.ones(3) / np.sqrt(3.0)

# Pénalité de norme du NNLS sur le chemin NORMALISATION seulement (voir _nnls3). Effet
# identique de 0,003 à 0,03 ; écart aux pixels sains 0,06 niveau de gris. La calibration
# (auto_calibrate) reste à 0 : les 2214 calibs et les cibles ne bougent pas.
LAM_NORM = 0.01


def _gray_removed(S):
    """Matrice de stain (colonnes H/E/S) projetée hors de l'axe gris : S⊥ = S − ĝ(ĝᵀS).
    Chaque colonne devient purement chromatique (somme des composantes = 0)."""
    return S - np.outer(GRAY, GRAY @ S)


def _nnls3(x, S, lam=0.0):
    """NNLS exact batché : argmin_c≥0 ‖x − S·cᵀ‖ pour 3 colonnes (x (N,3), S (3,3)).
    L'optimum non-négatif à 3 variables appartient à l'un des 7 ensembles actifs non
    vides ou à la solution nulle ; on les énumère (déterministe, vectorisé, sans SciPy).
    Sur S⊥ (rang 2) la solution min-norm à 3 actifs sort souvent négative → écartée, et
    le NNLS retombe sur une combinaison 1-2 colonnes propre (calme la sur-sat. de E).

    lam > 0 : choix de l'ensemble actif sur ‖résidu‖² + lam·‖c‖². Sur S⊥, H⊥ et S⊥ sont
    presque opposés : un bruit chromatique perpendiculaire à leur axe (pixels quasi blancs)
    n'est atteint EXACTEMENT que par c_H, c_S ~30x le signal qui s'annulent, et des gains
    différents par colorant cassent l'annulation → mouchetures cyan (7 % des patchs avec la
    cible v2, Prefect 2026-10-06). Avec lam la solution approchée l'emporte, le reste part
    dans K (recopié tel quel par normalize). lam = 0 : comportement historique."""
    x = np.asarray(x, np.float64)
    best_c = np.zeros((len(x), 3))
    best_err = np.einsum("ij,ij->i", x, x)
    for size in (1, 2, 3):
        for sub in combinations(range(3), size):
            A = S[:, sub]
            coeff = x @ np.linalg.pinv(A, rcond=1e-10).T
            feas = np.all(coeff >= -1e-10, axis=1)
            if not feas.any():
                continue
            coeff = np.maximum(coeff, 0.0)
            res = x - coeff @ A.T
            err = np.einsum("ij,ij->i", res, res) + lam * np.einsum("ij,ij->i", coeff, coeff)
            imp = feas & (err < best_err)
            if imp.any():
                best_c[imp] = 0.0
                for j, s in enumerate(sub):
                    best_c[imp, s] = coeff[imp, j]
                best_err[imp] = err[imp]
    return best_c


def _robust_loc_scale(a, axis=0):
    """Médiane + échelle robuste (1.4826·MAD ≈ σ pour du gaussien). Résiste aux
    pixels sombres/pigments qui gonflent mean/std. Pour T/Sat (non zéro-inflés)."""
    med = np.median(a, axis=axis)
    scale = 1.4826 * np.median(np.abs(a - med), axis=axis) + 1e-6
    return med, scale


def _conc_stats(c):
    """loc/scale robustes par colorant, sur le SUPPORT POSITIF. Les concentrations
    sont zéro-inflées (la plupart des pixels n'ont pas un colorant donné) : médiane
    et MAD GLOBALES s'effondrent à 0 → le canal serait annulé à la normalisation. On
    estime donc médiane/MAD sur c[:,j] > eps (le lobe « colorant présent »)."""
    med = np.zeros(c.shape[1]); scale = np.zeros(c.shape[1])
    for j in range(c.shape[1]):
        pos = c[c[:, j] > 1e-4, j]
        if len(pos) < 16:
            med[j], scale[j] = 0.0, 1e-6
        else:
            med[j] = np.median(pos)
            scale[j] = 1.4826 * np.median(np.abs(pos - med[j])) + 1e-6
    return med, scale


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


def auto_calibrate(thumbs, n_pixels_per=500_000, seed=0):
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
    # 95e percentile (verre le plus clair), pas la moyenne : robuste au tissu clair
    # résiduel mal masqué. Borné [180,255].
    I0 = (np.clip(np.percentile(np.concatenate(bg), 95, axis=0), 180., 255.)
          if bg else np.array([240., 240., 240.]))

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
    H0 = np.array([[0.65, 0.70, 0.29],    # H hématoxyline : OD dominante R+G (paraît bleu)
                   [0.09, 0.99, 0.11],    # E éosine       : OD dominante G   (paraît magenta)
                   [0.10, 0.21, 0.97]])   # S safran        : OD dominante B   (paraît jaune)
    H0 /= np.linalg.norm(H0, axis=1, keepdims=True)
    X = np.clip(pool, 0, None)
    W0 = np.clip(X @ np.linalg.pinv(H0), 0, None)
    best_W, best_H, _ = non_negative_factorization(
        X, n_components=3, init="custom", H=H0.copy(), W=W0,
        update_H=True, max_iter=500, random_state=seed)
    # RMSE par pixel sur le W AJUSTÉ (pas W0 : le H bouge, le W initial ne le suit pas
    # → ×18 mesuré) et normalisée par la taille du pool, sinon deux lames de pool
    # différent ne sont pas comparables (Frobenius ∝ √N).
    best_err = float(np.linalg.norm(X - best_W @ best_H) / np.sqrt(X.size))
    vecs = np.array([v / np.linalg.norm(v) for v in best_H])   # 3 vecteurs unitaires

    # --- assignation H/E/S : permutation qui maximise le cosinus à la base canonique H0.
    # On matche le VECTEUR entier (pas un seul canal) → robuste, l'inversion H↔S d'un
    # argmax mono-canal devient structurellement impossible. ---
    best_perm = min(permutations(range(3)),
                    key=lambda p: float(np.sum(1.0 - np.sum(vecs[list(p)] * H0, axis=1))))
    assign = {name: vecs[best_perm[i]] for i, name in enumerate(("H", "E", "S"))}

    # --- stats des concentrations (mu, sigma) sur la CHROMATICITÉ (gris retiré) ---
    # NNLS exact (comme la décompo) + loc/scale ROBUSTES (médiane, MAD) : mu/sigma
    # portent médiane et échelle robuste, mêmes clés JSON (schéma inchangé).
    S = np.column_stack([assign["H"], assign["E"], assign["S"]])
    S_chroma = _gray_removed(S)
    chroma_pool = pool - (pool @ GRAY)[:, None] * GRAY
    c = _nnls3(chroma_pool, S_chroma)
    c_med, c_scale = _conc_stats(c)
    b_med, b_scale = _robust_loc_scale(bright)
    s_med, s_scale = _robust_loc_scale(sat)
    out = {
        "method": "NMF_auto_chroma_separated",
        "chroma_version": __version__,
        "I0": I0.tolist(),
        "stain_vectors": {k: v.tolist() for k, v in assign.items()},
        "stats": {"mu": c_med.tolist(), "sigma": c_scale.tolist(),
                  "brightness_mu": float(b_med), "brightness_sigma": float(b_scale),
                  "sat_mu": float(s_med), "sat_sigma": float(s_scale)},
        "n_slides": len(thumbs),
        "n_pixels": int(len(pool)),
        "nmf_reconstruction_error": float(best_err),
    }
    out["calib_id"] = calib_id(out)
    return out


# ---------------------------------------------------------------------------
#  Normaliseur
# ---------------------------------------------------------------------------

def _channels_rank3(od, S):
    """od (N,3) → (c, K) en RANG 3 PLEIN, sans retirer le gris. c = NNLS(S, od) ≥ 0,
    K = od − S·c = résidu hors-cône HES (pigments : méconium, hémosidérine).

    Pourquoi pas S⊥ ici : S⊥ (rang 2) engendre TOUT le plan chromatique où vit le signal,
    son cône couvre le plan → K ≡ 0 (mesuré 1e-16 sur tissu réel, méconium compris) et
    aucun pixel ne peut porter les 3 colorants à la fois. Le cône simplicial de S (rang 3)
    est un sous-ensemble strict de R³ : un absorbeur hors-cône y laisse un vrai résidu.
    L'invariance d'épaisseur s'obtient par les ratios (od = t·S·c les laisse inchangés),
    pas par la projection. Chemin décomposition sémantique uniquement — la normalisation
    garde `_channels_from_od` (gris séparé), robuste à la dérive d'exposition."""
    c = _nnls3(od, S)
    return c, od - c @ S.T


def _channels_from_od(od, S):
    """od (N,3) → (T, Sat, c, K). Chemin NORMALISATION. Sépare la brightness AVANT la
    décomposition :
      T   = od·ĝ                     brightness (épaisseur tissulaire), scalaire/pixel
      Sat = ‖od − T·ĝ‖               intensité chromatique
      c   = NNLS(S⊥, od − T·ĝ) ≥ 0   concentrations sur la chromaticité (gris retiré)
      K   = od − (T·ĝ + S⊥·c)        résiduel non expliqué (pigments : méconium/sidérine)
    S⊥ = S gray-removed est de rang 2. NNLS exact (pas pinv+clip) : la brightness reste
    portée par T (le safran ne s'allume pas sur le gris) et la non-négativité est résolue
    proprement → moins de sur-saturation du canal E. Reconstruction T·ĝ + S⊥·c + K = od."""
    S_chroma = _gray_removed(S)
    T = od @ GRAY
    chroma = od - T[:, None] * GRAY
    Sat = np.linalg.norm(chroma, axis=1)
    c = _nnls3(chroma, S_chroma, lam=LAM_NORM)
    K = od - (T[:, None] * GRAY + c @ S_chroma.T)
    return T, Sat, c, K


class CHROMA:
    def __init__(self, target_calib: dict, z_clip: float = 6.0):
        sv = target_calib["stain_vectors"]
        st = target_calib["stats"]
        self.I0_target = np.asarray(target_calib["I0"], float)
        self.S_target = np.column_stack([np.asarray(sv[k], float) for k in ("H", "E", "S")])
        self.S_target_chroma = _gray_removed(self.S_target)
        self.mu_target = np.asarray(st["mu"], float)
        self.sigma_target = np.asarray(st["sigma"], float)
        self.z_clip = float(z_clip)     # borne le z-score src→cible (stats robustes)
        # stats brightness/saturation (optionnelles : absentes des calibs v1)
        self.bmu = st.get("brightness_mu")
        self.bsig = st.get("brightness_sigma")

    @staticmethod
    def _od(img, I0):
        return -np.log10(np.clip(img.astype(np.float64), 1.0, None) / I0)

    def decompose(self, img, I0=None):
        """img (H,W,3) uint8 → dict des canaux (H,E,S,K,T,Sat + ratios + od), maps 2D.

        Rang 3 plein (gris NON retiré) : concentrations identifiables et K = vrai résidu
        hors-cône. T et Sat restent fournis comme covariables descriptives, plus comme
        quelque chose qu'on soustrait avant."""
        I0 = self.I0_target if I0 is None else np.asarray(I0, float)
        h, w = img.shape[:2]
        od = self._od(img, I0).reshape(-1, 3)
        c, K = _channels_rank3(od, self.S_target)
        T = od @ GRAY
        Sat = np.linalg.norm(od - T[:, None] * GRAY, axis=1)
        H, E, S = c[:, 0], c[:, 1], c[:, 2]
        # plancher physique au dénominateur (~10% d'une concentration présente typique
        # ≈0.13) : borne les ratios là où le colorant de référence est absent (sinon
        # H/E explose à ~1e9 sur les 9% de pixels sans éosine). Ratio ceiling ≈ 1/eps.
        eps = 1e-2
        return {"H": H.reshape(h, w), "E": E.reshape(h, w), "S": S.reshape(h, w),
                "K": np.linalg.norm(K, axis=1).reshape(h, w),
                "K_vector": K.reshape(h, w, 3),   # résiduel signé → direction (méconium vs sidérine)
                "T": T.reshape(h, w), "Sat": Sat.reshape(h, w),
                "Cellularite": (H / (E + eps)).reshape(h, w),      # noyau/cytoplasme
                "Fibrose": (S / (H + E + eps)).reshape(h, w),      # proportion collagène
                "od": od.reshape(h, w, 3)}

    get_channels = decompose

    def normalize(self, img, I0_source, S_source, mu_source, sigma_source,
                  brightness_source=None, gains=None):
        """img RGB uint8 → RGB uint8 normalisé vers la cible.

        Brightness (T) et chromaticité (concentrations) normalisées SÉPARÉMENT :
          - concentrations : gain multiplicatif médiane src→cible (voir plus bas) ;
          - brightness T : match affine src→cible si stats dispo (sinon T inchangé) ;
          - résiduel K (pigments) : INCHANGÉ.
        gains = (gH,gE,gS,gK) multiplient concentrations + K avant recomposition (défaut 1).

        Concentrations : gain multiplicatif médiane source→cible par canal
        (c·mu_target/mu_source), PAS z-score affine. Le z-score clippait à 0 les canaux
        très en-dessous de la médiane de leur lame (éosine effacée sur les membranes
        méconium d'une lame éosine-riche) → destruction de signal. Le ratio préserve les
        canaux faibles. sigma_source ignoré (gardé dans la signature pour les appelants)."""
        h, w = img.shape[:2]
        od = self._od(img, I0_source).reshape(-1, 3)
        T, _, c_pos, K = _channels_from_od(od, S_source)
        # gain borné [1/z_clip, z_clip] contre l'amplification quand mu_source→0 ;
        # mu_source≈0 → gain 1 (identité, cf _demo mu=0).
        gain = np.divide(self.mu_target, mu_source,
                         out=np.ones_like(np.asarray(mu_source, float)),
                         where=np.asarray(mu_source, float) > 1e-6)
        gain = np.clip(gain, 1.0 / self.z_clip, self.z_clip)
        c_norm = c_pos * gain
        if brightness_source is not None and self.bmu is not None:
            bmu_s, bsig_s = brightness_source
            T = (T - bmu_s) / bsig_s * self.bsig + self.bmu
        if gains is not None:
            c_norm = c_norm * np.asarray(gains[:3], float)
            K = K * float(gains[3])
        od_norm = T[:, None] * GRAY + c_norm @ self.S_target_chroma.T + K
        rgb = self.I0_target * (10.0 ** (-od_norm.reshape(h, w, 3)))
        return np.clip(rgb, 0, 255).astype(np.uint8)


def apply_calib(rgb_uint8, source_calib: dict, norm: CHROMA, gains=None):
    """RGB uint8 → RGB uint8 normalisé, à partir d'une calib source (dict). Cœur partagé
    par make_stain_fn (série) et le worker CHROMA du pool CPU (stain_norm.worker_init)."""
    sv = source_calib["stain_vectors"]
    st = source_calib["stats"]
    I0_s = np.asarray(source_calib["I0"], float)
    S_s = np.column_stack([np.asarray(sv[k], float) for k in ("H", "E", "S")])
    mu_s = np.asarray(st["mu"], float)
    sig_s = np.asarray(st["sigma"], float)
    bright_s = (st["brightness_mu"], st["brightness_sigma"]) if "brightness_mu" in st else None
    return norm.normalize(rgb_uint8, I0_s, S_s, mu_s, sig_s, brightness_source=bright_s,
                          gains=gains)


def make_stain_fn(source_calib: dict, norm: CHROMA):
    """Fabrique un callable PIL→PIL bindé à la calib source d'une lame (branché comme
    transforms.Lambda dans embed_multimag._build_transform)."""
    from PIL import Image
    return lambda im: Image.fromarray(
        apply_calib(np.asarray(im.convert("RGB")), source_calib, norm))


def _demo():
    """Self-check. Les tests 1-3 portent sur `decompose` (rang 3) et PEUVENT échouer :
    une matrice de stain fausse ou permutée fait tomber le test 1."""
    rng = np.random.default_rng(0)
    # vecteurs d'ABSORBANCE (pas de couleur apparente) : H absorbe R+G donc paraît bleue,
    # E absorbe G donc paraît magenta, S absorbe B donc paraît jaune.
    S = np.column_stack([[0.65, 0.70, 0.29],   # H hématoxyline
                         [0.09, 0.99, 0.11],   # E éosine
                         [0.10, 0.21, 0.97]])  # S safran
    S /= np.linalg.norm(S, axis=0)
    c_true = rng.uniform(0, 1.2, size=(64, 64, 3))
    I0 = np.array([240., 240., 240.])
    od = c_true.reshape(-1, 3) @ S.T
    img = np.clip(I0 * 10.0 ** (-od.reshape(64, 64, 3)), 0, 255).astype(np.uint8)

    calib = {"I0": I0.tolist(),
             "stain_vectors": {"H": S[:, 0].tolist(), "E": S[:, 1].tolist(), "S": S[:, 2].tolist()},
             "stats": {"mu": [0., 0., 0.], "sigma": [1., 1., 1.]}}
    norm = CHROMA(calib)

    def patch(od_vec, n=16):
        return np.clip(I0 * 10.0 ** (-np.tile(od_vec, (n, n, 1))), 0, 255).astype(np.uint8)

    # 1) un colorant pur n'allume QUE son canal (c'est ce test qui attrape une inversion
    #    H↔S à l'assignation NMF, cf. bug 2026-07-19)
    for i, name in enumerate(("H", "E", "S")):
        ch = norm.decompose(patch(0.8 * S[:, i]), I0)
        others = [n_ for n_ in ("H", "E", "S") if n_ != name]
        assert abs(ch[name].mean() - 0.8) < 0.05, f"{name} pur : {name}={ch[name].mean():.3f} au lieu de 0.8"
        for o in others:
            assert ch[o].mean() < 0.02, f"{name} pur allume {o} ({ch[o].mean():.3f})"

    # 2) K = résidu hors-cône. Un mélange HES est DANS le cône → K≈0 ; un absorbeur
    #    achromatique (ĝ, pigment formolé/carbone) est hors-cône → K s'allume.
    k_hes = norm.decompose(patch(S @ np.array([0.5, 0.4, 0.3])), I0)["K"].mean()
    k_pig = norm.decompose(patch(1.0 * GRAY), I0)["K"].mean()
    assert k_hes < 0.02, f"K devrait être nul sur du HES pur (K={k_hes:.3f})"
    assert k_pig > 0.05, f"K devrait capter un absorbeur hors-cône (K={k_pig:.3f})"

    # 3) invariance d'épaisseur : od = t·S·c, les RATIOS ne doivent pas bouger (c'est par
    #    les ratios qu'on obtient l'invariance, pas par la projection hors du gris)
    base = S @ np.array([0.35, 0.30, 0.20])
    r1 = norm.decompose(patch(base), I0)["Cellularite"].mean()
    r2 = norm.decompose(patch(2 * base), I0)["Cellularite"].mean()
    assert abs(r1 - r2) / r1 < 0.15, f"ratio non invariant à l'épaisseur ({r1:.3f} vs {r2:.3f})"

    # 4) reconstruction : normalize(src==cible) ≈ identité (chemin gris-séparé, inchangé)
    out = norm.normalize(img, I0, S, np.zeros(3), np.ones(3))
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 2.0, "identité off"
    out_g1 = norm.normalize(img, I0, S, np.zeros(3), np.ones(3), gains=(1, 1, 1, 1))
    assert np.array_equal(out_g1, out), "gains=(1,1,1,1) doit être identité"

    # 5) gains=0 → plus de chroma ni pigment, il reste la brightness grise (R≈G≈B)
    out_g0 = norm.normalize(img, I0, S, np.zeros(3), np.ones(3), gains=(0, 0, 0, 0))
    spread = out_g0.astype(float).std(axis=2).mean()
    assert spread < 2.0, f"gains=0 doit donner du gris (spread={spread:.2f})"
    print("chroma _demo OK — colorants purs séparés, K hors-cône, ratios invariants, "
          "normalisation ≈ identité")


if __name__ == "__main__":
    _demo()
