"""
Paso 10 v2 — Curvas de nivel robustas, modelo de la depresión foveal y confianza
================================================================================
Extiende macula_level_curves.py (v1) sin modificarlo. Cambios:

  1. SUPRESIÓN DE VASOS antes de trazar las curvas. Los vasos son oscuros
     y delgados; la fóvea es una depresión ancha. Un cierre morfológico en
     escala de grises con un disco más ancho que los vasos y mucho más
     pequeño que la fóvea los rellena sin tocar la depresión foveal. Así
     las regiones {gris <= v} dejan de fugarse por los vasos.

  2. CENTRO ROBUSTO. En v1 el centro es UN píxel (el de mínima intensidad
     cruda dentro del área del paso 8). En v2 es el mínimo de la imagen sin
     vasos y suavizada dentro de la misma área; se marca si cae en el borde
     del área (señal de que el área del paso 8 está mal ubicada).

  3. FAMILIA DE REGIONES ISO-NIVEL. Para cada nivel v, la región conexa de
     {gris <= v} que contiene el centro (igual que v1), con área, radio
     equivalente r_eq = sqrt(A/π), circularidad y solidez.

  4. PERFIL RADIAL SIN DIRECCIÓN PREFERENTE. v1 usa una sola recta
     horizontal. v2 usa la mediana de 72 rayos (perfil azimutal) y el
     perfil área–nivel (v frente a r_eq(v)).

  5. MODELO DE LA DEPRESIÓN FOVEAL. Al perfil azimutal se le ajusta
         I(r) = B + c·r − D·exp(−r² / 2σ²)
     (depresión gaussiana sobre un fondo con pendiente lineal). Da tres
     descriptores interpretables: profundidad D (niveles de gris), ancho σ
     y radio a media profundidad r50 = σ·sqrt(2·ln 2). El CONTORNO DE MEDIA
     PROFUNDIDAD es la curva de nivel cuyo nivel es I(r50): une el modelo
     con las curvas de nivel y es el contorno principal de v2.
     Se conserva además el codo (RDP + mayor ángulo de giro, mismo criterio
     de v1) sobre el perfil área–nivel, para comparar con v1.

  6. CONFIANZA. R² del ajuste, dispersión robusta de σ entre 8 sectores,
     acuerdo entre r50 del modelo y r_eq del contorno de media profundidad,
     circularidad y solidez de ese contorno, profundidad mínima y centro en
     el borde. Umbrales heurísticos: hay que calibrarlos con fóvea anotada.

  7. ESCALA FÍSICA APROXIMADA. Con el campo de la cámara (STARE: TopCon
     TRV-50, 35°) y 288 µm por grado (Drasdo y Fowler, 1974):
     µm/px = campo_grados · 288 / diámetro_FOV_px. Aproximación lineal
     válida cerca del polo posterior; la normalización por diámetro papilar
     del protocolo MACULA-AI la reemplazará.

Uso:
  python3 macula_level_curves_v2.py                 # todas las imágenes de la raíz
  python3 macula_level_curves_v2.py --input im0299.ppm
  python3 macula_level_curves_v2.py --fov_deg 35 --channel green
  python3 macula_level_curves_v2.py --vessel_kernel 0   # ablación: sin suprimir vasos
"""

import argparse
import csv
import os
import warnings

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

from macula_min_intensity import compute_macula_center
from macula_level_curves import (
    analyze as analyze_v1, discover_images, knee_contour_level,
)
from experiment_contours import best_macular_contour, contour_shape


UM_PER_DEG = 288.0          # µm de retina por grado visual (Drasdo y Fowler, 1974)
HALF = np.sqrt(2 * np.log(2))   # r50 = σ·HALF (radio a media profundidad)

# Radios anatómicos de referencia (µm), Kolb, Webvision.
ANATOMY_RADII_UM = {
    "foveola (Ø 0,35 mm)": 175.0,
    "ZAF (Ø 0,5 mm)": 250.0,
    "fóvea (Ø 1,5 mm)": 750.0,
}

# Umbrales heurísticos de confianza (a calibrar con fóvea anotada)
CONF_RULES = {
    "r2_high": 0.98, "r2_low": 0.95,
    "circ_high": 0.60, "circ_low": 0.40,
    "solid_high": 0.85, "solid_low": 0.70,
    "cv_high": 0.30, "cv_low": 0.50,
    "agree_high": 0.25,
    "depth_min": 5.0,
}


# ════════════════════════════════════════════════════════════════
# Preprocesado
# ════════════════════════════════════════════════════════════════
def odd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def fov_diameter_px(fov):
    ys, xs = np.where(fov > 0)
    return float(max(xs.max() - xs.min(), ys.max() - ys.min()))


def physical_scale_um_per_px(fov, fov_deg):
    """µm por píxel a partir del campo angular de la cámara (aprox.)."""
    if not fov_deg:
        return None
    return fov_deg * UM_PER_DEG / fov_diameter_px(fov)


def suppress_vessels(gray, kernel_px):
    """Cierre morfológico en gris: rellena estructuras oscuras más estrechas
    que `kernel_px` (vasos) y conserva depresiones anchas (fóvea)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)


def analysis_images(img, channel, vessel_kernel_px, smooth_sigma):
    """Devuelve (gris crudo, gris sin vasos y suavizado) en float32."""
    if channel == "green":
        gray = img[:, :, 1].copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base = suppress_vessels(gray, vessel_kernel_px) if vessel_kernel_px else gray
    work = cv2.GaussianBlur(base.astype(np.float32), (0, 0), smooth_sigma)
    return gray.astype(np.float32), work


def retina_mask(work, fov):
    """Mayor componente conexo de {gris > 30} dentro del FOV (como en v1)."""
    retina = ((work > 30) & (fov > 0)).astype(np.uint8)
    retina = cv2.morphologyEx(
        retina, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    num, lbl = cv2.connectedComponents(retina)
    if num > 1:
        sizes = [np.count_nonzero(lbl == i) for i in range(1, num)]
        retina = (lbl == 1 + int(np.argmax(sizes))).astype(np.uint8)
    return retina


def robust_center(work, area_mask, sigma=4.0, border_px=3):
    """Mínimo de la imagen sin vasos y suavizada dentro del área del paso 8.
    Marca si el mínimo cae en el borde del área."""
    sm = cv2.GaussianBlur(work, (0, 0), sigma)
    m = area_mask > 0
    vals = np.where(m, sm, np.inf)
    y, x = np.unravel_index(int(np.argmin(vals)), vals.shape)
    inner = cv2.erode(m.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (2 * border_px + 1,) * 2))
    return (int(y), int(x)), not bool(inner[y, x])


# ════════════════════════════════════════════════════════════════
# Familia de regiones iso-nivel y perfiles
# ════════════════════════════════════════════════════════════════
def iso_region_family(work, center, retina, roi_radius, close_k=5):
    """Regiones conexas {work <= v} que contienen el centro, para cada
    nivel entero v, hasta que la región toca el borde de la ROI."""
    cy, cx = center
    h, w = work.shape
    roi = np.zeros((h, w), np.uint8)
    cv2.circle(roi, (cx, cy), roi_radius, 1, -1)
    retina_e = cv2.erode(retina, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)))
    inside = (roi > 0) & (retina_e > 0)
    field = work.copy()
    field[~inside] = 255.0

    ring = np.zeros((h, w), np.uint8)
    cv2.circle(ring, (cx, cy), roi_radius, 1, 3)
    kern = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            if close_k >= 3 else None)

    v0 = int(np.ceil(field[cy, cx]))
    v1 = int(np.floor(field[inside].max()))
    fam = []
    for v in range(v0, v1 + 1):
        mask = (field <= v).astype(np.uint8)
        if kern is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        lab = labels[cy, cx]
        if lab == 0:
            continue
        comp = (labels == lab).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = float(np.count_nonzero(comp))
        per = float(cv2.arcLength(c, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(c)))
        ys, xs = np.nonzero(comp)
        fam.append({
            "level": v,
            "area": area,
            "r_eq": float(np.sqrt(area / np.pi)),
            "circ": float(min(4 * np.pi * area / per ** 2, 1.0)) if per > 0 else 0.0,
            "solidity": float(min(area / hull_area, 1.0)) if hull_area > 0 else 0.0,
            "centroid": (float(ys.mean()), float(xs.mean())),
            "touches_roi": bool(np.any(comp & ring)),
            "contour": c.reshape(-1, 2),
        })
        if fam[-1]["touches_roi"]:
            break
    return fam


def azimuthal_profiles(work, center, roi_radius, n_rays=72):
    """Intensidad sobre n_rays rayos desde el centro (bilineal).
    Devuelve radios (px), ángulos (rad) y matriz [rayo, radio]."""
    cy, cx = center
    radii = np.arange(0, roi_radius + 1, dtype=np.float32)
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False).astype(np.float32)
    map_x = (cx + np.outer(np.cos(ang), radii)).astype(np.float32)
    map_y = (cy + np.outer(np.sin(ang), radii)).astype(np.float32)
    prof = cv2.remap(work, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return radii, ang, prof


# ════════════════════════════════════════════════════════════════
# Modelo de la depresión foveal
# ════════════════════════════════════════════════════════════════
def depression_model(r, B, D, s, c):
    """Fondo B con pendiente c, menos una depresión gaussiana de
    profundidad D y ancho s."""
    return B + c * r - D * np.exp(-r ** 2 / (2 * s ** 2))


def fit_depression(r, y):
    """Ajuste por mínimos cuadrados con cotas. Devuelve dict o None."""
    r = np.asarray(r, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    r, y = r[ok], y[ok]
    if len(r) < 10:
        return None
    tail = y[-max(5, len(y) // 10):].mean()
    p0 = [tail, max(tail - y[:3].mean(), 1.0), len(r) / 4, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p, _ = curve_fit(depression_model, r, y, p0=p0,
                             bounds=([0, 0, 2, -1], [300, 255, 1.5 * r.max(), 1]),
                             maxfev=20000)
        except (RuntimeError, ValueError):
            return None
    res = y - depression_model(r, *p)
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(res ** 2) / ss if ss > 0 else 0.0
    B, D, s, c = map(float, p)
    return {"B": B, "D": D, "sigma": s, "c": c, "r50": s * HALF,
            "r2": float(r2), "rmse": float(np.sqrt(np.mean(res ** 2)))}


def sector_fits(radii, ang, prof, n_sectors=8):
    """σ ajustado al perfil mediano de cada sector angular."""
    width = 2 * np.pi / n_sectors
    out = []
    for s in range(n_sectors):
        sel = (ang >= s * width) & (ang < (s + 1) * width)
        f = fit_depression(radii, np.nanmedian(prof[sel], axis=0))
        out.append(f["sigma"] if f else np.nan)
    return np.array(out)


def robust_cv(x):
    """Dispersión robusta: 1,4826·MAD / mediana (un sector atípico no la arrastra)."""
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)) / med) if med > 0 else np.nan


# ════════════════════════════════════════════════════════════════
# Codo (mismo criterio que v1) con margen
# ════════════════════════════════════════════════════════════════
def _rdp(pts, eps):
    """Ramer–Douglas–Peucker igual al de v1, pero con el producto cruz 2D
    calculado a mano: np.cross con vectores 2D está obsoleto en NumPy 2 y
    falla en versiones recientes."""
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    line = end - start
    L = np.hypot(line[0], line[1])
    rel = pts - start
    if L == 0:
        d = np.hypot(rel[:, 0], rel[:, 1])
    else:
        d = np.abs(line[0] * rel[:, 1] - line[1] * rel[:, 0]) / L
    idx = int(np.argmax(d))
    if d[idx] > eps:
        left = _rdp(pts[:idx + 1], eps)
        right = _rdp(pts[idx:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def knee_with_margin(x, y, eps_frac=0.035):
    """Aproximación por rectas (RDP) + vértice de mayor ángulo de giro, con
    la regla de v1 (se ignoran el primer y el último vértice interior).
    Devuelve además el margen = ángulo del codo / segundo mayor ángulo."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return None
    xr = max(x.max() - x.min(), 1e-9)
    yr = max(y.max() - y.min(), 1e-9)
    Pn = np.column_stack([(x - x.min()) / xr, (y - y.min()) / yr])
    eps = eps_frac
    for _ in range(6):
        Vn = _rdp(Pn, eps)
        if len(Vn) >= 4 or eps <= 0.005:
            break
        eps /= 1.6
    M = len(Vn)
    if M < 3:
        return None
    angle = np.zeros(M)
    for k in range(1, M - 1):
        a, b = Vn[k] - Vn[k - 1], Vn[k + 1] - Vn[k]
        angle[k] = abs(np.arctan2(a[0] * b[1] - a[1] * b[0],
                                  a[0] * b[0] + a[1] * b[1]))
    cand = list(range(1, M - 1))
    if len(cand) > 2:
        cand = cand[1:-1]
    order = sorted(cand, key=lambda j: angle[j], reverse=True)
    k = order[0]
    second = angle[order[1]] if len(order) > 1 else 0.0
    verts = np.column_stack([Vn[:, 0] * xr + x.min(), Vn[:, 1] * yr + y.min()])
    return {"r": float(verts[k, 0]), "y": float(verts[k, 1]),
            "angle_deg": float(np.degrees(angle[k])),
            "margin": float(angle[k] / second) if second > 0 else float("inf"),
            "verts": verts}


def family_member_at(fam, key, value):
    """Región de la familia cuyo `key` (r_eq o level) es el más cercano a value."""
    if not fam or value is None or not np.isfinite(value):
        return None
    return min(fam, key=lambda f: abs(f[key] - value))


def confidence_label(m):
    R = CONF_RULES
    reasons = []
    if m["center_on_border"]:
        reasons.append("centro en el borde del área del paso 8")
    if not np.isfinite(m["fit_r2"]) or m["fit_r2"] < R["r2_low"]:
        reasons.append("el perfil no sigue una depresión foveal (R² bajo)")
    if not np.isfinite(m["fit_D"]) or m["fit_D"] < R["depth_min"]:
        reasons.append("depresión casi nula")
    if m["half_circ"] < R["circ_low"] or m["half_solidity"] < R["solid_low"]:
        reasons.append("contorno de media profundidad irregular")
    if not np.isfinite(m["sector_cv"]) or m["sector_cv"] > R["cv_low"]:
        reasons.append("σ muy distinto entre sectores")
    if reasons:
        return "baja", reasons
    high = (m["fit_r2"] >= R["r2_high"] and m["half_circ"] >= R["circ_high"]
            and m["half_solidity"] >= R["solid_high"]
            and m["sector_cv"] <= R["cv_high"]
            and m["half_agreement"] <= R["agree_high"])
    if high:
        return "alta", []
    why = []
    if m["fit_r2"] < R["r2_high"]:
        why.append(f"R²={m['fit_r2']:.3f}")
    if m["half_circ"] < R["circ_high"]:
        why.append(f"circ={m['half_circ']:.2f}")
    if m["half_solidity"] < R["solid_high"]:
        why.append(f"solidez={m['half_solidity']:.2f}")
    if m["sector_cv"] > R["cv_high"]:
        why.append(f"dispersión σ entre sectores={m['sector_cv']:.2f}")
    if m["half_agreement"] > R["agree_high"]:
        why.append(f"desacuerdo modelo/contorno={m['half_agreement']:.2f}")
    return "media", why


# ════════════════════════════════════════════════════════════════
# Análisis completo
# ════════════════════════════════════════════════════════════════
def analyze_v2(img, channel="green", roi_frac=0.30, smooth_sigma=2.0,
               vessel_kernel_px=None, center_sigma=4.0, close_k=5,
               fov_deg=None, n_rays=72):
    h, w = img.shape[:2]
    if vessel_kernel_px is None:
        vessel_kernel_px = odd(min(h, w) / 40)          # 15 px en STARE

    # Pasos 1–9 de v1: área macular del paso 8 y centro por píxel mínimo
    pre = compute_macula_center(img, channel=channel)
    fov = pre["fov"]
    area_mask = pre["macula_area_mask"]
    center_v1 = pre["new_center"]

    gray, work = analysis_images(img, channel, vessel_kernel_px, smooth_sigma)
    retina = retina_mask(work, fov)
    ys_r, xs_r = np.where(retina > 0)
    ocular_radius = 0.5 * max(ys_r.max() - ys_r.min(), xs_r.max() - xs_r.min())
    roi_radius = int(round(roi_frac * ocular_radius))

    center, on_border = robust_center(work, area_mask, sigma=center_sigma)

    fam = iso_region_family(work, center, retina, roi_radius, close_k=close_k)
    r_eq = np.array([f["r_eq"] for f in fam])
    lev = np.array([f["level"] for f in fam], float)

    radii, ang, prof = azimuthal_profiles(work, center, roi_radius, n_rays)
    az_med = np.nanmedian(prof, axis=0)
    az_q1 = np.nanpercentile(prof, 25, axis=0)
    az_q3 = np.nanpercentile(prof, 75, axis=0)

    fit = fit_depression(radii, az_med)
    sect = sector_fits(radii, ang, prof)
    sector_cv = robust_cv(sect)

    # Contorno de media profundidad: curva de nivel con nivel = I(r50)
    half_region = None
    if fit is not None:
        half_level = depression_model(fit["r50"], fit["B"], fit["D"],
                                      fit["sigma"], fit["c"])
        half_region = family_member_at(fam, "level", half_level)
    half_agree = (abs(half_region["r_eq"] - fit["r50"]) / fit["r50"]
                  if half_region is not None else np.inf)

    knee_area = knee_with_margin(r_eq, lev)
    knee_region = family_member_at(fam, "r_eq", knee_area["r"]) if knee_area else None

    scale = physical_scale_um_per_px(fov, fov_deg)
    nan = float("nan")
    m = {
        "center_v1": center_v1,
        "center": center,
        "center_shift_px": float(np.hypot(center[0] - center_v1[0],
                                          center[1] - center_v1[1])),
        "center_on_border": on_border,
        "n_levels": len(fam),
        "fit_B": fit["B"] if fit else nan,
        "fit_D": fit["D"] if fit else nan,
        "fit_sigma": fit["sigma"] if fit else nan,
        "fit_c": fit["c"] if fit else nan,
        "fit_r50": fit["r50"] if fit else nan,
        "fit_r2": fit["r2"] if fit else nan,
        "sector_sigmas": sect,
        "sector_cv": sector_cv,
        "half_level": half_region["level"] if half_region else None,
        "half_r_eq": half_region["r_eq"] if half_region else nan,
        "half_circ": half_region["circ"] if half_region else 0.0,
        "half_solidity": half_region["solidity"] if half_region else 0.0,
        "half_agreement": float(half_agree),
        "knee_area_r": knee_area["r"] if knee_area else nan,
        "knee_area_margin": knee_area["margin"] if knee_area else nan,
        "knee_circ": knee_region["circ"] if knee_region else nan,
        "um_per_px": scale,
    }
    m["confidence"], m["confidence_reasons"] = confidence_label(m)

    return {
        "metrics": m, "fam": fam, "r_eq": r_eq, "lev": lev,
        "radii": radii, "az_med": az_med, "az_q1": az_q1, "az_q3": az_q3,
        "fit": fit, "half_region": half_region,
        "knee_area": knee_area, "knee_region": knee_region,
        "gray": gray, "work": work, "fov": fov, "area_mask": area_mask,
        "roi_radius": roi_radius, "ocular_radius": ocular_radius,
        "vessel_kernel_px": vessel_kernel_px,
    }


# ════════════════════════════════════════════════════════════════
# Visualización
# ════════════════════════════════════════════════════════════════
CONF_COLOR = {"alta": "#16a34a", "media": "#ca8a04", "baja": "#dc2626"}


def _closed(c):
    return np.vstack([c, c[0]])


def _v1_knee_contour(v1):
    lvl = knee_contour_level(v1["knee"], v1["lvl"])
    if lvl is None:
        return None, None
    lut = {int(lv): c for lv, c in v1["curves"]}
    if lvl not in lut:
        av = np.array(sorted(lut))
        lvl = int(av[np.argmin(np.abs(av - lvl))])
    return lvl, lut[lvl]


def plot_image_panel(img, v1, v2, out_path):
    """(A) original con regiones v2 y contornos de v1/v2, (B) verde sin vasos,
    (C) perfiles radiales + modelo, (D) σ por sector."""
    m = v2["metrics"]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cy, cx = m["center"]
    vy, vx = m["center_v1"]
    roi = v2["roi_radius"]
    win = int(roi * 1.15)
    s = m["um_per_px"]
    unit = " µm" if s else " px"

    def to_u(r):
        return np.asarray(r, float) * s if s else np.asarray(r, float)

    fam = v2["fam"]
    norm = (Normalize(vmin=min(f["level"] for f in fam), vmax=max(f["level"] for f in fam))
            if fam else None)
    cmap = cm.turbo
    _, kc1 = _v1_knee_contour(v1)
    best1 = best_macular_contour(v1)

    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1], projection="polar")

    for ax, base, cmap_b, title in (
        (axA, rgb, None, "(A) Original — curvas de nivel v2"),
        (axB, v2["work"], "gray",
         f"(B) Canal verde sin vasos (cierre {v2['vessel_kernel_px']} px) + suavizado"),
    ):
        ax.imshow(base, cmap=cmap_b)
        for f in fam[::2]:
            ax.plot(*_closed(f["contour"]).T, color=cmap(norm(f["level"])),
                    lw=0.6, alpha=0.8)
        if kc1 is not None:
            ax.plot(*_closed(kc1).T, color="white", lw=1.1, ls="--",
                    label="v1: contorno del codo (recta horizontal)")
        if best1 is not None:
            ax.plot(*_closed(best1["c"]).T, color="#22d3ee", lw=1.5,
                    label="v1: mejor contorno (más circular)")
        if v2["half_region"] is not None:
            ax.plot(*_closed(v2["half_region"]["contour"]).T, color="#16a34a",
                    lw=3.0, label="v2: contorno de media profundidad")
        if v2["fit"] is not None:
            ax.add_patch(plt.Circle((cx, cy), v2["fit"]["r50"], color="#16a34a",
                                    fill=False, lw=1.0, ls=":",
                                    label="v2: círculo r50 del modelo"))
        ax.add_patch(plt.Circle((cx, cy), roi, color="magenta", fill=False,
                                lw=1.0, label="ROI (30 % del radio ocular)"))
        ax.scatter(vx, vy, s=90, marker="o", facecolor="none", edgecolor="white",
                   lw=1.5, zorder=6, label="centro v1 (píxel mínimo)")
        ax.scatter(cx, cy, s=180, marker="*", color="#16a34a", edgecolor="white",
                   lw=0.8, zorder=7, label="centro v2 (robusto)")
        ax.set_xlim(cx - win, cx + win)
        ax.set_ylim(cy + win, cy - win)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    axA.legend(loc="lower left", fontsize=7.3, framealpha=0.85)

    # (C) Perfiles radiales + modelo
    r = v2["radii"]
    axC.fill_between(to_u(r), v2["az_q1"], v2["az_q3"], color="#94a3b8", alpha=0.25,
                     label="v2: rango intercuartílico de 72 rayos")
    axC.plot(to_u(r), v2["az_med"], color="#334155", lw=1.8,
             label="v2: mediana de 72 rayos (sin vasos)")
    axC.plot(to_u(v2["r_eq"]), v2["lev"], color="#0891b2", lw=1.4,
             label="v2: área–nivel (v frente a r_eq)")
    axC.plot(to_u(v1["dist"]), v1["y_mean"], color="#dc2626", lw=1.0, alpha=0.8,
             label="v1: recta horizontal (con vasos)")
    fit = v2["fit"]
    if fit is not None:
        axC.plot(to_u(r), depression_model(r, fit["B"], fit["D"], fit["sigma"], fit["c"]),
                 color="#16a34a", lw=2.4, ls="--",
                 label=f"modelo: D={fit['D']:.1f}, σ={to_u(fit['sigma']):.0f}{unit}, "
                       f"R²={fit['r2']:.3f}")
        hl = depression_model(fit["r50"], fit["B"], fit["D"], fit["sigma"], fit["c"])
        axC.scatter([to_u(fit["r50"])], [hl], s=260, marker="*", color="#16a34a",
                    edgecolor="black", zorder=7,
                    label=f"r50 (media profundidad) = {to_u(fit['r50']):.0f}{unit}")
    if v1["knee"] is not None:
        axC.axvline(to_u(v1["knee"]["dist"]), color="#dc2626", ls=":", lw=1.2,
                    label=f"codo v1 = {to_u(v1['knee']['dist']):.0f}{unit}")
    if s:
        top = axC.get_ylim()[1]
        for name, rad in ANATOMY_RADII_UM.items():
            axC.axvline(rad, color="#a16207", ls="--", lw=0.8, alpha=0.7)
            axC.text(rad, top, f" {name}", rotation=90, va="top", ha="right",
                     fontsize=7.5, color="#a16207")
    axC.set_xlabel("Distancia al centro (µm, escala aproximada)" if s
                   else "Distancia al centro (px)")
    axC.set_ylabel("Intensidad (canal verde)")
    axC.set_title("(C) Perfiles radiales y modelo de la depresión foveal", fontsize=11)
    axC.grid(True, ls=":", alpha=0.5)
    axC.legend(loc="lower right", fontsize=7.3)

    # (D) σ por sector
    sk = m["sector_sigmas"]
    th = (np.arange(len(sk)) + 0.5) * 2 * np.pi / len(sk)
    axD.bar(th, to_u(np.nan_to_num(sk)), width=2 * np.pi / len(sk) * 0.85,
            color="#16a34a", alpha=0.6, edgecolor="black")
    axD.set_theta_direction(-1)
    axD.set_theta_offset(0)
    axD.set_title(f"(D) Ancho σ de la depresión por sector (8 × 45°)\n"
                  f"dispersión robusta = {m['sector_cv']:.2f}", fontsize=11, pad=14)

    scale_txt = f"{s:.1f} µm/px (aprox.)" if s else "sin escala física"
    r50_txt = (f"r50={to_u(m['fit_r50']):.0f}{unit}" if np.isfinite(m["fit_r50"])
               else "sin ajuste")
    fig.suptitle(
        f"Curvas de nivel v2 — confianza {m['confidence'].upper()}  ·  {r50_txt}  ·  "
        f"R²={m['fit_r2']:.3f}  ·  contorno: circ={m['half_circ']:.2f}, "
        f"solidez={m['half_solidity']:.2f}  ·  {scale_txt}",
        fontsize=13, fontweight="bold", color=CONF_COLOR[m["confidence"]])
    if m["confidence_reasons"]:
        fig.text(0.5, 0.955, "Motivos: " + "; ".join(m["confidence_reasons"]),
                 ha="center", fontsize=10, color=CONF_COLOR[m["confidence"]])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_grid(rows, out_path, window=130):
    """Una fila por imagen: v1 (mejor contorno + codo) frente a v2."""
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.9 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (stem, img, v1, v2) in enumerate(rows):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        m = v2["metrics"]
        s = m["um_per_px"]
        _, kc1 = _v1_knee_contour(v1)
        best1 = best_macular_contour(v1)

        ax = axes[i, 0]
        cy1, cx1 = v1["center"]
        ax.imshow(rgb)
        if kc1 is not None:
            ax.plot(*_closed(kc1).T, color="#dc2626", lw=1.2)
        if best1 is not None:
            ax.plot(*_closed(best1["c"]).T, color="#22d3ee", lw=2.4)
        ax.scatter(cx1, cy1, s=90, marker="*", color="yellow", edgecolor="black", zorder=5)
        ax.set_xlim(cx1 - window, cx1 + window)
        ax.set_ylim(cy1 + window, cy1 - window)
        circ1 = f"circ={best1['circ']:.2f}" if best1 else ""
        ax.set_title(f"{stem} — v1  ({circ1})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[i, 1]
        cy, cx = m["center"]
        ax.imshow(rgb)
        if v2["half_region"] is not None:
            ax.plot(*_closed(v2["half_region"]["contour"]).T, color="#16a34a", lw=2.6)
        ax.scatter(cx, cy, s=110, marker="*", color="#16a34a", edgecolor="white", zorder=5)
        ax.set_xlim(cx - window, cx + window)
        ax.set_ylim(cy + window, cy - window)
        r_txt = (f"r50={m['fit_r50'] * s:.0f} µm, " if s and np.isfinite(m["fit_r50"]) else "")
        ax.set_title(f"{stem} — v2  ({r_txt}circ={m['half_circ']:.2f}, "
                     f"confianza {m['confidence']})", fontsize=10,
                     color=CONF_COLOR[m["confidence"]])
        ax.set_xticks([])
        ax.set_yticks([])
    h = [plt.Line2D([], [], color="#22d3ee", lw=2.4, label="v1: mejor contorno"),
         plt.Line2D([], [], color="#dc2626", lw=1.2, label="v1: contorno del codo"),
         plt.Line2D([], [], color="#16a34a", lw=2.6, label="v2: contorno de media profundidad")]
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Curvas de nivel: v1 frente a v2 (canal verde)",
                 fontsize=14, fontweight="bold", y=1.012)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Salidas
# ════════════════════════════════════════════════════════════════
SUMMARY_FIELDS = [
    "imagen", "confianza", "motivos", "um_por_px_aprox",
    "centro_v1_y", "centro_v1_x", "centro_v2_y", "centro_v2_x",
    "desplazamiento_centro_px", "centro_en_borde",
    "v1_codo_r_px", "v1_codo_circ", "v1_mejor_r_px", "v1_mejor_circ",
    "v2_modelo_profundidad_D", "v2_modelo_sigma_px", "v2_modelo_r50_px",
    "v2_modelo_r50_um", "v2_modelo_pendiente_c", "v2_modelo_R2",
    "v2_dispersion_sigma_sectores",
    "v2_contorno_nivel", "v2_contorno_r_eq_px", "v2_contorno_circ",
    "v2_contorno_solidez", "v2_desacuerdo_modelo_contorno",
    "v2_codo_area_nivel_r_px", "v2_codo_area_nivel_circ", "v2_niveles",
]


def _num(x, nd=2):
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    return round(x, nd) if np.isfinite(x) else ""


def summary_row(stem, v1, v2):
    m = v2["metrics"]
    s = m["um_per_px"]
    _, kc1 = _v1_knee_contour(v1)
    k1_r, k1_c = (contour_shape(kc1)[2:] if kc1 is not None else (None, None))
    best1 = best_macular_contour(v1)
    return {
        "imagen": stem, "confianza": m["confidence"],
        "motivos": "; ".join(m["confidence_reasons"]),
        "um_por_px_aprox": _num(s),
        "centro_v1_y": m["center_v1"][0], "centro_v1_x": m["center_v1"][1],
        "centro_v2_y": m["center"][0], "centro_v2_x": m["center"][1],
        "desplazamiento_centro_px": _num(m["center_shift_px"], 1),
        "centro_en_borde": int(m["center_on_border"]),
        "v1_codo_r_px": _num(k1_r, 1), "v1_codo_circ": _num(k1_c, 3),
        "v1_mejor_r_px": _num(best1["r_eq"], 1) if best1 else "",
        "v1_mejor_circ": _num(best1["circ"], 3) if best1 else "",
        "v2_modelo_profundidad_D": _num(m["fit_D"], 1),
        "v2_modelo_sigma_px": _num(m["fit_sigma"], 1),
        "v2_modelo_r50_px": _num(m["fit_r50"], 1),
        "v2_modelo_r50_um": _num(m["fit_r50"] * s, 0) if s else "",
        "v2_modelo_pendiente_c": _num(m["fit_c"], 3),
        "v2_modelo_R2": _num(m["fit_r2"], 4),
        "v2_dispersion_sigma_sectores": _num(m["sector_cv"], 3),
        "v2_contorno_nivel": m["half_level"],
        "v2_contorno_r_eq_px": _num(m["half_r_eq"], 1),
        "v2_contorno_circ": _num(m["half_circ"], 3),
        "v2_contorno_solidez": _num(m["half_solidity"], 3),
        "v2_desacuerdo_modelo_contorno": _num(m["half_agreement"], 3),
        "v2_codo_area_nivel_r_px": _num(m["knee_area_r"], 1),
        "v2_codo_area_nivel_circ": _num(m["knee_circ"], 3),
        "v2_niveles": m["n_levels"],
    }


def write_family_csv(fam, path, scale):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["nivel", "area_px", "r_eq_px", "r_eq_um_aprox", "circularidad",
                    "solidez", "centroide_y", "centroide_x", "toca_roi"])
        for f in fam:
            w.writerow([f["level"], int(f["area"]), round(f["r_eq"], 2),
                        round(f["r_eq"] * scale, 1) if scale else "",
                        round(f["circ"], 4), round(f["solidity"], 4),
                        round(f["centroid"][0], 1), round(f["centroid"][1], 1),
                        int(f["touches_roi"])])


def write_profile_csv(v2, path, scale):
    fit = v2["fit"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["r_px", "r_um_aprox", "mediana_72_rayos", "q1", "q3", "modelo"])
        for i, r in enumerate(v2["radii"]):
            mod = (depression_model(r, fit["B"], fit["D"], fit["sigma"], fit["c"])
                   if fit else np.nan)
            w.writerow([int(r), round(float(r) * scale, 1) if scale else "",
                        _num(v2["az_med"][i], 3), _num(v2["az_q1"][i], 3),
                        _num(v2["az_q3"][i], 3), _num(mod, 3)])


def run_all(images, out_dir, channel, fov_deg, vessel_kernel_px):
    os.makedirs(out_dir, exist_ok=True)
    rows, table = [], []
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] No se pudo leer {path}")
            continue
        print(f"\n{'=' * 60}\n[IMG] {stem}\n{'=' * 60}")
        v1 = analyze_v1(img, channel=channel)
        v2 = analyze_v2(img, channel=channel, fov_deg=fov_deg,
                        vessel_kernel_px=vessel_kernel_px)
        m = v2["metrics"]
        d = os.path.join(out_dir, stem)
        os.makedirs(d, exist_ok=True)
        plot_image_panel(img, v1, v2, os.path.join(d, f"v2_{stem}.png"))
        write_family_csv(v2["fam"], os.path.join(d, f"v2_familia_iso_niveles_{stem}.csv"),
                         m["um_per_px"])
        write_profile_csv(v2, os.path.join(d, f"v2_perfil_radial_{stem}.csv"), m["um_per_px"])
        table.append(summary_row(stem, v1, v2))
        rows.append((stem, img, v1, v2))
        s = m["um_per_px"] or 1.0
        print(f"  centro v1 {m['center_v1']} → v2 {m['center']} "
              f"(Δ={m['center_shift_px']:.0f}px, en borde={m['center_on_border']})")
        print(f"  modelo: D={m['fit_D']:.1f}, σ={m['fit_sigma']:.1f}px, "
              f"r50={m['fit_r50'] * s:.0f}{' µm' if m['um_per_px'] else ' px'}, "
              f"R²={m['fit_r2']:.3f}, dispersión σ={m['sector_cv']:.2f}")
        print(f"  contorno media profundidad: circ={m['half_circ']:.2f}, "
              f"solidez={m['half_solidity']:.2f}, desacuerdo={m['half_agreement']:.2f}")
        print(f"  → confianza {m['confidence']}"
              + (f" ({'; '.join(m['confidence_reasons'])})" if m["confidence_reasons"] else ""))

    if not table:
        return
    with open(os.path.join(out_dir, "resumen_v1_vs_v2.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(table)
    plot_comparison_grid(rows, os.path.join(out_dir, "comparacion_v1_vs_v2.png"))
    print(f"\n[✓] Resumen:     {os.path.join(out_dir, 'resumen_v1_vs_v2.csv')}")
    print(f"[✓] Comparación: {os.path.join(out_dir, 'comparacion_v1_vs_v2.png')}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Curvas de nivel v2 (robustas).")
    pa.add_argument("--input", help="Imagen única; si se omite, todas las de --input_dir.")
    pa.add_argument("--input_dir", default=".")
    pa.add_argument("--output_dir", default="results_v2")
    pa.add_argument("--channel", default="green", choices=["luminance", "green"])
    pa.add_argument("--fov_deg", type=float, default=35.0,
                    help="Campo angular de la cámara en grados (STARE: 35). 0 = sin escala física.")
    pa.add_argument("--vessel_kernel", type=int, default=None,
                    help="Diámetro (px) del cierre que suprime vasos. 0 = sin supresión.")
    a = pa.parse_args()

    if a.input:
        imgs = [a.input]
    else:
        imgs = [os.path.join(a.input_dir, f) for f in discover_images(a.input_dir)]
    if not imgs:
        raise SystemExit(f"[ERROR] No hay imágenes en {a.input_dir}")
    run_all(imgs, a.output_dir, a.channel, a.fov_deg or None, a.vessel_kernel)
    print("[✓] Completado.")
