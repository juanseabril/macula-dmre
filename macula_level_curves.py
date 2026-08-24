"""
Paso 10 — Curvas de nivel radiales desde el centro de mácula
============================================================
A partir del CENTRO DE MÁCULA (píxel de mínima intensidad del paso 9),
se traza una recta radial HORIZONTAL desde el centro hasta el máximo
posible dentro del FOV. Para la intensidad de cada punto de esa recta
se genera su CURVA DE NIVEL (iso-intensidad) en la imagen en escala de
grises, de forma que cada curva intersecta su punto de la recta.

Estrategia tipo Canny para cerrar los contornos:
  1. Suavizado gaussiano (1er paso de Canny) → curvas continuas.
  2. Umbral por histéresis: para el nivel v se toma la región
     {gris <= v} y se conserva SOLO la componente conexa que
     contiene el centro de mácula (crecimiento desde una semilla,
     análogo a la histéresis de Canny).
  3. Cierre morfológico (MORPH_CLOSE) para puentear huecos.
  4. El borde de esa región (findContours) es siempre un contorno
     CERRADO que pasa por el punto de la recta con intensidad v.

Uso:  python3 macula_level_curves.py --input im0299.ppm
"""

import argparse
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import dmre
from macula_min_intensity import compute_macula_center


def to_analysis_gray(img, channel="luminance"):
    """Imagen en gris del análisis: luminancia (BGR2GRAY) o canal verde."""
    if channel == "green":
        return img[:, :, 1].copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def horizontal_radial_line(center, retina, max_len):
    """
    Recta radial horizontal desde el centro de mácula, en la dirección
    de mayor longitud disponible, acotada a `max_len` px (ROI macular =
    30% del radio ocular).

    Returns
    -------
    xs : np.ndarray  columnas de la recta (ordenadas desde el centro)
    cy : int         fila (constante)
    """
    cy, cx = center
    row = (retina[cy] > 0).astype(np.int8)
    if row[cx] == 0:
        raise ValueError("El centro no cae dentro de la retina.")

    # Tramo CONTIGUO de retina que contiene el centro (evita píxeles sueltos)
    left = cx
    while left - 1 >= 0 and row[left - 1]:
        left -= 1
    right = cx
    while right + 1 < len(row) and row[right + 1]:
        right += 1

    len_right = right - cx
    len_left = cx - left

    if len_right >= len_left:
        end = min(right, cx + max_len)         # hacia la derecha, acotado
        xs = np.arange(cx, end + 1)
    else:
        end = max(left, cx - max_len)          # hacia la izquierda, acotado
        xs = np.arange(cx, end - 1, -1)
    return xs, cy


def isolevel_contour_through_point(work, level, fov, seed, close_k=5):
    """
    Curva de nivel cerrada (estrategia tipo Canny) para una intensidad.

    work  : gris suavizado con exterior del FOV puesto a 255.
    level : nivel de intensidad v.
    seed  : (y, x) centro de mácula (semilla de la histéresis).
    """
    cy, cx = seed

    # Umbral: región oscura <= v
    mask = (work <= level).astype(np.uint8)

    # Cierre morfológico → puentea huecos (cierra el contorno)
    if close_k >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Histéresis: conservar SOLO la componente conexa que contiene la semilla
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    lab = labels[cy, cx]
    if lab == 0:
        return None
    comp = (labels == lab).astype(np.uint8)

    # Borde de la región = contorno CERRADO que pasa por el punto de nivel v
    contours, _ = cv2.findContours(
        comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2)  # (x, y)


def generate_level_curves(img, center, fov, roi_frac=0.30,
                          smooth_sigma=2.0, close_k=5, channel="luminance"):
    """
    Genera una curva de nivel por cada punto de la recta radial horizontal,
    acotada a una ROI macular = `roi_frac` del radio ocular.

    channel : "luminance" (BGR2GRAY) o "green" (canal verde).
    """
    gray = to_analysis_gray(img, channel).astype(np.float32)

    # 1) Suavizado gaussiano (primer paso de Canny)
    gray_s = cv2.GaussianBlur(gray, (0, 0), smooth_sigma)

    # Máscara de RETINA real (el borde negro del FOV queda excluido).
    # Se usa el mayor componente conexo de {gris_suavizado > 30} dentro
    # del FOV para tener un límite fiable del fondo de ojo.
    retina = ((gray_s > 30) & (fov > 0)).astype(np.uint8)
    retina = cv2.morphologyEx(
        retina, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    num, lbl = cv2.connectedComponents(retina)
    if num > 1:
        biggest = 1 + np.argmax(
            [np.count_nonzero(lbl == i) for i in range(1, num)]
        )
        retina = (lbl == biggest).astype(np.uint8)

    # Radio ocular ≈ mitad del mayor lado del fondo de ojo; ROI = 30% de él
    ys_r, xs_r = np.where(retina > 0)
    ocular_radius = 0.5 * max(ys_r.max() - ys_r.min(), xs_r.max() - xs_r.min())
    roi_radius = int(round(roi_frac * ocular_radius))

    cyc, cxc = center
    roi = np.zeros(retina.shape, np.uint8)
    cv2.circle(roi, (cxc, cyc), roi_radius, 1, -1)

    # Exterior de la retina (con margen) y exterior de la ROI a 255 →
    # confina la búsqueda de contornos a la ROI macular
    retina_e = cv2.erode(
        retina, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    work = gray_s.copy()
    work[(retina_e == 0) | (roi == 0)] = 255.0

    xs, cy = horizontal_radial_line(center, retina, roi_radius)

    curves = []          # (level, contorno)
    line_levels = []     # nivel en cada punto de la recta
    seen = set()

    for x in xs:
        v = float(gray_s[cy, x])
        line_levels.append((x, v))
        key = int(round(v))
        if key in seen:                    # nivel ya trazado → mismo punto/curva
            continue
        seen.add(key)
        c = isolevel_contour_through_point(
            work, key, fov, center, close_k=close_k
        )
        if c is not None and len(c) >= 3:
            curves.append((key, c))

    curves.sort(key=lambda t: t[0])
    print(f"     Radio ocular≈{ocular_radius:.0f}px, "
          f"ROI={roi_frac*100:.0f}% → {roi_radius}px")
    return curves, xs, cy, line_levels, gray, roi_radius, ocular_radius


def compute_contour_intensities(gray, curves):
    """
    Intensidad promedio (y desviación) a lo largo de cada contorno,
    muestreando la imagen en escala de grises en los píxeles del contorno.

    Returns list de dicts ordenados de la fóvea hacia la periferia.
    """
    h, w = gray.shape
    rows = []
    for i, (level, c) in enumerate(curves, start=1):
        cx = np.clip(c[:, 0], 0, w - 1)
        cyv = np.clip(c[:, 1], 0, h - 1)
        vals = gray[cyv, cx]
        rows.append({
            "idx": i,
            "level": int(level),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "n": int(len(vals)),
        })
    return rows


def visualize_intensity_profile(rows, out_path):
    """
    Gráfica 2D: X = contornos etiquetados (C1..Cn, de fóvea a periferia),
    Y = intensidad promedio a lo largo de cada contorno.
    """
    idx = np.array([r["idx"] for r in rows])
    means = np.array([r["mean"] for r in rows])
    stds = np.array([r["std"] for r in rows])
    levels = np.array([r["level"] for r in rows], dtype=float)
    labels = [f"C{r['idx']}" for r in rows]

    norm = Normalize(vmin=levels.min(), vmax=levels.max())
    cmap = cm.turbo

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.42), 6.5))

    # Línea guía + banda de dispersión
    ax.plot(idx, means, "-", color="#64748b", linewidth=1.2, zorder=1)
    ax.fill_between(idx, means - stds, means + stds,
                    color="#94a3b8", alpha=0.20, zorder=0,
                    label="± 1σ a lo largo del contorno")

    # Marcadores coloreados por nivel de la curva
    ax.scatter(idx, means, c=levels, cmap=cmap, norm=norm,
               s=70, edgecolor="black", linewidths=0.6, zorder=3,
               label="Intensidad media")

    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Contorno  (C1 = fóvea  →  Cn = periferia)")
    ax.set_ylabel("Intensidad promedio a lo largo del contorno")
    ax.set_title("Intensidad promedio por curva de nivel",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Nivel de la curva (intensidad umbral)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def build_radial_function(line_levels, center, curves, gray):
    """
    Para cada punto de la recta horizontal devuelve su distancia al centro
    y el nivel de intensidad PROMEDIO del contorno que corta ese punto.

    Returns
    -------
    dist   : np.ndarray  distancia al centro (px) de cada punto
    y_mean : np.ndarray  intensidad media del contorno que corta el punto
    y_raw  : np.ndarray  intensidad del propio punto (referencia)
    lvl    : np.ndarray  nivel del contorno asignado a cada punto
    """
    rows = compute_contour_intensities(gray, curves)
    level_to_mean = {r["level"]: r["mean"] for r in rows}
    avail = np.array(sorted(level_to_mean.keys()))

    _, cx = center
    dist, y_mean, y_raw, lvl = [], [], [], []
    for x, v in line_levels:
        key = int(round(v))
        if key not in level_to_mean:               # nivel sin curva → el más cercano
            key = int(avail[np.argmin(np.abs(avail - key))])
        dist.append(abs(x - cx))
        y_mean.append(level_to_mean[key])
        y_raw.append(v)
        lvl.append(key)
    return (np.array(dist), np.array(y_mean),
            np.array(y_raw), np.array(lvl, dtype=float))


def _rdp(pts, eps):
    """Ramer–Douglas–Peucker: simplifica una polilínea a segmentos de recta."""
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    line = end - start
    L = np.hypot(line[0], line[1])
    if L == 0:
        d = np.hypot(*(pts - start).T)
    else:
        d = np.abs(np.cross(np.tile(line, (len(pts), 1)),
                            pts - start)) / L
    idx = int(np.argmax(d))
    if d[idx] > eps:
        left = _rdp(pts[:idx + 1], eps)
        right = _rdp(pts[idx:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def piecewise_linear_and_knee(dist, y, eps_frac=0.035):
    """
    Aproxima (dist, y) con segmentos de recta (RDP) y localiza el vértice
    donde el cambio de pendiente entre rectas consecutivas es mayor.

    Returns
    -------
    verts : (M,2) vértices de la aproximación lineal
    knee  : dict con idx, dist, intensity, dslope, slopes  (o None)
    """
    P = np.column_stack([dist.astype(float), y.astype(float)])
    xr = max(dist.max() - dist.min(), 1e-9)
    yr = max(y.max() - y.min(), 1e-9)
    Pn = np.column_stack([(P[:, 0] - dist.min()) / xr,
                          (P[:, 1] - y.min()) / yr])

    # epsilon en coordenadas normalizadas; se relaja si quedan pocos tramos
    eps = eps_frac
    for _ in range(6):
        Vn = _rdp(Pn, eps)
        if len(Vn) >= 4 or eps <= 0.005:
            break
        eps /= 1.6

    verts = np.column_stack([Vn[:, 0] * xr + dist.min(),
                             Vn[:, 1] * yr + y.min()])

    if len(verts) < 3:
        return verts, None

    M = len(verts)
    slopes = np.diff(verts[:, 1]) / np.diff(verts[:, 0])

    # Significancia del quiebre = ÁNGULO DE GIRO entre las dos rectas
    # consecutivas, medido en coordenadas NORMALIZADAS (así coincide con
    # el codo que se ve en la gráfica). Es más robusto que |Δpendiente|,
    # que sobrevalora los quiebres donde las pendientes ya son grandes.
    angle = np.zeros(M)  # ángulo en cada vértice (0 en los extremos)
    for k in range(1, M - 1):
        v1 = Vn[k] - Vn[k - 1]
        v2 = Vn[k + 1] - Vn[k]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        angle[k] = abs(np.arctan2(cross, dot))

    # Vértices candidatos = interiores (1..M-2). Se ignoran los EXTREMOS:
    # el primer interior (borde del piso foveal) y el último (borde de la
    # meseta), quedándonos con el giro intermedio más pronunciado.
    cand = list(range(1, M - 1))
    if len(cand) > 2:
        cand = cand[1:-1]

    k = max(cand, key=lambda j: angle[j])
    knee = {
        "idx": k,
        "dist": float(verts[k, 0]),
        "intensity": float(verts[k, 1]),
        "angle_deg": float(np.degrees(angle[k])),
        "dslope": float(abs(slopes[k] - slopes[k - 1])),
        "slopes": slopes,
    }
    return verts, knee


def visualize_radial_function(dist, y_mean, y_raw, lvl, verts, knee, out_path):
    """
    Gráfica tipo función con la aproximación por rectas y el codo marcado:
      X = puntos a lo largo de la recta horizontal (distancia al centro, px)
      Y = nivel de intensidad promedio del contorno que corta cada punto
    """
    norm = Normalize(vmin=lvl.min(), vmax=lvl.max())
    cmap = cm.turbo

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Datos: puntos coloreados por nivel + referencia cruda
    ax.plot(dist, y_raw, "--", color="#cbd5e1", linewidth=1.0, zorder=1,
            label="Intensidad del propio punto (referencia)")
    ax.scatter(dist, y_mean, c=lvl, cmap=cmap, norm=norm, s=32,
               edgecolor="none", zorder=2, alpha=0.75,
               label="Intensidad media del contorno")

    # Aproximación por rectas (RDP)
    ax.plot(verts[:, 0], verts[:, 1], "-", color="#111827", linewidth=2.2,
            marker="s", markersize=6, markerfacecolor="white",
            markeredgecolor="#111827", zorder=4,
            label="Aproximación por rectas")

    # Codo: mayor cambio de pendiente entre rectas
    if knee is not None:
        ax.axvline(knee["dist"], color="#dc2626", linestyle=":", linewidth=1.3,
                   zorder=3)
        ax.scatter([knee["dist"]], [knee["intensity"]], s=280, marker="*",
                   color="#dc2626", edgecolor="white", linewidths=1.0,
                   zorder=5,
                   label="Codo (mayor cambio de pendiente)")
        ax.annotate(
            f"  codo\n  d={knee['dist']:.0f} px\n  I={knee['intensity']:.1f}",
            xy=(knee["dist"], knee["intensity"]),
            xytext=(12, -38), textcoords="offset points",
            color="#dc2626", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Punto a lo largo de la recta horizontal — "
                  "distancia al centro de mácula (px)")
    ax.set_ylabel("Nivel de intensidad promedio del contorno")
    ax.set_title("Función radial aproximada por rectas — codo de cambio "
                 "de pendiente", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Nivel del contorno (intensidad umbral)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def select_intersection_contours(verts, dist, lvl, curves):
    """
    Devuelve solo los contornos que pasan por las INTERSECCIONES entre
    rectas consecutivas (los vértices interiores de la aproximación).

    Returns list de (nivel, contorno, (dist_vertice, intensidad_vertice)).
    """
    level_to_contour = {int(lv): c for lv, c in curves}
    avail = np.array(sorted(level_to_contour.keys()))
    n = len(lvl)

    selected, seen = [], set()
    for vx, vy in verts[1:-1]:                 # vértices interiores = intersecciones
        idx = int(np.clip(round(vx), 0, n - 1))
        level = int(lvl[idx])
        if level not in level_to_contour:      # nivel sin curva → el más cercano
            level = int(avail[np.argmin(np.abs(avail - level))])
        if level in seen:
            continue
        seen.add(level)
        selected.append((level, level_to_contour[level], (float(vx), float(vy))))
    return selected


def visualize_intersection_contours(img, center, selected, knee_level,
                                    out_path):
    """
    Dibuja SOLO los contornos de las intersecciones entre rectas sobre la
    imagen original.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cyc, cxc = center

    if not selected:
        return
    levels = np.array([lv for lv, _, _ in selected], dtype=float)
    norm = Normalize(vmin=levels.min(), vmax=levels.max())
    cmap = cm.turbo

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(rgb)

    for lv, c, (vx, vy) in selected:
        col = cmap(norm(lv))
        cc = np.vstack([c, c[0]])                          # cerrar el lazo
        is_knee = (knee_level is not None and lv == knee_level)
        ax.plot(cc[:, 0], cc[:, 1], color=col,
                linewidth=3.0 if is_knee else 1.6,
                zorder=4 if is_knee else 3)

    # Centro de mácula
    ax.scatter(cxc, cyc, s=150, marker="*", color="red",
               edgecolor="white", linewidths=0.9, zorder=5,
               label="Centro de mácula")
    if knee_level is not None:
        ax.plot([], [], color="black", linewidth=3.0,
                label=f"Contorno del codo (nivel {knee_level})")

    ax.set_title("Contornos de las intersecciones entre rectas "
                 "sobre la imagen original", fontsize=13, fontweight="bold")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(loc="upper right", fontsize=9)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Nivel del contorno (intensidad umbral)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def visualize(img, center, curves, xs, cy, original_center, roi_radius,
              knee_radius, out_path):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cyc, cxc = center

    levels = np.array([lv for lv, _ in curves], dtype=float)
    norm = Normalize(vmin=levels.min(), vmax=levels.max())
    cmap = cm.turbo

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, base, cmap_base, title in (
        (axes[0], rgb, None, "(A) Curvas de nivel sobre imagen original"),
        (axes[1], gray, "gray", "(B) Curvas de nivel sobre escala de grises"),
    ):
        ax.imshow(base, cmap=cmap_base)

        for lv, c in curves:
            col = cmap(norm(lv))
            cc = np.vstack([c, c[0]])                # cerrar el lazo
            ax.plot(cc[:, 0], cc[:, 1], color=col, linewidth=0.8)

        # ROI macular (30% del radio ocular)
        roi_circle = plt.Circle((cxc, cyc), roi_radius, color="magenta",
                                 fill=False, linewidth=1.2, linestyle="-",
                                 label="ROI macular (30% radio ocular)")
        ax.add_patch(roi_circle)
        # Radio del codo (cambio de pendiente)
        if knee_radius is not None:
            knee_circle = plt.Circle((cxc, cyc), knee_radius, color="red",
                                     fill=False, linewidth=1.4, linestyle="--",
                                     label="Radio del codo (cambio de pendiente)")
            ax.add_patch(knee_circle)
        # Recta radial horizontal
        ax.plot([cxc, xs[-1]], [cyc, cyc], color="white",
                linewidth=1.5, linestyle="--", label="Recta radial horizontal")
        # Centro de mácula
        ax.scatter(cxc, cyc, s=140, marker="*", color="red",
                   edgecolor="white", linewidths=0.8, label="Centro de mácula",
                   zorder=5)
        # Centro anterior (paso 8)
        oy, ox = original_center
        ax.scatter(ox, oy, s=60, marker="o", facecolor="none",
                   edgecolor="cyan", linewidths=1.2, label="Centro paso 8",
                   zorder=5)

        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.legend(loc="upper right", fontsize=8)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Intensidad (nivel de la curva)")

    fig.suptitle(
        "Curvas de nivel radiales desde el centro de mácula "
        "(cierre tipo Canny)",
        fontsize=15, fontweight="bold",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def overlay_cv2(img, curves, center, xs, cy, roi_radius, knee_radius, out_path):
    """Superposición directa de contornos sobre una imagen de salida."""
    out = img.copy()
    levels = np.array([lv for lv, _ in curves], dtype=float)
    norm = Normalize(vmin=levels.min(), vmax=levels.max())
    cmap = cm.turbo
    for lv, c in curves:
        r, g, b, _ = cmap(norm(lv))
        color = (int(b * 255), int(g * 255), int(r * 255))  # BGR
        cv2.polylines(out, [c.reshape(-1, 1, 2).astype(np.int32)],
                      True, color, 1, cv2.LINE_AA)
    cyc, cxc = center
    cv2.circle(out, (cxc, cyc), roi_radius, (255, 0, 255), 1, cv2.LINE_AA)
    if knee_radius is not None:
        cv2.circle(out, (cxc, cyc), knee_radius, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.line(out, (cxc, cyc), (int(xs[-1]), cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.drawMarker(out, (cxc, cyc), (0, 0, 255), cv2.MARKER_STAR, 16, 2)
    cv2.imwrite(out_path, out)


def analyze(img, channel="luminance"):
    """
    Ejecuta todo el análisis y devuelve los artefactos intermedios
    (centro, gris del análisis, curvas de nivel, función radial y codo)
    para reutilizarlos en visualizaciones o experimentos.

    channel : "luminance" (BGR2GRAY) o "green" (canal verde).
    """
    data = compute_macula_center(img, channel=channel)
    center = data["new_center"]                 # (y, x)
    fov = data["fov"]
    original_center = data["original_center"]

    curves, xs, cy, line_levels, gray, roi_radius, ocular_radius = (
        generate_level_curves(img, center, fov, roi_frac=0.30,
                              smooth_sigma=2.0, close_k=5, channel=channel)
    )
    dist, y_mean, y_raw, lvl = build_radial_function(
        line_levels, center, curves, gray
    )
    verts, knee = piecewise_linear_and_knee(dist, y_mean)

    return {
        "center": center, "fov": fov, "original_center": original_center,
        "curves": curves, "xs": xs, "cy": cy, "line_levels": line_levels,
        "gray": gray, "roi_radius": roi_radius, "ocular_radius": ocular_radius,
        "dist": dist, "y_mean": y_mean, "y_raw": y_raw, "lvl": lvl,
        "verts": verts, "knee": knee,
    }


def knee_contour_level(knee, lvl):
    """Nivel del contorno que pasa por el codo (o None)."""
    if knee is None:
        return None
    idx = int(np.clip(round(knee["dist"]), 0, len(lvl) - 1))
    return int(lvl[idx])


def run(img, out="results", channel="luminance"):
    os.makedirs(out, exist_ok=True)

    print(f"[10] Curvas de nivel radiales (ROI macular, canal={channel})...")
    a = analyze(img, channel=channel)
    center = a["center"]
    fov = a["fov"]
    original_center = a["original_center"]
    curves, xs, cy = a["curves"], a["xs"], a["cy"]
    gray, roi_radius = a["gray"], a["roi_radius"]
    dist, y_mean, y_raw, lvl = a["dist"], a["y_mean"], a["y_raw"], a["lvl"]
    verts, knee = a["verts"], a["knee"]

    print(f"     Recta radial: {len(xs)} px "
          f"(x {xs[0]}→{xs[-1]}, y={cy})")
    print(f"     Curvas de nivel generadas: {len(curves)}")
    if curves:
        _lv = [c[0] for c in curves]
        print(f"     Niveles: min={min(_lv)}, max={max(_lv)}")

    knee_radius = int(round(knee["dist"])) if knee is not None else None
    if knee is not None:
        print(f"     Codo: d={knee['dist']:.0f}px, "
              f"I={knee['intensity']:.1f}, giro={knee['angle_deg']:.1f}°")

    png = os.path.join(out, "macula_level_curves.png")
    visualize(img, center, curves, xs, cy, original_center, roi_radius,
              knee_radius, png)
    print(f"[✓] Curvas de nivel: {png}")

    overlay = os.path.join(out, "macula_level_curves_overlay.png")
    overlay_cv2(img, curves, center, xs, cy, roi_radius, knee_radius, overlay)
    print(f"[✓] Overlay:         {overlay}")

    # Gráfica 2D: contorno (X) vs intensidad promedio del contorno (Y)
    rows = compute_contour_intensities(gray, curves)
    profile = os.path.join(out, "macula_contour_intensity_profile.png")
    visualize_intensity_profile(rows, profile)
    print(f"[✓] Perfil 2D:       {profile}")

    csv_path = os.path.join(out, "macula_contour_intensity_profile.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("contorno,nivel,intensidad_media,intensidad_std,n_pixeles\n")
        for r in rows:
            f.write(f"C{r['idx']},{r['level']},{r['mean']:.3f},"
                    f"{r['std']:.3f},{r['n']}\n")
    print(f"[✓] CSV:             {csv_path}")

    # Gráfica tipo función aproximada por rectas + codo
    func = os.path.join(out, "macula_radial_intensity_function.png")
    visualize_radial_function(dist, y_mean, y_raw, lvl, verts, knee, func)
    print(f"[✓] Función radial:  {func}")

    func_csv = os.path.join(out, "macula_radial_intensity_function.csv")
    with open(func_csv, "w", encoding="utf-8") as f:
        f.write("distancia_px,nivel_contorno,intensidad_media_contorno,"
                "intensidad_punto\n")
        for d, lv, ym, yr in zip(dist, lvl, y_mean, y_raw):
            f.write(f"{int(d)},{int(lv)},{ym:.3f},{yr:.3f}\n")
    print(f"[✓] CSV función:     {func_csv}")

    # Solo los contornos de las intersecciones entre rectas, sobre el original
    selected = select_intersection_contours(verts, dist, lvl, curves)
    knee_level = None
    if knee is not None:
        knee_level = int(lvl[int(np.clip(round(knee["dist"]), 0, len(lvl) - 1))])
    inter = os.path.join(out, "macula_intersection_contours.png")
    visualize_intersection_contours(img, center, selected, knee_level, inter)
    print(f"[✓] Contornos intersección ({len(selected)}): {inter}")


SUPPORTED_EXT = {".ppm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def discover_images(root):
    return sorted(
        p for p in os.listdir(root)
        if os.path.splitext(p)[1].lower() in SUPPORTED_EXT
        and os.path.isfile(os.path.join(root, p))
    )


if __name__ == "__main__":
    pa = argparse.ArgumentParser(
        description="Curvas de nivel radiales desde el centro de mácula."
    )
    pa.add_argument(
        "--input",
        help="Imagen única. Si se omite, procesa TODAS las de --input_dir.",
    )
    pa.add_argument(
        "--input_dir",
        default=".",
        help="Carpeta con las imágenes (por defecto: la raíz actual).",
    )
    pa.add_argument("--output_dir", default="results")
    pa.add_argument("--channel", default="luminance",
                    choices=["luminance", "green"],
                    help="Gris del análisis: luminancia o canal verde.")
    a = pa.parse_args()

    if a.input:
        images = [a.input]
    else:
        images = [
            os.path.join(a.input_dir, f)
            for f in discover_images(a.input_dir)
        ]
        if not images:
            print(f"[ERROR] No se encontraron imágenes en {a.input_dir}")
            raise SystemExit(1)
        print(f"[*] {len(images)} imágenes: "
              f"{', '.join(os.path.basename(i) for i in images)}")

    ok, fail = 0, 0
    for path in images:
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'='*60}\n[IMG] {path}\n{'='*60}")
        im = cv2.imread(path)
        if im is None:
            print(f"[ERROR] No se pudo leer {path}")
            fail += 1
            continue
        try:
            # Una imagen -> salida directa; varias -> subcarpeta por imagen
            out = a.output_dir if a.input else os.path.join(a.output_dir, stem)
            run(im, out, channel=a.channel)
            ok += 1
        except Exception as e:
            print(f"[ERROR] Falló {path}: {e}")
            fail += 1

    print(f"\n[✓] Completado. OK={ok}, fallidas={fail}")
