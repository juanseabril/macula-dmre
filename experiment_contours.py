"""
Experimento comparativo entre imágenes  (canal VERDE)
=====================================================
Para cada imagen de la raíz, con el análisis hecho sobre el CANAL VERDE:

  1. Compara DOS contornos:
       - C2  : el SEGUNDO contorno detectado cerca de la fóvea.
       - Codo: el contorno que pasa por el codo de la función radial.
     Los superpone sobre la imagen original y el canal verde.

  2. Determina el MEJOR contorno macular: entre los contornos a la escala
     del codo, el más circular (compacto) — el que mejor se adapta a la
     mácula (región aproximadamente redonda).

Salidas (results_experiment/):
  - experiment_contours_green.png   (C2 vs codo)
  - best_macular_contour.png        (mejor contorno macular resaltado)
  - experiment_contours_green.csv

Uso:  python3 experiment_contours.py            (verde por defecto)
      python3 experiment_contours.py --channel luminance
"""

import argparse
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from macula_level_curves import analyze, discover_images, knee_contour_level

C2_COLOR = "#22d3ee"       # cian
KNEE_COLOR = "#dc2626"     # rojo
BEST_COLOR = "#16a34a"     # verde
WINDOW = 130               # recorte alrededor del centro de mácula (px)


def contour_shape(c):
    """Área, perímetro, radio equivalente y circularidad (1 = círculo)."""
    cc = c.reshape(-1, 1, 2).astype(np.int32)
    area = float(cv2.contourArea(cc))
    per = float(cv2.arcLength(cc, True))
    r_eq = float(np.sqrt(area / np.pi)) if area > 0 else 0.0
    circ = float(4 * np.pi * area / (per * per)) if per > 0 else 0.0
    return area, per, r_eq, min(circ, 1.0)


def pick_c2_and_knee(a):
    """(nivel, contorno) de C2 y del codo."""
    curves = a["curves"]
    c2 = curves[1] if len(curves) >= 2 else (None, None)
    klvl = knee_contour_level(a["knee"], a["lvl"])
    lut = {int(lv): c for lv, c in curves}
    if klvl is not None and klvl not in lut:
        av = np.array(sorted(lut))
        klvl = int(av[np.argmin(np.abs(av - klvl))])
    kc = (klvl, lut[klvl]) if klvl is not None else (None, None)
    return c2, kc


def best_macular_contour(a):
    """
    Mejor contorno macular = el más circular entre los que están a la
    escala del codo (r_eq en [0.6, 1.3]·radio_codo). Si no hay, el más
    circular de todos.
    """
    curves = a["curves"]
    knee = a["knee"]
    kr = knee["dist"] if knee is not None else None

    cand = []
    for lv, c in curves:
        area, per, r_eq, circ = contour_shape(c)
        if area <= 0:
            continue
        cand.append({"level": int(lv), "c": c, "r_eq": r_eq, "circ": circ})
    if not cand:
        return None

    band = cand
    if kr is not None:
        band = [x for x in cand if 0.6 * kr <= x["r_eq"] <= 1.3 * kr] or cand
    return max(band, key=lambda x: x["circ"])


def draw(ax, base, cmap, center, contours, title):
    """contours = lista de (contorno, color, grosor)."""
    ax.imshow(base, cmap=cmap)
    cyc, cxc = center
    for c, col, lw in contours:
        if c is None:
            continue
        cc = np.vstack([c, c[0]])
        ax.plot(cc[:, 0], cc[:, 1], color=col, linewidth=lw)
    ax.scatter(cxc, cyc, s=90, marker="*", color="yellow",
               edgecolor="black", linewidths=0.6, zorder=5)
    ax.set_xlim(cxc - WINDOW, cxc + WINDOW)
    ax.set_ylim(cyc + WINDOW, cyc - WINDOW)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def run(input_dir=".", out_dir="results_experiment", channel="green"):
    os.makedirs(out_dir, exist_ok=True)
    files = discover_images(input_dir)
    if not files:
        print(f"[ERROR] No hay imágenes en {input_dir}")
        raise SystemExit(1)
    print(f"[*] Experimento ({channel}) sobre {len(files)} imágenes")

    n = len(files)
    fig1, ax1 = plt.subplots(n, 2, figsize=(10, 5 * n))   # C2 vs codo
    fig2, ax2 = plt.subplots(n, 2, figsize=(10, 5 * n))   # mejor contorno
    if n == 1:
        ax1, ax2 = ax1[np.newaxis, :], ax2[np.newaxis, :]

    rows = []
    for i, fname in enumerate(files):
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] No se pudo leer {path}")
            continue
        stem = os.path.splitext(fname)[0]
        print(f"\n[IMG] {fname}")
        a = analyze(img, channel=channel)
        center = a["center"]
        green = img[:, :, 1]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        (c2_lv, c2), (kc_lv, kc) = pick_c2_and_knee(a)
        best = best_macular_contour(a)

        _, _, c2_r, c2_circ = contour_shape(c2) if c2 is not None else (0, 0, 0, 0)
        _, _, kc_r, kc_circ = contour_shape(kc) if kc is not None else (0, 0, 0, 0)

        # Fig 1 — C2 (cian) vs codo (rojo)
        cs = [(c2, C2_COLOR, 1.8), (kc, KNEE_COLOR, 2.4)]
        draw(ax1[i, 0], rgb, None, center, cs, f"{stem} — Original")
        draw(ax1[i, 1], green, "gray", center, cs, f"{stem} — Canal verde")

        # Fig 2 — mejor contorno macular (verde grueso) + codo (rojo fino)
        cs2 = [(kc, KNEE_COLOR, 1.2)]
        if best is not None:
            cs2.append((best["c"], BEST_COLOR, 2.8))
        lab = (f"{stem} — mejor: nivel {best['level']}, "
               f"r={best['r_eq']:.0f}px, circ={best['circ']:.2f}"
               if best else stem)
        draw(ax2[i, 0], rgb, None, center, cs2, lab)
        draw(ax2[i, 1], green, "gray", center, cs2, f"{stem} — Canal verde")

        print(f"     C2   : nivel={c2_lv}, r_eq={c2_r:.1f}px, circ={c2_circ:.2f}")
        print(f"     codo : nivel={kc_lv}, r_eq={kc_r:.1f}px, circ={kc_circ:.2f}")
        if best:
            print(f"     MEJOR: nivel={best['level']}, r_eq={best['r_eq']:.1f}px, "
                  f"circ={best['circ']:.2f}")
        rows.append({
            "imagen": stem,
            "c2_lv": c2_lv, "c2_r": c2_r, "c2_circ": c2_circ,
            "kc_lv": kc_lv, "kc_r": kc_r, "kc_circ": kc_circ,
            "best_lv": best["level"] if best else None,
            "best_r": best["r_eq"] if best else None,
            "best_circ": best["circ"] if best else None,
        })

    # Leyendas / títulos
    h1 = [plt.Line2D([], [], color=C2_COLOR, lw=2, label="C2 (2º cerca de fóvea)"),
          plt.Line2D([], [], color=KNEE_COLOR, lw=2, label="Contorno del codo"),
          plt.Line2D([], [], color="yellow", marker="*", lw=0,
                     markeredgecolor="black", label="Centro de mácula")]
    fig1.legend(handles=h1, loc="upper center", ncol=3, fontsize=11,
                bbox_to_anchor=(0.5, 1.004))
    fig1.suptitle(f"C2 vs codo — canal {channel}", fontsize=15,
                  fontweight="bold", y=1.015)
    fig1.tight_layout(rect=[0, 0, 1, 0.985])
    p1 = os.path.join(out_dir, "experiment_contours_green.png")
    fig1.savefig(p1, dpi=220, bbox_inches="tight")
    fig1.savefig(p1.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig1)

    h2 = [plt.Line2D([], [], color=BEST_COLOR, lw=3, label="Mejor contorno macular"),
          plt.Line2D([], [], color=KNEE_COLOR, lw=1.2, label="Contorno del codo (ref.)"),
          plt.Line2D([], [], color="yellow", marker="*", lw=0,
                     markeredgecolor="black", label="Centro de mácula")]
    fig2.legend(handles=h2, loc="upper center", ncol=3, fontsize=11,
                bbox_to_anchor=(0.5, 1.004))
    fig2.suptitle(f"Mejor contorno que se adapta a la mácula — canal {channel}",
                  fontsize=15, fontweight="bold", y=1.015)
    fig2.tight_layout(rect=[0, 0, 1, 0.985])
    p2 = os.path.join(out_dir, "best_macular_contour.png")
    fig2.savefig(p2, dpi=220, bbox_inches="tight")
    fig2.savefig(p2.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig2)

    csv_path = os.path.join(out_dir, "experiment_contours_green.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("imagen,c2_nivel,c2_r_eq,c2_circ,codo_nivel,codo_r_eq,"
                "codo_circ,mejor_nivel,mejor_r_eq,mejor_circ\n")
        for r in rows:
            f.write(f"{r['imagen']},{r['c2_lv']},{r['c2_r']:.2f},"
                    f"{r['c2_circ']:.3f},{r['kc_lv']},{r['kc_r']:.2f},"
                    f"{r['kc_circ']:.3f},{r['best_lv']},{r['best_r']:.2f},"
                    f"{r['best_circ']:.3f}\n")

    # Ranking por circularidad del mejor contorno
    ranked = sorted([r for r in rows if r["best_circ"] is not None],
                    key=lambda r: r["best_circ"], reverse=True)
    print("\n=== Ranking por circularidad del mejor contorno macular ===")
    for r in ranked:
        flag = "OK " if r["best_circ"] >= 0.55 else "revisar"
        print(f"  {r['imagen']}: circ={r['best_circ']:.2f} "
              f"r={r['best_r']:.0f}px  [{flag}]")

    print(f"\n[✓] Figura C2 vs codo:   {p1}")
    print(f"[✓] Figura mejor contorno: {p2}")
    print(f"[✓] CSV:                 {csv_path}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--input_dir", default=".")
    pa.add_argument("--output_dir", default="results_experiment")
    pa.add_argument("--channel", default="green",
                    choices=["luminance", "green"])
    args = pa.parse_args()
    run(args.input_dir, args.output_dir, channel=args.channel)
    print("[✓] Completado.")
