"""
Paso 9 — Centro macular por mínima intensidad
=============================================
A partir del ÁREA macular generada en el paso 8 (contorno suavizado
del pipeline de dmre.py), se construye la máscara del área, se busca
el píxel de menor intensidad dentro de ella y se define como nuevo
centro de la mácula.

Uso:  python macula_min_intensity.py --input im0299.ppm
"""

import argparse
import os

import cv2
import numpy as np

import dmre


def build_macula_area_mask(contour, shape):
    """Rasteriza el contorno del paso 8 a una máscara binaria (área macular)."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if contour is None or len(contour) < 3:
        raise ValueError(
            "El paso 8 no produjo un contorno válido; "
            "no se puede construir el área macular."
        )
    pts = contour.astype(np.int32).reshape(-1, 1, 2)  # (x, y)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def compute_macula_center(img, channel="luminance"):
    """
    Reproduce el pipeline de dmre.py hasta el paso 9 y devuelve
    el centro de mácula (píxel de mínima intensidad dentro del
    área del paso 8).

    channel : "luminance" (BGR2GRAY) o "green" (canal verde). Define
              sobre qué imagen se busca el píxel de mínima intensidad.

    Returns
    -------
    dict con:
        new_center       : (y, x) centro de mácula (mínima intensidad)
        min_intensity    : float
        original_center  : (y, x) centroide del contorno (paso 8)
        macula_area_mask : máscara binaria del área macular
        fov              : máscara del campo de visión
        img_norm         : imagen normalizada
    """
    # ── Reproducir el pipeline hasta el paso 8 (contorno) ──
    img_norm = dmre.normalize_fundus(img)
    fov = dmre.create_fov_mask(img)

    disc_info = dmre.detect_optic_disc(img_norm[:, :, 1], fov)
    mac_cy, mac_cx, mac_r = dmre.estimate_macula_center(img_norm, fov, disc_info)

    v_raw, diff_img, faz_r = dmre.segment_vessels(
        img_norm, fov, disc_info, (mac_cy, mac_cx, mac_r)
    )
    v_clean = dmre.postprocess_vessels(v_raw, fov)

    skel = dmre.build_skeleton(v_clean)
    ep, bf = dmre.detect_keypoints(skel)

    selected, rejected = dmre.find_macular_boundary(
        v_clean, skel, ep, mac_cy, mac_cx, mac_r
    )
    if len(selected) < 8 and len(bf) > 0:
        all_pts = np.vstack([ep, bf]) if len(ep) > 0 else bf
        selected, rejected = dmre.find_macular_boundary(
            v_clean, skel, all_pts, mac_cy, mac_cx, mac_r
        )

    # Centro macular anterior (centroide de los puntos seleccionados)
    cent_r, cent_c = mac_cy, mac_cx
    if len(selected) >= 3:
        cent_r = int(selected[:, 0].mean())
        cent_c = int(selected[:, 1].mean())
    original_center = (cent_r, cent_c)  # (y, x)

    # Paso 8 — contorno suavizado = ÁREA macular
    contour = dmre.build_smooth_contour(selected, cent_r, cent_c)
    print("[8] Área macular (contorno del paso 8)...")
    macula_area_mask = build_macula_area_mask(contour, img.shape)
    print(f"    Área: {int(np.count_nonzero(macula_area_mask)):,} px")

    # Paso 9 — píxel de mínima intensidad = centro de mácula
    # (sobre luminancia o sobre el canal verde según `channel`)
    print(f"[9] Buscando píxel de mínima intensidad ({channel})...")
    src = img if channel == "luminance" else img[:, :, 1]
    new_center, min_intensity, _ = dmre.find_min_intensity_macula_center(
        src, macula_area_mask, original_center=original_center
    )
    print(f"    Centro anterior (y,x): {original_center}")
    print(f"    Centro de mácula (y,x): {new_center}")
    print(f"    Intensidad mínima:      {min_intensity:.2f}")

    return {
        "new_center": new_center,
        "min_intensity": min_intensity,
        "original_center": original_center,
        "macula_area_mask": macula_area_mask,
        "fov": fov,
        "img_norm": img_norm,
    }


def run(img, out="results"):
    os.makedirs(out, exist_ok=True)

    data = compute_macula_center(img)
    new_center = data["new_center"]
    min_intensity = data["min_intensity"]
    original_center = data["original_center"]
    macula_area_mask = data["macula_area_mask"]

    # ── Nueva gráfica ──
    out_path = os.path.join(out, "macula_min_intensity_center.png")
    dmre.visualize_min_intensity_macula_center(
        img,
        macula_area_mask,
        new_center,
        min_intensity,
        out_path,
        original_center=original_center,
    )
    print(f"[✓] Nueva gráfica: {out_path}")
    print(f"[✓] PDF:           {out_path.replace('.png', '.pdf')}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--input", default="im0299.ppm")
    pa.add_argument("--output_dir", default="results")
    a = pa.parse_args()
    im = cv2.imread(a.input)
    if im is None:
        print(f"[ERROR] No se pudo leer {a.input}")
        raise SystemExit(1)
    run(im, a.output_dir)
    print("[✓] Completado.")
