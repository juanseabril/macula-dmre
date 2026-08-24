"""
Macular Contour Estimation via Vascular Network Analysis — v3
=============================================================
Cambios clave:
  - Zona macular con umbral relajado para capturar vasos finos cercanos
  - Contorno basado en ENDPOINTS REALES del skeleton (consistente paso 4→5)
  - Umbral de componentes del skeleton reducido a 20% para mantener ramas
  - Sin elipse forzada — contorno libre que sigue la forma real

Uso:  python DMRE_v3.py --input im0299.ppm
"""

import argparse, os
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from scipy.ndimage import label as ndlabel
from skimage import measure
from skimage.morphology import skeletonize, remove_small_objects
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════
def create_fov_mask(img, thr=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    k = max(25, min(img.shape[:2])//40)
    if k%2==0: k+=1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern), cv2.MORPH_OPEN, kern)

def normalize_fundus(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def detect_optic_disc(green, fov):
    h, w = green.shape
    blur_k = max(51, min(h,w)//6)
    if blur_k%2==0: blur_k+=1
    blurred = cv2.GaussianBlur(green, (blur_k, blur_k), 0)
    blurred = cv2.bitwise_and(blurred, fov)
    _, _, _, max_loc = cv2.minMaxLoc(blurred, mask=fov)
    cx, cy = max_loc
    fov_coords = np.argwhere(fov > 0)
    fov_diam = max(fov_coords[:,0].max()-fov_coords[:,0].min(),
                   fov_coords[:,1].max()-fov_coords[:,1].min()) if len(fov_coords)>0 else min(h,w)
    return cy, cx, int(fov_diam * 0.08)

def estimate_macula_center(img_norm, fov, disc_info):
    h, w = img_norm.shape[:2]
    green = img_norm[:,:,1].copy()
    disc_cy, disc_cx, disc_r = disc_info
    M = cv2.moments(fov)
    fov_cx = int(M["m10"]/M["m00"]) if M["m00"]>0 else w//2
    fov_cy = int(M["m01"]/M["m00"]) if M["m00"]>0 else h//2
    fov_coords = np.argwhere(fov > 0)
    fov_radius = max(fov_coords[:,0].max()-fov_coords[:,0].min(),
                     fov_coords[:,1].max()-fov_coords[:,1].min())//2 if len(fov_coords)>0 else min(h,w)//2

    search = np.zeros((h,w), np.uint8)
    cv2.circle(search, (fov_cx,fov_cy), int(fov_radius*0.50), 255, -1)
    cv2.circle(search, (disc_cx,disc_cy), int(disc_r*2.0), 0, -1)
    search = cv2.bitwise_and(search, fov)
    if np.count_nonzero(search) < 100:
        search = np.zeros((h,w), np.uint8)
        cv2.circle(search, (fov_cx,fov_cy), int(fov_radius*0.70), 255, -1)
        cv2.circle(search, (disc_cx,disc_cy), int(disc_r*1.5), 0, -1)
        search = cv2.bitwise_and(search, fov)

    green_s = green.copy(); green_s[search==0] = 255
    blur_k = max(41, min(h,w)//8)
    if blur_k%2==0: blur_k+=1
    blurred = cv2.GaussianBlur(green_s, (blur_k,blur_k), 0)
    blurred[search==0] = 255
    _, _, min_loc, _ = cv2.minMaxLoc(blurred, mask=search)
    mac_cx, mac_cy = min_loc
    mac_r = int(disc_r * 1.8)
    print(f"      Disco: ({disc_cx},{disc_cy}) r={disc_r}")
    print(f"      Mácula: ({mac_cx},{mac_cy}) r={mac_r}")
    return mac_cy, mac_cx, mac_r


# ═══════════════════════════════════════════════
# 1. SEGMENTACIÓN (4 ZONAS)
#    Global + Disco + Anillo macular + FAZ excluida
# ═══════════════════════════════════════════════
def segment_vessels(img_norm, fov, disc_info, mac_info):
    """
    Pipeline mejorado basado en la literatura:
      1. Canal verde + inversión
      2. White Top-Hat (aísla vasos, suprime disco/exudados/fóvea)
      3. Bottom-Hat (realza bordes oscuros de vasos)
      4. Combinación: original + WTH - BTH
      5. CLAHE sobre resultado (ahora solo vasos prominentes)
      6. Gamma correction para realzar contraste fino
      7. Background subtraction + hysteresis por zonas
    """
    h, w = img_norm.shape[:2]
    scale = max(1, min(h,w)//512)

    green = img_norm[:,:,1].copy()
    mean_v = int(green[fov>0].mean()) if np.any(fov>0) else 128
    green[fov==0] = mean_v

    # ── 1. Invertir: vasos oscuros → brillantes ──
    green_inv = cv2.bitwise_not(green)
    green_inv[fov==0] = 0

    # ── 2. White Top-Hat: aísla estructuras brillantes (vasos) ──
    #    El kernel debe ser mayor que el vaso más grueso pero menor que el disco
    #    Típicamente: diámetro ≈ 1/30 del ancho de imagen
    th_k = max(15, min(h,w)//30)
    if th_k%2==0: th_k+=1
    se_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th_k, th_k))
    white_tophat = cv2.morphologyEx(green_inv, cv2.MORPH_TOPHAT, se_tophat)

    # ── 3. Bottom-Hat: realza bordes en zonas oscuras ──
    black_tophat = cv2.morphologyEx(green_inv, cv2.MORPH_BLACKHAT, se_tophat)

    # ── 4. Combinación: refuerza vasos, suprime fondo ──
    enhanced = cv2.add(green_inv, white_tophat)
    enhanced = cv2.subtract(enhanced, black_tophat)
    enhanced[fov==0] = 0

    # ── 5. CLAHE (ahora trabaja sobre vasos ya aislados) ──
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(enhanced)

    # ── 6. Gamma correction para realzar vasos finos ──
    gamma = 1.5
    table = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype(np.uint8)
    enhanced = cv2.LUT(enhanced, table)

    # ── 7. Background subtraction ──
    med_k = max(31, min(h,w)//8)
    if med_k%2==0: med_k+=1
    med_k = min(med_k, 255)
    background = cv2.medianBlur(enhanced, med_k)
    diff = cv2.subtract(enhanced, background)
    diff_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(diff)

    # Guardar el enhancement para visualización
    vessel_enhance = enhanced.copy()

    fov_vals = diff_enhanced[fov>0]
    otsu_val, _ = cv2.threshold(fov_vals, 0, 255, cv2.THRESH_OTSU)

    dk = max(3, scale)
    if dk%2==0: dk+=1
    kern_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk,dk))

    def hyst(img, th, tl, kern):
        s = np.uint8(img > th); wk = np.uint8(img > tl); r = s.copy()
        for _ in range(15):
            e = cv2.dilate(r, kern, iterations=1)
            n = cv2.bitwise_and(wk, e); c = cv2.bitwise_or(r, n)
            if np.array_equal(c, r): break
            r = c
        return r

    # A) GLOBAL (estricto)
    result_g = hyst(diff_enhanced, int(otsu_val*1.4), int(otsu_val*0.85), kern_d)

    # B) ANILLO DISCO (relajado — recuperar vasos del disco)
    disc_cy, disc_cx, disc_r = disc_info
    ring_outer = max(int(disc_r*2.5), disc_r+30)
    ring_inner = max(int(disc_r*0.4), 5)
    disc_ring = np.zeros((h,w), np.uint8)
    cv2.circle(disc_ring, (disc_cx,disc_cy), ring_outer, 255, -1)
    dc = np.zeros((h,w), np.uint8)
    cv2.circle(dc, (disc_cx,disc_cy), ring_inner, 255, -1)
    disc_ring = cv2.subtract(disc_ring, dc)
    result_d = hyst(diff_enhanced, int(otsu_val*0.55), int(otsu_val*0.25), kern_d)
    rdr = np.zeros_like(result_d)
    rdr[disc_ring > 0] = result_d[disc_ring > 0]

    # C) ANILLO MACULAR (relajado — capturar vasos finos cerca de la mácula)
    mac_cy, mac_cx, mac_r = mac_info
    faz_r = int(mac_r * 0.30)  # FAZ más pequeña
    mac_ring = np.zeros((h,w), np.uint8)
    cv2.circle(mac_ring, (mac_cx,mac_cy), int(mac_r*2.0), 255, -1)  # anillo exterior
    mac_core = np.zeros((h,w), np.uint8)
    cv2.circle(mac_core, (mac_cx,mac_cy), faz_r, 255, -1)  # FAZ interior
    mac_ring = cv2.subtract(mac_ring, mac_core)

    result_m = hyst(diff_enhanced, int(otsu_val*0.60), int(otsu_val*0.35), kern_d)
    rmr = np.zeros_like(result_m)
    rmr[mac_ring > 0] = result_m[mac_ring > 0]

    # D) FAZ exclusión
    faz_mask = np.ones((h,w), np.uint8) * 255
    cv2.circle(faz_mask, (mac_cx,mac_cy), faz_r, 0, -1)

    # COMBINAR: global + disco + macular
    result = cv2.bitwise_or(result_g, rdr)
    result = cv2.bitwise_or(result, rmr)
    result = result * 255
    result = cv2.bitwise_and(result, faz_mask)

    # Opening
    ko = max(3, scale)
    if ko%2==0: ko+=1
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko,ko)))

    # Erosión para adelgazar
    ke = max(1, scale - 1)
    if ke >= 2:
        result = cv2.erode(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke,ke)))

    print(f"      Otsu={otsu_val}, FAZ_r={faz_r}, TopHat_k={th_k}")
    print(f"      Disco anillo: [{ring_inner}-{ring_outer}], thr=[{int(otsu_val*0.55)},{int(otsu_val*0.25)}]")
    print(f"      Macula anillo: [{faz_r}-{int(mac_r*2.0)}], thr=[{int(otsu_val*0.60)},{int(otsu_val*0.35)}]")

    return cv2.bitwise_and(result, fov), vessel_enhance, faz_r


# ═══════════════════════════════════════════════
# 2. POSTPROCESAMIENTO
# ═══════════════════════════════════════════════
def postprocess_vessels(mask, fov):
    h, w = mask.shape[:2]
    scale = max(1, min(h,w)//512)
    bm = mask > 0
    if np.any(bm):
        bm = remove_small_objects(bm, min_size=100*(scale**2))
    cleaned = bm.astype(np.uint8) * 255

    labeled, n = ndlabel(cleaned > 0)
    if n > 0:
        sizes = [(i, np.sum(labeled==i)) for i in range(1, n+1)]
        sizes.sort(key=lambda x: x[1], reverse=True)
        mx = sizes[0][1]
        keep = np.zeros_like(cleaned)
        for cid, sz in sizes:
            if sz >= mx * 0.015: keep[labeled==cid] = 255
        cleaned = keep

    kc = max(3, scale)
    if kc%2==0: kc+=1
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc,kc)))
    border = max(10, scale*5)
    fov_e = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border*2+1,border*2+1)))
    return cv2.bitwise_and(cleaned, fov_e)


# ═══════════════════════════════════════════════
# 3. SKELETON (umbral reducido a 20%)
# ═══════════════════════════════════════════════
def _nc(sb, r, c):
    h,w=sb.shape; n=0
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0: continue
            nr,nc_=r+dr,c+dc
            if 0<=nr<h and 0<=nc_<w and sb[nr,nc_]: n+=1
    return n

def build_skeleton(mask):
    binary = mask > 0
    binary[0,:]=False; binary[-1,:]=False; binary[:,0]=False; binary[:,-1]=False
    skel = skeletonize(binary)
    labeled = measure.label(skel)
    regions = measure.regionprops(labeled)
    if not regions: return skel.astype(np.uint8)*255

    largest_idx = np.argmax([r.area for r in regions])
    largest_area = regions[largest_idx].area

    # 15% — mantener más ramas finas cerca de la mácula
    clean_skel = np.zeros_like(skel)
    clean_skel[labeled == (largest_idx+1)] = True
    for region in regions:
        if region.area >= largest_area * 0.15 and region.label != (largest_idx+1):
            clean_skel[labeled == region.label] = True

    # Poda de ramas muy cortas
    skel_out = clean_skel.astype(np.uint8) * 255
    scale = max(1, min(mask.shape)//512)
    min_branch = max(6, scale*3)

    changed = True
    while changed:
        changed = False; sb = skel_out > 0
        for r, c in np.argwhere(sb):
            if _nc(sb,r,c) != 1: continue
            branch = [(r,c)]; cr,cc = r,c; vis = {(r,c)}
            for _ in range(min_branch):
                f = False
                for dr in (-1,0,1):
                    for dc in (-1,0,1):
                        if dr==0 and dc==0: continue
                        nr,nc_ = cr+dr, cc+dc
                        if 0<=nr<sb.shape[0] and 0<=nc_<sb.shape[1] and sb[nr,nc_] and (nr,nc_) not in vis:
                            vis.add((nr,nc_))
                            if _nc(sb,nr,nc_)>=3: f=True; break
                            branch.append((nr,nc_)); cr,cc=nr,nc_; f=True; break
                    if f: break
                if not f: break
            if len(branch) < min_branch:
                for br,bc in branch: skel_out[br,bc] = 0
                changed = True
    return skel_out


# ═══════════════════════════════════════════════
# 4. KEYPOINTS
# ═══════════════════════════════════════════════
def detect_keypoints(skeleton):
    skel = skeleton > 0; h, w = skel.shape
    endpoints = []; intersection_mask = np.zeros((h,w), np.uint8)
    for r in range(1, h-1):
        for c in range(1, w-1):
            if not skel[r,c]: continue
            v = int(np.sum(skel[r-1:r+2, c-1:c+2])) - 1
            if v == 1: endpoints.append((r,c))
            elif v > 2: intersection_mask[r,c] = 1

    labeled_inter = measure.label(intersection_mask)
    bifurcations = []
    for region in measure.regionprops(labeled_inter):
        cy, cx = round(region.centroid[0]), round(region.centroid[1])
        if 0<=cy<h and 0<=cx<w and skel[cy,cx]:
            bifurcations.append((cy,cx))
        else:
            best = min(region.coords, key=lambda p:(p[0]-cy)**2+(p[1]-cx)**2)
            bifurcations.append((best[0], best[1]))
    return np.array(endpoints), np.array(bifurcations)


# ═══════════════════════════════════════════════
# 5. CONTORNO MACULAR HÍBRIDO
#    Ray-cast sobre máscara + snap a endpoints
# ═══════════════════════════════════════════════
def find_macular_boundary(vessel_mask, skeleton, endpoints,
                           center_r, center_c, mac_r, angle_step=10):
    """
    Para cada sector de 10 grados:
      1. Lanzar rayo desde centro → primer pixel de vaso (mas cercano posible)
      2. Buscar endpoint del skeleton cerca del hit (snap 15px)
      3. Si hay endpoint cercano → usarlo (mas preciso)
      4. Si no → usar el hit directo
      5. Filtrar con mediana estricta
    """
    h, w = vessel_mask.shape
    vb = vessel_mask > 0
    n_sectors = 360 // angle_step
    min_search = int(mac_r * 0.20)
    max_search = int(mac_r * 1.2)
    snap_radius = 15

    boundary = []
    for sector in range(n_sectors):
        angle_rad = np.radians(sector * angle_step)
        ray_dc = np.cos(angle_rad)
        ray_dr = np.sin(angle_rad)

        hit_point = None
        for dist in range(min_search, max_search):
            r = int(center_r + dist * ray_dr)
            c = int(center_c + dist * ray_dc)
            if r < 0 or r >= h or c < 0 or c >= w:
                break
            if vb[r, c]:
                cons = 0
                for d2 in range(5):
                    rr = int(center_r + (dist+d2) * ray_dr)
                    cc = int(center_c + (dist+d2) * ray_dc)
                    if 0 <= rr < h and 0 <= cc < w and vb[rr, cc]:
                        cons += 1
                    else:
                        break
                if cons >= 3:
                    hit_point = (r, c, dist)
                    break

        if hit_point is None:
            continue

        hr, hc, hdist = hit_point

        # Snap a endpoint cercano si existe
        best_ep = None
        best_d = snap_radius + 1
        if len(endpoints) > 0:
            for er, ec in endpoints:
                d = np.sqrt((er - hr)**2 + (ec - hc)**2)
                if d < best_d:
                    best_d = d
                    best_ep = (er, ec)

        if best_ep is not None and best_d <= snap_radius:
            boundary.append(best_ep)
        else:
            boundary.append((hr, hc))

    if len(boundary) == 0:
        return np.array([]), np.array([])

    boundary = np.array(boundary)

    # Filtro mediana estricto
    dists = np.sqrt((boundary[:,0]-center_r)**2 + (boundary[:,1]-center_c)**2)
    median_d = np.median(dists)
    cutoff = median_d * 1.4

    valid_mask = dists <= cutoff
    valid_pts = boundary[valid_mask]
    rejected_pts = boundary[~valid_mask]

    print(f"      Rayos con hit: {len(boundary)}/{n_sectors}")
    print(f"      Mediana: {median_d:.0f}, corte: {cutoff:.0f}")
    print(f"      Validos: {len(valid_pts)}, Descartados: {len(rejected_pts)}")
    if len(valid_pts) > 0:
        d = np.sqrt((valid_pts[:,0]-center_r)**2 + (valid_pts[:,1]-center_c)**2)
        print(f"      Dist final: media={d.mean():.0f}, min={d.min():.0f}, max={d.max():.0f}")

    return valid_pts, rejected_pts



# ═══════════════════════════════════════════════
# 6. VECTORES DE INTENSIDAD
# ═══════════════════════════════════════════════
def compute_intensity_vectors(gray, cr, cc, points):
    h, w = gray.shape; vectors = []
    for pt in points:
        pr, pc = int(pt[0]), int(pt[1])
        n = int(np.sqrt((pr-cr)**2 + (pc-cc)**2))
        if n < 2: vectors.append(np.array([])); continue
        rows = np.clip(np.linspace(cr, pr, n).astype(int), 0, h-1)
        cols = np.clip(np.linspace(cc, pc, n).astype(int), 0, w-1)
        vectors.append(gray[rows, cols])
    return vectors


# ═══════════════════════════════════════════════
# 7. CONTORNO SUAVIZADO (libre, sin forzar elipse)
# ═══════════════════════════════════════════════
def build_smooth_contour(points, center_r, center_c):
    if len(points) < 4: return None
    xy = points[:, ::-1].astype(np.float64)

    # Deduplicar puntos muy cercanos (< 3px)
    unique = [xy[0]]
    for p in xy[1:]:
        if all(np.sqrt((p[0]-u[0])**2 + (p[1]-u[1])**2) > 3 for u in unique):
            unique.append(p)
    xy = np.array(unique)
    if len(xy) < 4: return None

    # Ordenar por ángulo
    dr = xy[:,1] - center_r; dc = xy[:,0] - center_c
    order = np.argsort(np.arctan2(dr, dc))
    xy_sorted = xy[order]

    try:
        closed = np.vstack([xy_sorted, xy_sorted[0]])
        k_spline = min(3, len(xy_sorted) - 1)
        smoothing = max(len(closed) * 0.5, 5)
        tck, u = splprep([closed[:,0], closed[:,1]], s=smoothing, per=True, k=k_spline)
        u_new = np.linspace(0, 1, 200)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new])
    except Exception as e:
        print(f"      [spline fallback: {e}]")
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xy)
        return xy[hull.vertices]


# ═══════════════════════════════════════════════
# VISUALIZACIÓN
# ═══════════════════════════════════════════════
def visualize(img, v_clean, skel, ep, bf,
              mac_center, mac_r, selected, rejected, contour,
              diff_img, disc_info, faz_r, path):

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    pr = max(4, min(h,w)//180); lt = max(2, min(h,w)//300)
    mc_r, mc_c = mac_center
    dcy, dcx, dr = disc_info

    fig, axes = plt.subplots(2, 4, figsize=(32,17), facecolor="#0a0e17")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02, wspace=0.05, hspace=0.14)
    for ax in axes.flat:
        ax.set_facecolor("#0a0e17"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#1e293b")

    titles = ["1 · Original", "2 · Vessel Enhancement", "3 · Segmentación",
              "4 · Skeleton + Endpoints",
              "5 · Selección Radial (10°)", "6 · Endpoints Seleccionados",
              "7 · Contorno Macular", "8 · Resultado Final"]

    # 1
    axes[0,0].imshow(rgb)
    # 2
    axes[0,1].imshow(diff_img, cmap="gray")

    # 3 — Segmentación con zonas marcadas
    seg = cv2.cvtColor(v_clean, cv2.COLOR_GRAY2RGB)
    cv2.circle(seg, (dcx,dcy), dr, (0,140,255), 2)           # disco
    cv2.circle(seg, (mc_c,mc_r), faz_r, (255,200,0), 2)      # FAZ
    cv2.circle(seg, (mc_c,mc_r), int(mac_r*2.0), (255,200,0), 1) # anillo macular
    axes[0,2].imshow(seg)

    # 4 — Skeleton con TODOS los endpoints y bifurcaciones
    sk = np.zeros((h,w,3), np.uint8); sk[skel>0] = [200,200,200]
    if len(ep):
        for r,c in ep: cv2.circle(sk, (c,r), pr, (0,255,0), -1)      # verde
    if len(bf):
        for r,c in bf: cv2.circle(sk, (c,r), pr, (255,100,0), -1)    # naranja
    axes[0,3].imshow(sk)

    # 5 — Malla radial + endpoints seleccionados (CONSISTENTE con paso 4)
    mesh = rgb.copy()
    cv2.circle(mesh, (mc_c,mc_r), 6, (245,158,11), -1)
    for a in range(0, 360, 10):
        rad = np.radians(a)
        ex = int(mc_c + mac_r*2.0*np.cos(rad))
        ey = int(mc_r + mac_r*2.0*np.sin(rad))
        cv2.line(mesh, (mc_c,mc_r), (ex,ey), (50,60,70), 1)
    # Todos los endpoints en gris
    if len(ep):
        for r,c in ep:
            d = np.sqrt((r-mc_r)**2 + (c-mc_c)**2)
            if d < mac_r * 2.5:
                cv2.circle(mesh, (c,r), pr-1, (80,80,80), -1)
    # Seleccionados en verde (mismos puntos que paso 4)
    if len(selected):
        for r,c in selected:
            cv2.circle(mesh, (c,r), pr+3, (0,255,0), -1)
            cv2.line(mesh, (mc_c,mc_r), (c,r), (0,180,0), 1)
    # Outliers en rojo
    if len(rejected):
        for r,c in rejected:
            cv2.circle(mesh, (c,r), pr+2, (255,80,80), -1)
    axes[1,0].imshow(mesh)

    # 6 — Puntos seleccionados sobre original
    sel_vis = rgb.copy()
    cv2.circle(sel_vis, (mc_c,mc_r), mac_r, (34,211,238), 1)
    if len(selected):
        for r,c in selected:
            cv2.circle(sel_vis, (c,r), pr+3, (34,211,238), -1)
        cent_r = int(selected[:,0].mean())
        cent_c = int(selected[:,1].mean())
        cv2.circle(sel_vis, (cent_c,cent_r), 8, (245,158,11), -1)
        cv2.circle(sel_vis, (cent_c,cent_r), 14, (245,158,11), 2)
    axes[1,1].imshow(sel_vis)

    # 7 — Contorno
    ct_vis = rgb.copy()
    if contour is not None and len(contour) > 2:
        pts = contour.astype(np.int32).reshape(-1,1,2)
        cv2.polylines(ct_vis, [pts], True, (52,211,153), lt+1)
    if len(selected):
        for r,c in selected:
            cv2.circle(ct_vis, (c,r), pr, (34,211,238), -1)
        cent_r = int(selected[:,0].mean())
        cent_c = int(selected[:,1].mean())
        cv2.circle(ct_vis, (cent_c,cent_r), 8, (245,158,11), -1)
    axes[1,2].imshow(ct_vis)

    # 8 — Final
    fn = rgb.copy(); ov = fn.copy()
    if contour is not None and len(contour) > 2:
        pts = contour.astype(np.int32).reshape(-1,1,2)
        cv2.fillPoly(ov, [pts], (34,211,238))
        cv2.addWeighted(ov, 0.15, fn, 0.85, 0, fn)
        cv2.polylines(fn, [pts], True, (34,211,238), lt+1)
    sk_ov = fn.copy(); sk_ov[skel>0] = [52,211,153]
    cv2.addWeighted(sk_ov, 0.20, fn, 0.80, 0, fn)
    if len(selected):
        for r,c in selected:
            cv2.circle(fn, (c,r), pr, (255,255,255), -1)
        cent_r = int(selected[:,0].mean())
        cent_c = int(selected[:,1].mean())
        cv2.circle(fn, (cent_c,cent_r), 10, (245,158,11), -1)
    axes[1,3].imshow(fn)

    for ax,t in zip(axes.flat, titles):
        ax.set_title(t, color="#f1f5f9", fontsize=15, fontweight="600", pad=16)
    fig.suptitle("Pipeline: Estimación del Contorno Macular vía Análisis Vascular",
                 color="#22d3ee", fontsize=22, fontweight="bold", y=0.96)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0e17")
    plt.close()
    print(f"[✓] Visualización: {path}")

def find_min_intensity_macula_center(
    image,
    macula_area_mask,
    original_center=None
):
    """
    Encuentra el píxel con menor intensidad dentro
    del área macular generada en el paso 8.

    Parameters
    ----------
    image : np.ndarray
        Imagen BGR original o preprocesada.

    macula_area_mask : np.ndarray
        Máscara binaria del área macular generada
        en el paso 8.

    original_center : tuple, optional
        Centro macular anterior en formato (y, x).

    Returns
    -------
    new_center : tuple
        Nuevo centro definido por el píxel de
        menor intensidad (y, x).

    min_intensity : float
        Intensidad mínima encontrada.

    intensity_map : np.ndarray
        Imagen de intensidad utilizada.

    """

    if image is None:
        raise ValueError(
            "La imagen no puede ser None."
        )

    if macula_area_mask is None:
        raise ValueError(
            "La máscara del área macular "
            "no puede ser None."
        )

    # -------------------------------------------------
    # Convertir a escala de grises
    # -------------------------------------------------

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image.copy()

    # -------------------------------------------------
    # Asegurar máscara binaria
    # -------------------------------------------------

    mask = (
        macula_area_mask > 0
    )

    if not np.any(mask):

        raise ValueError(
            "El área macular no contiene "
            "ningún píxel válido."
        )

    # -------------------------------------------------
    # Buscar intensidad mínima
    # -------------------------------------------------

    masked_intensity = gray[
        mask
    ]

    min_intensity = np.min(
        masked_intensity
    )

    # -------------------------------------------------
    # Encontrar coordenadas de todos
    # los píxeles con intensidad mínima
    # -------------------------------------------------

    candidates = np.argwhere(

        mask
        &
        (
            gray
            ==
            min_intensity
        )

    )

    # -------------------------------------------------
    # Si hay varios píxeles con el mismo valor,
    # seleccionar el centroide de los candidatos
    # para evitar depender de un único píxel.
    # -------------------------------------------------

    if len(candidates) > 1:

        new_y = int(
            np.round(
                np.mean(
                    candidates[:, 0]
                )
            )
        )

        new_x = int(
            np.round(
                np.mean(
                    candidates[:, 1]
                )
            )
        )

    else:

        new_y = int(
            candidates[0, 0]
        )

        new_x = int(
            candidates[0, 1]
        )

    new_center = (
        new_y,
        new_x
    )

    return (
        new_center,
        float(min_intensity),
        gray
    )

def visualize_min_intensity_macula_center(
    image,
    macula_area_mask,
    new_center,
    min_intensity,
    output_path,
    original_center=None
):
    """
    Genera una figura científica mostrando:

    1. Imagen original.
    2. Área macular.
    3. Centro anterior.
    4. Nuevo centro basado en mínima intensidad.
    5. Región ampliada.
    """

    import matplotlib.pyplot as plt

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    new_y, new_x = new_center

    fig, axes = plt.subplots(

        1,
        3,

        figsize=(
            18,
            6
        )

    )

    # =================================================
    # PANEL A
    # =================================================

    ax = axes[0]

    ax.imshow(

        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    )

    # Dibujar contorno del área

    contours, _ = cv2.findContours(

        macula_area_mask.astype(
            np.uint8
        ),

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE

    )

    for contour in contours:

        contour = contour.squeeze()

        if len(contour) > 2:

            ax.plot(

                contour[:, 0],

                contour[:, 1],

                linewidth=2,

                label="Macular area"

            )

    # Centro original

    if original_center is not None:

        old_y, old_x = (
            original_center
        )

        ax.scatter(

            old_x,

            old_y,

            s=120,

            marker="o",

            label="Original center"

        )

    # Nuevo centro

    ax.scatter(

        new_x,

        new_y,

        s=180,

        marker="*",

        label="Minimum-intensity center"

    )

    # Línea entre centros

    if original_center is not None:

        ax.plot(

            [
                old_x,
                new_x
            ],

            [
                old_y,
                new_y
            ],

            linestyle="--",

            linewidth=2

        )

    ax.set_title(

        "(A) Macular center localization"

    )

    ax.set_xlabel(
        "X (pixels)"
    )

    ax.set_ylabel(
        "Y (pixels)"
    )

    ax.legend()

    # =================================================
    # PANEL B
    # =================================================

    ax = axes[1]

    masked_image = np.full_like(

        gray,

        np.nan,

        dtype=float

    )

    masked_image[
        macula_area_mask > 0
    ] = gray[
        macula_area_mask > 0
    ]

    im = ax.imshow(

        masked_image,

        cmap="gray"

    )

    ax.scatter(

        new_x,

        new_y,

        s=200,

        marker="*"

    )

    ax.set_title(

        "(B) Intensity distribution "
        "within the estimated macular area"

    )

    ax.set_xlabel(
        "X (pixels)"
    )

    ax.set_ylabel(
        "Y (pixels)"
    )

    plt.colorbar(

        im,

        ax=ax,

        label="Intensity"

    )

    # =================================================
    # PANEL C
    # =================================================

    ax = axes[2]

    window = 50

    y1 = max(
        0,
        new_y - window
    )

    y2 = min(

        gray.shape[0],

        new_y + window

    )

    x1 = max(

        0,

        new_x - window

    )

    x2 = min(

        gray.shape[1],

        new_x + window

    )

    crop = gray[

        y1:y2,

        x1:x2

    ]

    ax.imshow(

        crop,

        cmap="gray"

    )

    ax.scatter(

        new_x - x1,

        new_y - y1,

        s=250,

        marker="*"

    )

    ax.set_title(

        f"(C) Minimum-intensity pixel\n"
        f"Intensity = {min_intensity:.2f}"

    )

    ax.set_xlabel(
        "X (pixels)"
    )

    ax.set_ylabel(
        "Y (pixels)"
    )

    plt.tight_layout()

    plt.savefig(

        output_path,

        dpi=600,

        bbox_inches="tight"

    )

    # PDF para publicación

    pdf_path = str(
        output_path
    ).replace(
        ".png",
        ".pdf"
    )

    plt.savefig(

        pdf_path,

        bbox_inches="tight"

    )

    plt.close()


# ═══════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════
def run(img, out="results"):
    os.makedirs(out, exist_ok=True)
    h, w = img.shape[:2]
    print(f"[*] Imagen: {w}×{h}")

    print("[0] Normalización...")
    img_norm = normalize_fundus(img)
    fov = create_fov_mask(img)
    fa = max(np.count_nonzero(fov), 1)

    print("[*] Estructuras anatómicas...")
    disc_info = detect_optic_disc(img_norm[:,:,1], fov)
    mac_cy, mac_cx, mac_r = estimate_macula_center(img_norm, fov, disc_info)

    print("[1] Segmentación (4 zonas: global+disco+macular+FAZ)...")
    v_raw, diff_img, faz_r = segment_vessels(img_norm, fov, disc_info, (mac_cy,mac_cx,mac_r))
    vp = np.count_nonzero(v_raw)
    print(f"    Vasos: {vp:,} px ({vp/fa*100:.1f}%)")

    print("[2] Postprocesamiento...")
    v_clean = postprocess_vessels(v_raw, fov)
    cp = np.count_nonzero(v_clean)
    print(f"    Clean: {cp:,} px ({cp/fa*100:.1f}%)")
    if cp < vp*0.01 and vp > 500:
        bm = remove_small_objects(v_raw>0, min_size=50*(max(1,min(h,w)//512)**2))
        v_clean = cv2.bitwise_and(bm.astype(np.uint8)*255, fov)

    print("[3] Skeleton (componentes ≥ 20% + poda)...")
    skel = build_skeleton(v_clean)
    print(f"    Skeleton: {np.count_nonzero(skel):,} px")

    print("[4] Keypoints...")
    ep, bf = detect_keypoints(skel)
    print(f"    EP: {len(ep)}, BF: {len(bf)}")

    print("[5] Contorno híbrido (ray-cast + snap a endpoints)...")
    selected, rejected = find_macular_boundary(v_clean, skel, ep, mac_cy, mac_cx, mac_r)

    # Si pocos, incluir bifurcaciones como candidatos de snap
    if len(selected) < 8 and len(bf) > 0:
        print("    Incluyendo bifurcaciones...")
        all_pts = np.vstack([ep, bf]) if len(ep) > 0 else bf
        selected, rejected = find_macular_boundary(v_clean, skel, all_pts, mac_cy, mac_cx, mac_r)

    cent_r, cent_c = mac_cy, mac_cx
    if len(selected) >= 3:
        cent_r = int(selected[:,0].mean())
        cent_c = int(selected[:,1].mean())
        print(f"    Centroide macular: ({cent_c},{cent_r})")

    print("[6] Vectores de intensidad...")
    gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    ivecs = compute_intensity_vectors(gray, cent_r, cent_c, selected)
    print(f"    Vectores: {len(ivecs)}")

    print("[7] Contorno suavizado...")
    contour = build_smooth_contour(selected, cent_r, cent_c)
    if contour is not None: print(f"    Puntos: {len(contour)}")
    else: print("    [WARN] Insuficientes")

    print("[*] Guardando...")
    visualize(img, v_clean, skel, ep, bf,
              (mac_cy,mac_cx), mac_r, selected, rejected, contour,
              diff_img, disc_info, faz_r,
              os.path.join(out, "pipeline_result.png"))
    cv2.imwrite(os.path.join(out, "vessel_clean.png"), v_clean)
    cv2.imwrite(os.path.join(out, "skeleton.png"), skel)
    cv2.imwrite(os.path.join(out, "vessel_enhancement.png"), diff_img)
    if len(ivecs) > 0:
        np.savez(os.path.join(out, "intensity_vectors.npz"),
                 *[v for v in ivecs if len(v) > 0])
    print(f"    → {out}/")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--input", required=True)
    pa.add_argument("--output_dir", default="results")
    a = pa.parse_args()
    im = cv2.imread(a.input)
    if im is None: print(f"[ERROR] {a.input}"); exit(1)
    run(im, a.output_dir)
    print("[✓] Completado.")