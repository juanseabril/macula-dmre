# Avances — curvas de nivel v2

17 de noviembre de 2026 · 7 imágenes STARE · canal verde

## Qué se hizo

1. **Reproducción de v1.** `experiment_contours.py` sobre las 7 imágenes da exactamente el mismo `experiment_contours_green.csv` de julio, con el entorno fijado en `requirements.txt`.
2. **Nuevo `macula_level_curves_v2.py`**, que no modifica v1 y reutiliza los pasos 1 a 9 de `dmre.py`. Cambios:
   - **Supresión de vasos.** Antes de trazar las curvas se aplica un cierre morfológico en gris con un disco de 15 px. Rellena los vasos, que son oscuros y delgados, y conserva la depresión foveal, que es ancha. Así las regiones dejan de fugarse por los vasos.
   - **Centro robusto.** El centro es el mínimo de la imagen sin vasos y suavizada, ya no un solo píxel crudo. Se marca cuando cae en el borde del área del paso 8.
   - **Perfil radial en todas las direcciones.** Se usa la mediana de 72 rayos y el perfil área–nivel, en lugar de una sola recta horizontal.
   - **Modelo de la depresión foveal.** Se ajusta `I(r) = B + c·r − D·exp(−r²/2σ²)`, que da tres descriptores: profundidad `D`, ancho `σ` y radio a media profundidad `r50 = σ·√(2 ln 2)`.
   - **Contorno de media profundidad.** Es la curva de nivel cuyo nivel es `I(r50)`, y une el modelo con las curvas de nivel. Pasa a ser el contorno principal de v2.
   - **Confianza alta, media o baja.** Se calcula con el R² del ajuste, la dispersión de σ entre 8 sectores, la circularidad y solidez del contorno, el acuerdo entre el modelo y el contorno, y si el centro cae en el borde.
   - **Escala física aproximada.** Se usa el campo de 35° de la TopCon TRV-50 y 288 µm por grado.
   - Se conserva el codo (RDP con mayor ángulo de giro) sobre el perfil área–nivel para comparar con v1.

## Resultados

| Imagen | v1: circ. del mejor contorno | v2: circ. del contorno | v2: solidez | v2: r50 (µm, aprox.) | R² del modelo | Confianza v2 |
|---|---|---|---|---|---|---|
| im0031 | 0,70 | 0,86 | 1,00 | 360 | 0,997 | alta |
| im0114 | 0,64 | 0,80 | 0,97 | 741 | 0,992 | alta |
| im0115 | 0,62 | 0,85 | 1,00 | 412 | 0,999 | alta |
| im0163 | 0,13 | 0,50 | 0,87 | 204 | 0,875 | baja |
| im0238 | 0,56 | 0,37 | 0,76 | 437 | 0,996 | baja |
| im0275 | 0,71 | 0,78 | 0,98 | 297 | 0,995 | media |
| im0299 | 0,58 | 0,79 | 0,99 | 486 | 0,997 | media |

- La **mediana de circularidad** sube de 0,62 en v1 a 0,79 en v2, y la mediana de solidez de v2 es 0,98: los contornos ya no tienen ramificaciones.
- El modelo de depresión gaussiana explica el perfil radial con **R² ≥ 0,99 en 6 de las 7 imágenes**. La excepción es im0163, que no muestra fóvea.
- La confianza marca como baja justo los dos casos que se ven mal a simple vista (ver más abajo). Como los umbrales se fijaron mirando estas mismas imágenes, esto todavía no es una validación.

### Ablación: ¿de dónde sale la mejora?

Se corrió el mismo v2 sin supresión de vasos (`--vessel_kernel 0`):

| | Mediana de circularidad | Mediana de solidez |
|---|---|---|
| v1 (mejor contorno) | 0,62 | — |
| v2 sin supresión de vasos | 0,54 | 0,90 |
| v2 completo | 0,79 | 0,98 |

**La mejora viene sobre todo de suprimir los vasos.** Sin ese paso, v2 queda al nivel de v1. Los resultados de la ablación están en `results_v2/ablacion_sin_vasos/`.

### Lectura en escala física

En las cinco imágenes con confianza alta o media, r50 va de unas 300 a unas 740 µm. Eso cae dentro del radio de la **fóvea**, que es de unas 750 µm (diámetro de 1,5 mm según Kolb, Webvision), y queda muy por debajo del radio de la mácula completa, que es de unas 2,75 mm.

Es decir, **lo que capturan las curvas de nivel es la depresión foveal, no el borde de la mácula.** Esto encaja con la ROI actual: el 30 % del radio ocular equivale a unos 1,4 mm, y con ese tamaño no se llega al borde macular.

La escala es aproximada. Supone un campo de 35° y 288 µm por grado (Drasdo y Fowler, 1974), y se reemplazará por la normalización con el diámetro del disco que propone el protocolo MACULA-AI.

### Casos con confianza baja

- **im0163.** La imagen está centrada en el disco y la fóvea no se ve. v1 dibujaba un contorno sobre los vasos (circularidad de 0,13). v2 lo marca como baja por tres motivos: R² de 0,875, centro en el borde del área y σ muy distinto entre sectores.
- **im0238.** La zona oscura se extiende hacia la izquierda, cerca del borde del campo, y el contorno sale irregular (circularidad de 0,37). Lo más probable es que el área del paso 8, y con ella el centro, esté desplazada.

## Limitaciones

- **No hay fóvea anotada.** Circularidad, solidez y R² son indicadores indirectos, no exactitud.
- **Los umbrales de confianza son heurísticos.** Están en `CONF_RULES` y hay que calibrarlos.
- **La escala física sale del campo angular, no del disco óptico.**
- **Son 7 imágenes.** Sirven para desarrollo, no como evidencia.

## Siguientes pasos

1. **Medir con fóvea anotada.** IDRiD, ADAM o REFUGE tienen el centro de la fóvea marcado. Con eso se calcula el error en µm y en diámetros papilares y el porcentaje de aciertos con menos de 0,25 DD, y se calibran los umbrales de confianza. Esos datasets piden registro, así que hay que descargarlos a mano.
2. **Segmentar el disco óptico** para obtener la escala física (protocolo §04.2). Hoy `detect_optic_disc` usa un radio fijo del 8 % del FOV.
3. **Buscar el centro más allá del área del paso 8** cuando el mínimo cae en el borde o el contorno sale irregular, como en im0238.
4. **Nota técnica sobre v1.** `_rdp` en `macula_level_curves.py` usa `np.cross` con vectores 2D. Eso está obsoleto en NumPy 2.0 y falla en versiones recientes (por ejemplo, 2.5). Con `numpy==2.0.2` funciona. v2 trae su propia versión corregida.

## Cómo reproducir

```bash
source .venv/bin/activate
python3 macula_level_curves_v2.py                       # → results_v2/
python3 macula_level_curves_v2.py --vessel_kernel 0 \
        --output_dir results_v2/ablacion_sin_vasos      # ablación
```

Salidas en `results_v2/`:

- `comparacion_v1_vs_v2.png`: v1 frente a v2, una fila por imagen.
- `resumen_v1_vs_v2.csv`: todas las métricas por imagen.
- `<imagen>/v2_<imagen>.png`: panel con las curvas, los perfiles, el modelo y σ por sector.
- `<imagen>/v2_perfil_radial_<imagen>.csv` y `<imagen>/v2_familia_iso_niveles_<imagen>.csv`.
