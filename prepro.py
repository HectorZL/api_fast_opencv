import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Crear carpeta de salida si no existe
output_dir = "foto_mejorada_prueba"
os.makedirs(output_dir, exist_ok=True)

# Cargar la imagen
img = cv2.imread('tufoto.jpg')

if img is None:
    print("Error: No se pudo cargar la imagen.")
else:
    # 1. Convertir a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, '1_grises.jpg'), gray_img)

    # 2. Reducción de ruido (filtro de mediana)
    denoised_img = cv2.medianBlur(gray_img, 5)
    cv2.imwrite(os.path.join(output_dir, '2_denoised.jpg'), denoised_img)

    # 3. Mejora de contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(denoised_img)
    cv2.imwrite(os.path.join(output_dir, '3_contraste_mejorado.jpg'), enhanced_img)

    # 4. Umbralización (Otsu)
    ret, binary_img = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Operaciones morfológicas (apertura)
    kernel = np.ones((3,3),np.uint8)
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_dir, '4_apertura.jpg'), opened_img)

    # 6. Detección de bordes (Canny)
    edges = cv2.Canny(opened_img, 100, 200)
    cv2.imwrite(os.path.join(output_dir, '5_bordes.jpg'), edges)

    # Guardar imagen original convertida a RGB para visualización
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_dir, '0_original.jpg'), img_rgb)

    # Mostrar resultados
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1), plt.imshow(img_rgb), plt.title('Original')
    plt.subplot(2, 3, 2), plt.imshow(gray_img, cmap='gray'), plt.title('Escala de Grises')
    plt.subplot(2, 3, 3), plt.imshow(denoised_img, cmap='gray'), plt.title('Denoised (Median Blur)')
    plt.subplot(2, 3, 4), plt.imshow(enhanced_img, cmap='gray'), plt.title('Contraste Mejorado (CLAHE)')
    plt.subplot(2, 3, 5), plt.imshow(opened_img, cmap='gray'), plt.title('Binaria y Apertura')
    plt.subplot(2, 3, 6), plt.imshow(edges, cmap='gray'), plt.title('Bordes (Canny)')

    plt.tight_layout()
    plt.show()
