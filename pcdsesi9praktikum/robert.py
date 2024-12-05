import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

def roberts_edge_detection(image):
    # Konversi gambar ke grayscale jika berwarna
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    # Membuat kernel Roberts
    roberts_x = np.array([[1, 0], 
                         [0, -1]])
    roberts_y = np.array([[0, 1], 
                         [-1, 0]])
    
    # Mendapatkan ukuran gambar
    rows, cols = image.shape
    
    # Membuat matrix kosong untuk hasil
    edge_x = np.zeros_like(image)
    edge_y = np.zeros_like(image)
    edge_magnitude = np.zeros_like(image)
    
    # Aplikasikan operator Roberts
    for i in range(rows-1):
        for j in range(cols-1):
            # Hitung gradien x dan y
            gx = np.sum(image[i:i+2, j:j+2] * roberts_x)
            gy = np.sum(image[i:i+2, j:j+2] * roberts_y)
            
            # Simpan nilai gradien
            edge_x[i,j] = gx
            edge_y[i,j] = gy
            
            # Hitung magnitude
            edge_magnitude[i,j] = np.sqrt(gx**2 + gy**2)
    
    return edge_magnitude

# Baca gambar
image = imageio.imread('sasuke.png')

# Terapkan deteksi tepi Roberts
edges = roberts_edge_detection(image)

# Tampilkan hasil
plt.figure(figsize=(12,4))

plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Gambar Asli')
plt.axis('off')

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Hasil Deteksi Tepi Roberts')
plt.axis('off')

plt.tight_layout()
plt.show()