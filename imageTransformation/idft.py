import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/lena.png',0)

dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
plt.subplot(221)
plt.axis('off')
plt.title('dft')
plt.imshow(20*np.log(cv2.magnitude(dft[:,:,0],dft[:,:,1])),cmap='gray')

dft_shift = np.fft.fftshift(dft)
plt.subplot(222)
plt.axis('off')
plt.title('dft_shift')
plt.imshow(20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])),cmap='gray')

idft_shift = np.fft.ifftshift(dft_shift)
plt.subplot(223)
plt.axis('off')
plt.title('origin')
plt.imshow(img,cmap='gray')

idft = cv2.idft(idft_shift)
plt.subplot(224)
plt.axis('off')
plt.title('idft_shift')
plt.imshow(cv2.magnitude(idft[:,:,0],idft[:,:,1]),cmap='gray')

plt.show()
