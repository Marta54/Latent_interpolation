from image_preprocess import make_square
import matplotlib.pyplot as plt

path = "C:\\Users\\msro1\\Downloads\\archive (1)\\images\\Zweilous\\1.jpg"
img = make_square(path, size = 512)
plt.imshow(img)