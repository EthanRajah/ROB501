import matplotlib.pyplot as plt
from imageio import imread
from histogram_eq import histogram_eq

I = imread("../images/uoft_soldiers_tower_light.png")
J = histogram_eq(I)

plt.imshow(I, cmap = "gray", vmin = 0, vmax = 255)
plt.show()
plt.imshow(J, cmap = "gray", vmin = 0, vmax = 255)
plt.show()