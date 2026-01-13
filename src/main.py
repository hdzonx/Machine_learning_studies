import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1)

X, y = mnist["data"], mnist["target"]

some_digit = X.iloc[0].to_numpy()
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

print(y[0])

y = y.astype(np.uint8)

print(y)