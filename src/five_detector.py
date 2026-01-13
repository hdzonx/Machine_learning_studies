# Importações
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# =========================
# 1. Carregar o dataset MNIST
# =========================
mnist = fetch_openml('mnist_784', version=1, as_frame=True)

X = mnist.data           # DataFrame (70000 x 784)
y = mnist.target         # Series (strings)

# =========================
# 2. Dividir em treino e teste (estratificado)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=10000,
    random_state=42,
    stratify=y
)

# =========================
# 3. Criar rótulos binários (é o dígito 5?)
# =========================
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# =========================
# 4. Treinar o classificador SGD
# =========================
sgd_clf = SGDClassifier(
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

sgd_clf.fit(X_train, y_train_5)

# =========================
# 5. Testar com uma imagem
# =========================
some_digit = X_train.iloc[0].to_numpy()

predicao = sgd_clf.predict([some_digit])
print("É o dígito 5?", predicao[0])

# =========================
# 6. Visualizar a imagem
# =========================
plt.imshow(some_digit.reshape(28, 28), cmap="gray")
plt.axis("off")
plt.show()
