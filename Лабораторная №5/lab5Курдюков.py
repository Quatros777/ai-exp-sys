%load_ext rpy2.ipython

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Создаем случайные точки вокруг двух центров кластеризации
data, labels = make_blobs(n_samples=58, centers=2, random_state=40)

# Построение графика сгенерированных точек
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.title('График сгенерированных точек')
plt.show()

# Инициализируем веса и коэффициент обучения
weights = np.random.rand(2, 2)
learning_rate = 0.5

def train_kohonen_network(data, weights, learning_rate, epochs=50):
    for epoch in range(epochs):
        for point in data:
            # Рассчитываем расстояния между точкой и весами каждого нейрона
            distances = np.linalg.norm(point - weights, axis=1)
            
            # Находим индекс нейрона, который ближе всего к точке
            winner_index = np.argmin(distances)
            
            weights[winner_index] += learning_rate * (point - weights[winner_index])
        
        # Уменьшим коэффициент обучения
        learning_rate = (50 - epoch) / 100
        
        plt.figure(figsize=(8, 8))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k')
        plt.scatter(weights[:, 0], weights[:, 1], marker='X', s=200, c='r', label='Centroids')
        plt.title(f'Kohonen Network Clustering - Epoch {epoch + 1}')
        plt.legend()
        plt.show()

    return weights

# Обучаем нейронную сеть Кохонена
trained_weights = train_kohonen_network(data, weights, learning_rate)

# Построение итогового графика
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.scatter(trained_weights[:, 0], trained_weights[:, 1], marker='X', s=200, c='r', label='Centroids')
plt.title('Итоговый график')
plt.legend()
plt.show()
