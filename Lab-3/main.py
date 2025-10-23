import numpy as np
import skfuzzy as fuzz
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# ЕТАП 1. Генерація даних

centers = [
    [15, 10],   # кластер 1: низьке навантаження
    [45, 40],   # кластер 2: середнє навантаження
    [80, 70]    # кластер 3: високе навантаження
]

data, _ = make_blobs(
    n_samples=400,
    centers=centers,
    cluster_std=[3, 5, 9],
    random_state=42
)

plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c='lightgreen', s=30, edgecolors='k')
plt.title('Дані про завантаження серверів')
plt.xlabel('CPU навантаження (%)')
plt.ylabel('Використання пам’яті (%)')
plt.grid(True)
plt.show()

# ЕТАП 2. Алгоритм нечіткої кластеризації Fuzzy C-Means

print("\n=== Виконується нечітка кластеризація методом FCM ===")
center, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data.T,
    c=3,
    m=3,
    error=0.005,
    maxiter=100
)

print("\nКоординати знайдених центрів кластерів:")
for i, c in enumerate(center):
    print(f"  Центр {i + 1}: ({c[0]:.3f}, {c[1]:.3f})")

# ЕТАП 3. Формування міток кластерів

fuzzy_labels = np.argmax(u, axis=0)

for i in range(3):
    count = np.sum(fuzzy_labels == i)
    print(f"  Кількість об'єктів у кластері {i + 1}: {count}")

# ЕТАП 4. Візуалізація результатів кластеризації

plt.figure(figsize=(8, 6))
colors = ['lightcoral', 'lightgreen', 'lightskyblue']

for i in range(3):
    cluster_points = data[fuzzy_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=40, color=colors[i], label=f'Кластер {i + 1}')

plt.scatter(center[:, 0], center[:, 1], marker='*', color='black', s=200, label='Центри кластерів')

plt.title('Результати нечіткої кластеризації (Fuzzy C-Means)')
plt.xlabel('Ознака X1')
plt.ylabel('Ознака X2')
plt.legend()
plt.grid(True)
plt.show()

# ЕТАП 5. Аналіз збіжності алгоритму

plt.figure(figsize=(8, 5))
plt.plot(jm, marker='o', linestyle='-', linewidth=2)
plt.title('Зміна значень цільової функції при ітераціях')
plt.xlabel('Номер ітерації')
plt.ylabel('Значення цільової функції (Jm)')
plt.grid(True)
plt.show()

print("\nОстаточне значення цільової функції:", round(jm[-1], 4))
print("Коефіцієнт розмитості (FPC):", round(fpc, 4))
