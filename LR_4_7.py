import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ЗАВДАННЯ 2 — Лінійна регресія (варіант 4)
# X: 2, 4, 6, 8, 10, 12
# Y: 6.5, 4.4, 3.8, 3.5, 3.1, 3.0
# ==========================================

print("=" * 45)
print("ЗАВДАННЯ 2 — Метод найменших квадратів")
print("=" * 45)

X2 = np.array([2, 4, 6, 8, 10, 12], dtype=float)
Y2 = np.array([6.5, 4.4, 3.8, 3.5, 3.1, 3.0], dtype=float)

# Знаходимо коефіцієнти лінійної регресії y = b0 + b1*x
n = len(X2)
b1 = (n * np.sum(X2 * Y2) - np.sum(X2) * np.sum(Y2)) / \
     (n * np.sum(X2**2) - np.sum(X2)**2)
b0 = (np.sum(Y2) - b1 * np.sum(X2)) / n

print(f"Коефіцієнт b0 (зсув):  {b0:.4f}")
print(f"Коефіцієнт b1 (нахил): {b1:.4f}")
print(f"Рівняння регресії: y = {b0:.4f} + ({b1:.4f})*x")

# Прогнозовані значення
Y2_pred = b0 + b1 * X2

# Похибки
errors = Y2 - Y2_pred
sse = np.sum(errors**2)
print(f"Сума квадратів похибок (SSE): {sse:.4f}")

# Графік
plt.figure(figsize=(8, 5))
plt.scatter(X2, Y2, color='red', zorder=5, label='Експериментальні точки')
x_line = np.linspace(0, 14, 100)
y_line = b0 + b1 * x_line
plt.plot(x_line, y_line, color='blue', label=f'y = {b0:.2f} + ({b1:.2f})x')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Завдання 2 — Лінійна регресія (варіант 4)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('завдання2.png')
plt.show()
print("Графік збережено: завдання2.png")


# ==========================================
# ЗАВДАННЯ 3 — Інтерполяція полінома 4-го степеня
# x: 0.1, 0.3, 0.4, 0.6, 0.7
# y: 3.2, 3.0, 1.0, 1.8, 1.9
# ==========================================

print("\n" + "=" * 45)
print("ЗАВДАННЯ 3 — Інтерполяція")
print("=" * 45)

x3 = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y3 = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

# Крок 1 — Заповнення матриці Вандермонда X
n3 = len(x3)
X_matrix = np.zeros((n3, n3))
for i in range(n3):
    for j in range(n3):
        X_matrix[i, j] = x3[i] ** j

print("Матриця X (Вандермонда):")
print(np.round(X_matrix, 4))

# Крок 2 — Знаходимо коефіцієнти полінома
coeffs = np.linalg.solve(X_matrix, y3)
print(f"\nКоефіцієнти полінома [a0, a1, a2, a3, a4]:")
for i, c in enumerate(coeffs):
    print(f"  a{i} = {c:.4f}")

# Крок 3 — Функція полінома 4-го степеня
def polynom(x, a):
    return a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 + a[4]*x**4

# Крок 4 — Графік
x_plot = np.linspace(0.0, 0.8, 200)
y_plot = polynom(x_plot, coeffs)

plt.figure(figsize=(8, 5))
plt.scatter(x3, y3, color='red', zorder=5, s=80, label='Вузли інтерполяції')
plt.plot(x_plot, y_plot, color='green', label='Інтерполяційний поліном 4-го степеня')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Завдання 3 — Інтерполяція')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('завдання3.png')
plt.show()
print("Графік збережено: завдання3.png")

# Крок 5 — Значення в проміжних точках 0.2 і 0.5
val_02 = polynom(0.2, coeffs)
val_05 = polynom(0.5, coeffs)
print(f"\nЗначення полінома в проміжних точках:")
print(f"  P(0.2) = {val_02:.4f}")
print(f"  P(0.5) = {val_05:.4f}")