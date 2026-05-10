import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm

# Варіант 4: генерація даних
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)


print("=== Варіант 4 ===")
print("Теоретична модель: y = 0.7·x² + 1.0·x + 3.0 + шум\n")

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

print("Лінійна регресія:")
print(f"  Коефіцієнт: {lin_reg.coef_[0][0]:.4f}")
print(f"  Вільний член: {lin_reg.intercept_[0]:.4f}")

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print(f"\nX[0] = {X[0]}")
print(f"X_poly[0] = {X_poly[0]}")

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

a0 = poly_reg.intercept_[0]
a1 = poly_reg.coef_[0][0]
a2 = poly_reg.coef_[0][1]

print("\nПоліноміальна регресія (degree=2):")
print(f"  intercept_ (a0) = {a0:.4f}")
print(f"  coef_[0] (a1)   = {a1:.4f}  (теоретично: 1.0)")
print(f"  coef_[1] (a2)   = {a2:.4f}  (теоретично: 0.7)")
print(f"\n  Модель: y = {a2:.2f}·x² + {a1:.2f}·x + {a0:.2f}")
print(f"  Теорія: y = 0.70·x² + 1.00·x + 3.00")

# --- Метрики ---
print("\nМетрики поліноміальної регресії:")
print("  R2 =", round(sm.r2_score(y, y_poly_pred), 4))
print("  MAE =", round(sm.mean_absolute_error(y, y_poly_pred), 4))
print("  MSE =", round(sm.mean_squared_error(y, y_poly_pred), 4))

# --- Графік ---
X_sort = np.sort(X, axis=0)
X_sort_poly = poly_features.transform(X_sort)
y_sort_lin = lin_reg.predict(X_sort)
y_sort_poly = poly_reg.predict(X_sort_poly)

plt.figure(figsize=(9, 6))
plt.scatter(X, y, color='steelblue', alpha=0.6, label='Дані (Варіант 4)')
plt.plot(X_sort, y_sort_lin, 'r--', linewidth=2, label='Лінійна регресія')
plt.plot(X_sort, y_sort_poly, 'g-', linewidth=2.5, label='Поліноміальна регресія (degree=2)')
plt.xlabel('$x_1$')
plt.ylabel('y')
plt.title('Варіант 4: y = 0.7·x² + x + 3 + шум')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()