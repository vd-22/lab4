import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Варіант 4: генерація даних
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)
y = y.ravel()  # перетворення у 1D

# --- Функція побудови кривих навчання ---
def plot_learning_curves(model, X, y, title=''):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_errors, val_errors = [], []
    for m_i in range(1, len(X_train)):
        model.fit(X_train[:m_i], y_train[:m_i])
        y_train_predict = model.predict(X_train[:m_i])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m_i], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='Навчальний набір')
    plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='Перевірочний набір')
    plt.xlabel('Розмір навчального набору')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 3.5)

# === 1. Лінійна регресія ===
plt.figure(figsize=(8, 5))
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, title='Криві навчання — Лінійна модель (Варіант 4)')
plt.tight_layout()
plt.show()

# === 2. Поліноміальна регресія степеня 10 ===
plt.figure(figsize=(8, 5))
polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_10, X, y,
                     title='Криві навчання — Поліноміальна модель degree=10 (Варіант 4)')
plt.tight_layout()
plt.show()

# === 3. Поліноміальна регресія степеня 2 ===
plt.figure(figsize=(8, 5))
polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_2, X, y,
                     title='Криві навчання — Поліноміальна модель degree=2 (Варіант 4)')
plt.tight_layout()
plt.show()

print("Висновок:")
print("- Лінійна модель: обидві криві стабілізуються на відносно великій помилці → недонавчання.")
print("- Поліноміальна degree=10: великий розрив між кривими → перенавчання.")
print("- Поліноміальна degree=2: найкращий баланс, криві сходяться і мають малу помилку.")