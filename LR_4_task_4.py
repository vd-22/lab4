import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)


ypred = regr.predict(Xtest)

print("Коефіцієнти регресії (regr.coef_):")
print(regr.coef_)
print("\nВільний член (regr.intercept_):", regr.intercept_)

r2 = r2_score(ytest, ypred)
mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)

print(f"\nR2 score:              {r2:.4f}")
print(f"Mean Absolute Error:   {mae:.4f}")
print(f"Mean Squared Error:    {mse:.4f}")


fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), alpha=0.7, label='Передбачення')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, label='Ідеальна пряма')
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
ax.set_title(f'Лінійна регресія (Diabetes)\nR²={r2:.3f}  MAE={mae:.1f}  MSE={mse:.1f}')
ax.legend()
plt.tight_layout()
plt.show()