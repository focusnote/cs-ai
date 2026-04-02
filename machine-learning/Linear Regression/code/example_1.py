import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1) 生成模拟数据
rng = np.random.RandomState(42)
X = 2 * rng.rand(100, 1)
y = 4 + 3 * X.squeeze() + rng.randn(100) * 0.5

# 2) 普通最小二乘（解析解）
X_b = np.c_[np.ones((100, 1)), X]  # 添加偏置项
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print('解析解 theta:', theta_best)

# 3) sklearn 线性回归拟合
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
print('sklearn intercept:', model.intercept_)
print('sklearn coef:', model.coef_)

# 4) 预测并评估
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)
print('预测值 (X_new):', y_pred)

# 5) 可视化结果
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.6, label='训练数据')
plt.plot(X_new, y_pred, 'r-', linewidth=2, label='线性拟合')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归示例')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6) 手动梯度下降实现（可选）
eta = 0.1
n_iterations = 1000
theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((100, 1)), X]
y_reshaped = y.reshape(-1, 1)
for iteration in range(n_iterations):
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y_reshaped)
    theta = theta - eta * gradients

print('梯度下降 theta:', theta.ravel())
print('梯度下降预测 (X_new):', np.c_[np.ones((2, 1)), X_new].dot(theta).ravel())
