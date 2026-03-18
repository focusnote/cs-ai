# KNN算法代码案例
## sklearn实现
### 分类问题和回归问题的实现
**导包**
```python
# 导入KNN分类器和回归器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
```
**样本定义**
```python
# 定义样本特征和标签
X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 1, 2, 3]
```
- 分类问题
```python
# 创建KNN分类器
knn_clf = KNeighborsClassifier(n_neighbors=3)
# 拟合模型
knn_clf.fit(X, y)
# 预测新样本
new_sample = [[1.5, 1.5]]
predicted_class = knn_clf.predict(new_sample)
print("预测类别:", predicted_class)
```
- 回归问题
```python
# 创建KNN回归器
knn_reg = KNeighborsRegressor(n_neighbors=3)
# 拟合模型
knn_reg.fit(X, y)
# 预测新样本
new_sample = [[1.5, 1.5]]
predicted_value = knn_reg.predict(new_sample)
print("预测值:", predicted_value)
```

### 数据预处理
**导包**
```python
# 导入归一化工具
from sklearn.preprocessing import MinMaxScaler

# 导入数据标准化工具
from sklearn.preprocessing import StandardScaler
```
**样本定义**
```python
# 定义样本特征
X_train = [[0, 0], [1, 1], [2, 2], [3, 3]]
```
- 归一化
```python
# 创建归一化器
scaler = MinMaxScaler()
# 拟合并转换训练数据
X_train_new = scaler.fit_transform(X_train)
print("归一化后的训练数据:", X_train_new)
```
- 标准化
```python
# 创建标准化器
scaler = StandardScaler()
# 拟合并转换训练数据
X_train_new = scaler.fit_transform(X_train)
print("标准化后的训练数据:", X_train_new)
```
