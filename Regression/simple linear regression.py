import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 测试数据
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2.2, 2.8, 4.5, 3.7, 5.8])

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(x, y)

# 获取模型参数
intercept = model.intercept_ # 截距
slope = model.coef_[0] # 斜率

# 预测
y_pred = model.predict(x)

# 输出结果
print(f"截距（β0）：{intercept}")
print(f"斜率（β1）：{slope}")
print(f"MSE：{mean_squared_error(y, y_pred)}")
print(f"R^2(决定系数):{r2_score(y, y_pred)}")

# 可视化结果
plt.scatter(x, y, color='blue', label='实际值')
plt.plot(x, y_pred, color='red', label='拟合线')
plt.xlabel('invest amount')
plt.ylabel('profit')
plt.legend()
plt.title('simple linear regression')
plt.show()