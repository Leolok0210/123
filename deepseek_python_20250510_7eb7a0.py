import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 数据加载与预处理
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 特征工程
X_train = train_df.drop(['Id', 'Species'], axis=1)
y_train = train_df['Species']
X_test = test_df.drop('Id', axis=1)

# 模型训练
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    min_samples_split=3,
    random_state=2023
)
rf_model.fit(X_train, y_train)

# 生成标准格式提交文件
submission = pd.DataFrame({
    'Id': test_df['Id'].astype(int),  # 确保ID为整数类型
    'Species': rf_model.predict(X_test)  # 自动匹配原始标签
})

# 格式验证
assert list(submission.columns) == ['Id', 'Species'], "列名不匹配"
assert submission.shape[1] == 2, "列数错误"
assert submission['Id'].is_monotonic_increasing, "ID未排序"

# 保存结果
submission.to_csv('submission.csv', index=False)

print("生成提交文件示例：")
print(submission.head(4).to_string(index=False))