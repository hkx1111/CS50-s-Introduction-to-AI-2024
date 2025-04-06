1. **项目主要目标**：
   - 构建一个k最近邻分类器(k=1)
   - 预测在线购物用户是否会完成购买(Revenue列)
   - 评估模型的敏感性和特异性

2. **核心要求和功能点**：
   - 实现`load_data`函数：读取CSV并返回(evidence, labels)
   - 实现`train_model`函数：训练k=1的KNN分类器
   - 实现`evaluate`函数：计算敏感性和特异性
   - 数据预处理要求：
     - 将Month转换为0-11的数字
     - VisitorType和Weekend转换为0/1
     - 确保各列数据类型正确

3. **限制条件**：
   - 只能使用Python标准库、numpy、pandas和scikit-learn
   - 不能修改shopping.csv文件
   - 不能修改除指定函数外的其他代码
   - 必须使用KNeighborsClassifier(k=1)

4. **评分标准/测试重点**：
   - 数据加载和预处理是否正确
   - 模型训练是否符合规范
   - 评估指标计算是否准确
   - 输出格式是否符合要求
   - 代码风格和可读性

5. **关键CS概念/技术**：
   - 机器学习基础(k-NN算法)
   - 数据预处理和特征工程
   - 分类模型评估指标(敏感性/特异性)
   - CSV文件处理
   - 类型转换和数据处理

6. **常见易错点**：
   - Month列转换容易出错(注意大小写和缩写)
   - VisitorType和Weekend的0/1编码
   - 各列数据类型的正确转换
   - 证据列表的顺序必须与CSV列顺序一致
   - 评估函数中敏感性和特异性的计算公式
   - 边界情况处理(如空值或异常数据)

---

### 实施计划：购物预测项目

#### 1. 任务分解
1. **数据加载与预处理**
   - 读取CSV文件
   - 处理Month列转换
   - 处理VisitorType和Weekend编码
   - 确保各列数据类型正确
   - 构建evidence和labels列表

2. **模型训练**
   - 实现k=1的KNN分类器
   - 拟合训练数据

3. **模型评估**
   - 计算敏感性(真正例率)
   - 计算特异性(真负例率)

4. **测试与验证**
   - 运行完整流程
   - 检查输出格式
   - 验证指标计算

#### 2. 推荐执行顺序及理由
1. 先完成`load_data`函数：
   - 基础性工作，其他部分依赖于此
   - 可以单独测试数据预处理是否正确

2. 接着实现`train_model`：
   - 依赖`load_data`的输出
   - scikit-learn接口相对简单

3. 最后完成`evaluate`：
   - 依赖前两个步骤的结果
   - 需要理解评估指标计算

#### 3. 各步骤详细说明

**步骤1: load_data实现**
- 目标：正确加载和预处理数据
- 输入：shopping.csv文件路径
- 输出：(evidence, labels)元组
- 关键点：
  - 使用csv模块读取文件
  - 实现Month到0-11的映射
  - VisitorType和Weekend的0/1编码
  - 确保各列数据类型符合要求

**步骤2: train_model实现** 
- 目标：训练k=1的KNN分类器
- 输入：evidence列表和labels列表
- 输出：训练好的KNeighborsClassifier实例
- 关键点：
  - 使用KNeighborsClassifier(n_neighbors=1)
  - 调用fit方法进行训练

**步骤3: evaluate实现**
- 目标：计算敏感性和特异性
- 输入：真实labels和预测结果
- 输出：(sensitivity, specificity)元组
- 关键点：
  - 真正例 = 预测为1且实际为1
  - 真负例 = 预测为0且实际为0
  - sensitivity = 真正例数 / 实际正例数
  - specificity = 真负例数 / 实际负例数

#### 4. 关键里程碑
1. 完成`load_data`并通过单元测试
   - 检查evidence列表结构和数据类型
   - 验证Month转换是否正确

2. 完成`train_model`并验证
   - 检查是否能成功训练模型
   - 验证模型类型和参数

3. 完成`evaluate`并验证计算
   - 用简单测试数据验证指标计算
   - 检查边界情况处理

4. 完整流程测试
   - 运行python shopping.py shopping.csv
   - 检查输出格式和合理性

#### 5. 各阶段需要复习的知识点

**数据加载阶段**:
- Python CSV模块使用
- 数据清洗和类型转换
- 特征编码技术

**模型训练阶段**:
- k-NN算法原理
- scikit-learn KNeighborsClassifier使用
- 训练集/测试集划分

**评估阶段**:
- 分类评估指标
- 混淆矩阵概念
- 敏感性和特异性计算

**整体项目**:
- Python列表和元组操作
- 机器学习工作流程
- 代码组织和模块化