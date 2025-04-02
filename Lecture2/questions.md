# 抽样与可能性加权

## 抽样基础

在贝叶斯网络中，抽样是一种近似推断技术，通过生成符合概率分布的随机样本来估计概率值。基本思想是多次随机生成样本，然后统计符合特定条件的样本比例，以此估计概率。

## 拒绝抽样的低效问题

文档中提到的抽样例子使用了一种称为"拒绝抽样"的方法：

```python
# 计算在火车延误的情况下 Appointment 的分布
N = 10000
data = []

# 重复抽样 10,000 次
for i in range(N):
    sample = generate_sample()
    # 丢弃不符合证据的样本
    if sample["train"] == "delayed":
        data.append(sample["appointment"])
```

这种方法的低效之处在于：
1. 我们生成了10,000个完整样本
2. 但只保留了符合"train = delayed"条件的样本
3. 其余不符合条件的样本都被丢弃了

如果火车延误概率很低（例如只有20%），那么大约80%的样本会被丢弃，造成计算资源的浪费。当我们有多个条件需要同时满足时，这种低效会更加明显。

## 可能性加权的解决方案

为解决这个问题，可能性加权方法提供了更高效的替代方案：

1. 固定证据变量值（例如，始终令Train="delayed"）
2. 只对非证据变量进行抽样
3. 对每个样本赋予权重，权重等于证据在其父节点条件下的条件概率

这种方法不会丢弃任何样本，而是通过权重来反映样本的可能性，从而更有效地利用计算资源。

# 马尔可夫链的生成

## 马尔可夫链的产生过程

马尔可夫链是基于转移模型生成的随机状态序列。以文中的天气预测例子为例：

1. 从初始状态开始（晴天或雨天，各有0.5的概率）
2. 根据当前状态和转移概率，确定下一个状态：
   - 如果今天是晴天，明天有0.8的概率是晴天，0.2的概率是雨天
   - 如果今天是雨天，明天有0.3的概率是晴天，0.7的概率是雨天
3. 重复此过程，生成一个状态序列

代码表示为：
```python
model = MarkovChain([start, transitions])
states = model.sample(50)  # 生成50个状态的序列
```

## 多次生成的结果差异

是的，程序多次创建马尔可夫链会产生不同的结果。这是因为：

1. 马尔可夫链的生成过程是**随机**的，每一步都根据概率分布进行抽样
2. 即使使用相同的初始状态和转移模型，随机抽样也会导致不同的状态序列
3. 只有在极长的序列中，各状态出现的频率才会接近其理论概率分布

例如，连续运行`model.sample(50)`多次，会得到不同的50天天气序列。这正是马尔可夫模型的特点：它们遵循概率规律，但每次实现都是随机的、独特的路径。

这种随机性使马尔可夫模型能够模拟真实世界中的随机过程，比如天气变化、股票价格波动或用户行为等。

---

# 可能性加权详解：从零开始理解

## 基本问题情境

让我用一个简化的例子来解释可能性加权的优势。假设我们有以下贝叶斯网络模型：

```
Rain → Train → Appointment
```

其中:
- Rain (下雨): {yes, no}
- Train (火车): {on_time, delayed}
- Appointment (约会): {attend, miss}

## 拒绝抽样过程

假设我们想知道：**已知火车延误，我们参加或错过约会的概率分布是什么？**

使用拒绝抽样，流程是这样的：

```python
# 假设我们要生成1000个样本
samples = []
accepted_count = 0

for i in range(1000):
    # 1. 先抽样Rain (根据其概率分布)
    rain = sample_rain()  # 可能是"yes"或"no"
    
    # 2. 再抽样Train (基于rain的条件)
    train = sample_train(rain)  # 可能是"on_time"或"delayed"
    
    # 3. 再抽样Appointment (基于train的条件)
    appointment = sample_appointment(train)
    
    # 4. 如果不符合证据(train="delayed")，丢弃整个样本
    if train == "delayed":
        samples.append(appointment)
        accepted_count += 1
```

**效率问题：** 如果火车只有20%的概率延误，那么大约80%的样本会被丢弃！

## 可能性加权过程

使用可能性加权，流程变为：

```python
weighted_samples = {"attend": 0, "miss": 0}
total_weight = 0

for i in range(1000):
    # 1. 先抽样Rain (根据其概率分布)
    rain = sample_rain()  # 可能是"yes"或"no"
    
    # 2. 固定Train="delayed"(不抽样)
    train = "delayed"
    
    # 3. 抽样Appointment (基于train="delayed"的条件)
    appointment = sample_appointment(train)
    
    # 4. 计算权重: P(Train="delayed"|Rain=rain)
    weight = probability_train_given_rain(train, rain)
    
    # 5. 将样本加入带权重的计数中
    weighted_samples[appointment] += weight
    total_weight += weight
```

最后，计算概率分布：
```python
P_attend = weighted_samples["attend"] / total_weight
P_miss = weighted_samples["miss"] / total_weight
```

## 具体数值例子

假设我们有以下概率:
- P(Rain=yes) = 0.3, P(Rain=no) = 0.7
- P(Train=delayed|Rain=yes) = 0.6, P(Train=delayed|Rain=no) = 0.1
- P(Appointment=attend|Train=delayed) = 0.6, P(Appointment=miss|Train=delayed) = 0.4

### 拒绝抽样中：
假设我们生成1000个样本:
- 约300个样本Rain=yes，其中约180个Train=delayed
- 约700个样本Rain=no，其中约70个Train=delayed
- 总共约250个样本Train=delayed (被接受)，750个样本被丢弃

### 可能性加权中：
我们生成1000个样本，每个样本都使用:
- 如果抽到Rain=yes (约300次)：权重=0.6
- 如果抽到Rain=no (约700次)：权重=0.1

所有1000个样本都被保留并计入结果，只是权重不同。

## 优势总结

1. **计算效率更高**：不浪费任何样本计算
2. **样本利用率100%**：所有生成的样本都有助于最终结果
3. **适用于稀有事件**：当证据事件概率很低时尤其有效
4. **可扩展性更好**：当有多个证据条件时，拒绝抽样的效率会急剧下降

可能性加权本质上是"不要丢弃样本，而是根据其符合证据的可能性给予权重"的思想，这在处理复杂概率模型时特别有价值。

---

# 正确答案解析

正确的句子是：**Assuming we know the train is on time, whether or not there is track maintenance does not affect the probability that the appointment is attended.**

翻译：假设我们知道火车准时，是否有轨道维护不会影响出席约会的概率。

## 为什么这个答案是正确的？

这个问题本质上是测试我们对贝叶斯网络中的**条件独立性**的理解。在贝叶斯网络中，通过查看网络结构，我们可以判断给定某些变量的情况下，其他变量之间是否独立。

让我们回顾一下这个贝叶斯网络的结构：
```
Rain → Maintenance
   ↘   ↓
     Train → Appointment
```

在这个网络中：
1. Rain（下雨）和Maintenance（轨道维护）都能影响Train（火车是否准时）
2. Train直接影响Appointment（是否参加约会）

**关键点**：根据贝叶斯网络中的条件独立性规则，**当我们已知一个节点的值时，它会"阻断"其父节点对其子节点的影响**。

因此，当我们已知Train=on_time（火车准时）时：
- Appointment变量只依赖于Train变量
- Rain和Maintenance（Train的父节点）不再对Appointment有影响

换句话说，**一旦你知道火车准时了，那么引起火车准时的原因（是否下雨、是否有轨道维护）对你能否参加约会没有任何进一步的影响**。

这符合我们的直觉：如果你只关心能否赴约，而你已经知道火车准时了，那么你不需要关心天气如何或者铁路是否在维护 - 这些因素已经通过"火车准时"这个信息被完全考虑进去了。