# 规范

完成 `joint_probability`、`update` 和 `normalize` 函数的实现。

## `joint_probability` 函数

该函数应接收一个包含人物信息的字典、关于每个人拥有各种基因拷贝数量的数据以及谁表现出该性状的数据作为输入。函数应返回所有这些事件发生的联合概率。

该函数接受四个值作为输入：`people`、`one_gene`、`two_genes` 和 `have_trait`。

*   `people`：一个人物字典，如“理解”部分所述。键代表姓名，值是包含 `mother` 和 `father` 键的字典。您可以假设 `mother` 和 `father` 要么都为空（数据集中没有父母信息），要么都将引用 `people` 字典中的其他人。
*   `one_gene`：一个集合，包含所有我们想要计算其拥有一份基因拷贝概率的人。
*   `two_genes`：一个集合，包含所有我们想要计算其拥有两份基因拷贝概率的人。
*   `have_trait`：一个集合，包含所有我们想要计算其具有该性状概率的人。

对于任何不在 `one_gene` 或 `two_genes` 中的人，我们想要计算他们没有基因拷贝的概率；对于任何不在 `have_trait` 中的人，我们想要计算他们不具有该性状的概率。

例如，如果家庭成员包括 Harry、James 和 Lily，那么在 `one_gene = {"Harry"}`、`two_genes = {"James"}` 和 `trait = {"Harry", "James"}` 的情况下调用此函数，应该计算以下事件的概率：Lily 拥有零份基因拷贝，Harry 拥有一份基因拷贝，James 拥有两份基因拷贝，Harry 表现出该性状，James 表现出该性状，而 Lily 不表现出该性状。

*   对于数据集中没有列出父母的任何人，使用概率分布 `PROBS["gene"]` 来确定他们拥有特定数量基因的概率。
*   对于数据集中有父母的任何人，每个父母会随机将他们的两个基因中的一个传给孩子，并且有 `PROBS["mutation"]` 的概率发生突变（从有该基因变为没有该基因，或反之亦然）。
*   使用概率分布 `PROBS["trait"]` 来计算一个人是否具有特定性状的概率。

## `update` 函数

该函数将一个新的联合分布概率添加到 `probabilities` 中现有的概率分布中。

该函数接受五个值作为输入：`probabilities`、`one_gene`、`two_genes`、`have_trait` 和 `p`。

*   `probabilities`：一个人物字典，如“理解”部分所述。每个人都映射到一个 "gene" 分布和一个 "trait" 分布。
*   `one_gene`：在当前联合分布中拥有一份基因拷贝的人的集合。
*   `two_genes`：在当前联合分布中拥有两份基因拷贝的人的集合。
*   `have_trait`：在当前联合分布中具有该性状的人的集合。
*   `p`：联合分布的概率。

对于 `probabilities` 中的每个人 `person`，该函数应通过将 `p` 添加到每个分布中的适当值来更新 `probabilities[person]["gene"]` 分布和 `probabilities[person]["trait"]` 分布。所有其他值应保持不变。

例如，如果 "Harry" 同时在 `two_genes` 和 `have_trait` 中，那么 `p` 将被添加到 `probabilities["Harry"]["gene"][2]` 和 `probabilities["Harry"]["trait"][True]` 中。

该函数不应返回任何值：它只需要更新 `probabilities` 字典。

## `normalize` 函数

该函数更新一个概率字典，使得每个概率分布都被归一化（即，总和为 1，且相对比例保持不变）。

该函数接受单个值：`probabilities`。

*   `probabilities`：一个人物字典，如“理解”部分所述。每个人都映射到一个 "gene" 分布和一个 "trait" 分布。

对于 `probabilities` 中每个人的两个分布，此函数应归一化该分布，使得分布中的值总和为 1，并且分布中的相对值保持不变。

例如，如果 `probabilities["Harry"]["trait"][True]` 等于 0.1 且 `probabilities["Harry"]["trait"][False]` 等于 0.3，那么您的函数应将前一个值更新为 0.25，后一个值更新为 0.75：数字现在总和为 1，并且后一个值仍然是前一个值的三倍。

该函数不应返回任何值：它只需要更新 `probabilities` 字典。

## 其他说明

您不应修改 `heredity.py` 中除规范要求您实现的三个函数之外的任何其他内容，但您可以编写额外的函数和/或导入其他 Python 标准库模块。如果您熟悉 numpy 或 pandas，也可以导入它们，但不应使用任何其他第三方 Python 模块。

---

## 最终目标

这个程序的最终目标是，对于这个家庭中的每一个人，分别计算出：

*   **基因概率分布**: 这个人拥有 0 个、1 个、或 2 个致病基因的概率分别是多少？
    *   即 P(张三有 0 个基因 | 所有已知信息)
    *   P(张三有 1 个基因 | 所有已知信息)
    *   P(张三有 2 个基因 | 所有已知信息) 这三个概率加起来应该等于 1。
*   **性状概率分布**: 这个人表现出该性状的概率是多少？不表现出该性状的概率是多少？
    *   即 P(张三表现出性状 | 所有已知信息)
    *   P(张三不表现出性状 | 所有已知信息) 这两个概率加起来也应该等于 1。

这里的“所有已知信息”包括：家庭成员关系（父母是谁）、部分成员已知的性状表现、以及 `PROBS` 字典里定义的先验概率和遗传规则。

这些我们最终想求的、针对单个个体的概率，就叫做**边际概率 (Marginal Probability)**。

## 为什么要通过 update 累加联合概率 p 来计算边际概率？

这是因为我们无法直接计算边际概率。我们能直接计算的是联合概率 `p`，也就是一个非常具体的、涉及所有人状态的完整场景发生的概率。

想象一下，我们想知道“张三有 1 个基因”这个事件发生的概率 P(张三=1基因)。这个事件可以在很多不同的“世界状态”下发生：

*   世界状态1: {张三=1基因, 李四=0基因, ..., 张三=有性状, 李四=无性状, ...}，其概率为 p1
*   世界状态2: {张三=1基因, 李四=2基因, ..., 张三=无性状, 李四=无性状, ...}，其概率为 p2
*   世界状态3: {张三=1基因, 李四=1基因, ..., 张三=有性状, 李四=有性状, ...}，其概率为 p3
*   ... 还有很多很多其他可能的状态 ...

根据**全概率定律 (Law of Total Probability)**，一个事件（张三=1基因）的总概率，等于所有包含该事件的、互斥的、完备的场景（即我们这里的一个个“世界状态”）的概率之和。

P(张三=1基因) = p1 + p2 + p3 + ... (所有包含“张三=1基因”的世界状态的联合概率之和)

`update` 函数的作用正是执行这个求和过程！

`main` 函数中的三重 `powerset` 循环，就是在系统性地生成每一个可能的世界状态。

`joint_probability` 计算出当前这个世界状态发生的概率 `p`。

`update` 函数检查：在当前这个世界状态下，张三是不是正好有 1 个基因？

*   如果是，就把这个世界状态的概率 `p` 加到 `probabilities[张三]["gene"][1]` 这个计数器上。
*   如果不是（比如张三在这个状态下有0个或2个基因），那就不加到这个计数器上（但可能会加到 `probabilities[张三]["gene"][0]` 或 `probabilities[张三]["gene"][2]` 上）。

当所有的 `powerset` 循环跑完后，`probabilities[张三]["gene"][1]` 这个计数器里累积的值，就正好是所有包含“张三=1基因”的世界状态的联合概率之和。这正是我们想要的（未归一化的）边际概率 P(张三=1基因)。

对其他的基因数（0, 2）和性状（True, False）也是完全一样的逻辑。`update` 通过累加，将复杂的联合概率信息“投影”到了我们关心的单个个体的边际概率上。

最后，`normalize` 函数再把这些累加起来的“概率质量”转换成标准的、总和为 1 的概率分布。

## 总结

我们之所以要累加联合概率，是因为这是计算边际概率的标准方法（全概率定律）。我们无法直接计算 P(张三=1基因)，但我们可以计算 P(张三=1基因 并且 其他人状态=X 并且 全家性状=Y)，然后把所有这些情况加起来。`update` 就是在执行这个“加起来”的操作。最终目标是得到每个人的边际概率分布。