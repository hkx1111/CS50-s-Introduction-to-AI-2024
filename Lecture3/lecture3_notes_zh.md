# CS50 人工智能 - Lecture 3 笔记

来源: https://cs50.harvard.edu/ai/2024/notes/3/

## 第三讲

### 优化

优化是从一组可能的选项中选择最佳选项。我们已经遇到过尝试找到最佳可能选项的问题，例如在 minimax 算法中，今天我们将学习可以用来解决更广泛问题的工具。

### 局部搜索

局部搜索是一种搜索算法，它维护单个节点并通过移动到相邻节点进行搜索。这种类型的算法与我们之前看到的搜索类型不同。例如，在迷宫求解中，我们想要找到到达目标的最快方法，而局部搜索则对找到问题的最佳答案感兴趣。通常，局部搜索会带来一个不是最优但“足够好”的答案，从而节省计算能力。考虑以下局部搜索问题的示例：我们有四个位置固定的房屋。我们想要建造两家医院，以便最大限度地减少每栋房屋到医院的距离。这个问题可以可视化如下：

[房屋和医院的图示]

在此图中，我们看到了房屋和医院的可能配置。它们之间的距离使用曼哈顿距离（向上、向下和侧向移动的次数；在第 0 讲中更详细地讨论）来衡量，并且每栋房屋到最近医院的距离之和为 17。我们称之为成本，因为我们试图最大限度地减少这个距离。在这种情况下，状态将是房屋和医院的任何一种配置。

抽象这个概念，我们可以将房屋和医院的每种配置表示为下面的状态空间图。图片中的每个条形代表一个状态的值，在我们的示例中，这将是房屋和医院的特定配置的成本。

[状态空间图]

根据此可视化，我们可以为我们接下来的讨论定义一些重要的术语：

- **目标函数** 是我们用来最大化解决方案值的函数。
- **成本函数** 是我们用来最小化解决方案成本的函数（这是我们将在房屋和医院示例中使用的函数。我们想要最小化房屋到医院的距离）。
- **当前状态** 是函数当前正在考虑的状态。
- **邻居状态** 是当前状态可以转换到的状态。在上面的一维状态空间图中，邻居状态是当前状态任一侧的状态。在我们的示例中，邻居状态可以是由于将其中一家医院向任何方向移动一步而产生的状态。邻居状态通常与当前状态相似，因此，它们的值接近当前状态的值。

请注意，局部搜索算法的工作方式是考虑当前状态中的一个节点，然后将该节点移动到当前状态的邻居之一。这与 minimax 算法不同，例如，在 minimax 算法中，状态空间中的每个状态都被递归地考虑。

### 爬山法

爬山法是一种局部搜索算法。在该算法中，将邻居状态与当前状态进行比较，如果其中任何一个更好，我们将当前节点从当前状态更改为该邻居状态。什么符合更好的条件取决于我们是使用目标函数（偏好更高的值）还是递减函数（偏好更低的值）。

爬山算法的伪代码如下所示：

```
function Hill-Climb(problem):
  current = problem 的初始状态
  repeat:
    neighbor = current 的最佳邻居
    if neighbor 不比 current 好:
      return current
    current = neighbor
```

在该算法中，我们从当前状态开始。在某些问题中，我们将知道当前状态是什么，而在其他问题中，我们将不得不从随机选择一个状态开始。然后，我们重复以下操作：我们评估邻居，选择具有最佳值的邻居。然后，我们将此邻居的值与当前状态的值进行比较。如果邻居更好，我们将当前状态切换到邻居状态，然后重复该过程。当我们将最佳邻居与当前状态进行比较，并且当前状态更好时，该过程结束。然后，我们返回当前状态。

使用爬山算法，我们可以开始改进我们在示例中分配给医院的位置。经过几次转换，我们得到以下状态：

[改进的房屋和医院配置图示]

在此状态下，成本为 11，这比初始状态的成本 17 有所改进。但是，这还不是最佳状态。例如，将左侧的医院移动到左上角房屋的下方将使成本降至 9，这比 11 更好。但是，此版本的爬山算法无法到达那里，因为所有邻居状态的成本至少与当前状态一样高。从这个意义上讲，爬山算法是目光短浅的，通常会满足于比其他一些解决方案更好的解决方案，但不一定是所有可能解决方案中最好的解决方案。

**局部和全局最小值和最大值**

如上所述，爬山算法可能会陷入局部最大值或最小值。局部最大值（复数：maxima）是值高于其邻居状态的状态。与此相反，全局最大值是状态空间中所有状态中值最高的状态。

[局部最大值和全局最大值图示]

相反，局部最小值（复数：minima）是值低于其邻居状态的状态。与此相反，全局最小值是状态空间中所有状态中值最低的状态。

[局部最小值和全局最小值图示]

爬山算法的问题在于它们可能会在局部最小值和最大值处结束。一旦算法到达一个邻居比当前状态更差的点（对于函数的目标而言），算法就会停止。局部最大值和最小值的特殊类型包括平坦局部最大值/最小值，其中多个值相等的
