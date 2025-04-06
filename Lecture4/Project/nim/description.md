## [Nim](https://cs50.harvard.edu/ai/2024/projects/4/nim/#nim)

编写一个通过强化学习自学玩Nim游戏的AI。

```yaml
$ python play.py
Playing training game 1
Playing training game 2
Playing training game 3
...
Playing training game 9999
Playing training game 10000
Done training

Piles:
Pile 0: 1
Pile 1: 3
Pile 2: 5
Pile 3: 7

AI's Turn
AI chose to take 1 from pile 2.
```

## [完成时间](https://cs50.harvard.edu/ai/2024/projects/4/nim/#when-to-do-it)

截止时间：[2026年1月1日星期四上午7:59 GMT+8](https://time.cs50.io/20251231T235900Z)

## [如何获取帮助](https://cs50.harvard.edu/ai/2024/projects/4/nim/#how-to-get-help)

1. 通过[Ed](https://cs50.edx.org/ed)提问！
2. 通过CS50的任何[社区](https://cs50.harvard.edu/ai/2024/communities/)提问！

## [背景](https://cs50.harvard.edu/ai/2024/projects/4/nim/#background)

在Nim游戏中，我们开始时有若干堆物体。玩家轮流行动：在自己的回合中，玩家可以从任意一个非空堆中移除任意非负数量的物体。移除最后一个物体的玩家输掉游戏。

你可能想到一些简单的策略：如果只有一堆且剩下三个物体，你的最佳选择是移除两个物体，让对手移除最后一个物体。但如果有多堆物体，策略会变得复杂得多。在这个项目中，我们将通过强化学习构建一个AI来学习这个游戏的策略。通过反复与自己对抗并从经验中学习，最终我们的AI将学会采取哪些行动以及避免哪些行动。

具体来说，我们将使用Q学习。在Q学习中，我们尝试为每个`(状态, 动作)`对学习一个奖励值（一个数字）。导致游戏失败的动作奖励为-1，导致对手失败的动作奖励为1，而让游戏继续的动作即时奖励为0，但会有一些未来的奖励。

在Python程序中如何表示状态和动作？Nim游戏的“状态”就是所有堆的当前大小。例如，状态可能是`[1, 1, 3, 5]`，表示堆0有1个物体，堆1有1个物体，堆2有3个物体，堆3有5个物体。Nim游戏中的“动作”是一对整数`(i, j)`，表示从堆`i`中移除`j`个物体。因此，动作`(3, 5)`表示“从堆3中移除5个物体”。将该动作应用于状态`[1, 1, 3, 5]`将导致新状态`[1, 1, 3, 0]`（相同的状态，但堆3现在为空）。

Q学习的关键公式如下。每次我们在状态`s`中采取动作`a`时，可以更新Q值`Q(s, a)`：

```css
Q(s, a) <- Q(s, a) + alpha * (新价值估计 - 旧价值估计)
```

在上述公式中，`alpha`是学习率（我们有多重视新信息与已有信息的比较）。`新价值估计`表示当前动作的奖励加上玩家将获得的所有未来奖励的估计。`旧价值估计`就是`Q(s, a)`的现有值。通过每次AI采取新动作时应用这个公式，随着时间的推移，我们的AI将开始学习在任何状态下哪些动作更好。

## [开始](https://cs50.harvard.edu/ai/2024/projects/4/nim/#getting-started)

- 从[https://cdn.cs50.net/ai/2023/x/projects/4/nim.zip](https://cdn.cs50.net/ai/2023/x/projects/4/nim.zip)下载分发代码并解压。

## [理解](https://cs50.harvard.edu/ai/2024/projects/4/nim/#understanding)

首先，打开`nim.py`。这个文件中定义了两个类（`Nim`和`NimAI`）以及两个函数（`train`和`play`）。`Nim`、`train`和`play`已经为你实现，而`NimAI`留了一些函数需要你实现。

看看`Nim`类，它定义了Nim游戏的玩法。在`__init__`函数中，注意每个Nim游戏需要跟踪一堆列表、当前玩家（0或1）以及游戏的赢家（如果有的话）。`available_actions`函数返回一个状态下所有可用动作的集合。例如，`Nim.available_actions([2, 1, 0, 0])`返回集合`{(0, 1), (1, 1), (0, 2)}`，因为三个可能的动作是从堆0中移除1或2个物体，或者从堆1中移除1个物体。

其余的函数用于定义游戏玩法：`other_player`函数确定给定玩家的对手，`switch_player`将当前玩家切换到对手，`move`在当前状态上执行一个动作并将当前玩家切换到对手。

接下来，看看`NimAI`类，它定义了我们的AI，它将学会玩Nim。注意在`__init__`函数中，我们从一个空的`self.q`字典开始。`self.q`字典将通过将`(状态, 动作)`对映射到一个数值来跟踪我们的AI学习到的所有当前Q值。作为一个实现细节，尽管我们通常将`状态`表示为列表，但由于列表不能用作Python字典键，我们将在`self.q`中获取或设置值时使用状态的元组版本。

例如，如果我们想将状态`[0, 0, 0, 2]`和动作`(3, 2)`的Q值设置为`-1`，我们会写类似这样的内容：

```perl
self.q[(0, 0, 0, 2), (3, 2)] = -1
```

还要注意，每个`NimAI`对象都有一个`alpha`和`epsilon`值，分别用于Q学习和动作选择。

`update`函数已经为你写好，它接受状态`old_state`、在该状态下采取的动作`action`、执行该动作后的结果状态`new_state`以及采取该动作的即时奖励`reward`。然后，该函数通过首先获取状态和动作的当前Q值（通过调用`get_q_value`），确定最佳可能的未来奖励（通过调用`best_future_reward`），然后使用这两个值更新Q值（通过调用`update_q_value`）来执行Q学习。这三个函数留给你实现。

最后，最后一个未实现的函数是`choose_action`函数，它选择在给定状态下采取的动作（贪婪地或使用epsilon-greedy算法）。

`Nim`和`NimAI`类最终在`train`和`play`函数中使用。`train`函数通过运行`n`个模拟游戏来训练AI，返回完全训练的AI。`play`函数接受一个训练好的AI作为输入，并让人类玩家与AI玩一局Nim游戏。

## [规范](https://cs50.harvard.edu/ai/2024/projects/4/nim/#specification)

完成`nim.py`中`get_q_value`、`update_q_value`、`best_future_reward`和`choose_action`的实现。对于这些函数中的每一个，任何时候函数接受`state`作为输入，你可以假设它是一个整数列表。任何时候函数接受`action`作为输入，你可以假设它是一个整数对`(i, j)`，表示堆`i`和数量`j`。

`get_q_value`函数应该接受`state`和`action`作为输入，并返回相应的状态/动作对的Q值。

- 记住Q值存储在字典`self.q`中。`self.q`的键应该是`(state, action)`对的形式，其中`state`是所有堆大小的元组，`action`是一个元组`(i, j)`，表示一个堆和一个数量。
- 如果`self.q`中不存在状态/动作对的Q值，那么函数应该返回`0`。

`update_q_value`函数接受状态`state`、动作`action`、现有的Q值`old_q`、当前奖励`reward`和未来奖励的估计`future_rewards`，并根据Q学习公式更新状态/动作对的Q值。

- 记住Q学习公式是：`Q(s, a) <- 旧价值估计 + alpha * (新价值估计 - 旧价值估计)`
- 记住`alpha`是与`NimAI`对象关联的学习率。
- 旧价值估计就是状态/动作对的现有Q值。新价值估计应该是当前奖励和估计的未来奖励的总和。

`best_future_reward`函数接受一个`state`作为输入，并根据`self.q`中的数据返回该状态下任何可用动作的最佳可能奖励。

- 对于给定状态下`self.q`中不存在的任何动作，你应该假设它的Q值为0。
- 如果状态下没有可用的动作，你应该返回0。

`choose_action`函数应该接受一个`state`作为输入（以及一个可选的`epsilon`标志，用于是否使用epsilon-greedy算法），并返回该状态下的一个可用动作。

- 如果`epsilon`是`False`，你的函数应该表现得贪婪，并返回该状态下最佳可能的可用动作（即具有最高Q值的动作，如果不知道Q值则使用0）。
- 如果`epsilon`是`True`，你的函数应该根据epsilon-greedy算法行为，以概率`self.epsilon`选择一个随机可用动作，否则选择最佳可用动作。
- 如果多个动作具有相同的Q值，其中任何一个选项都是可接受的返回值。

你不应该修改`nim.py`中规范要求你实现的函数之外的任何内容，尽管你可以编写额外的函数和/或导入其他Python标准库模块。你也可以导入`numpy`或`pandas`，如果熟悉它们的话，但不应使用任何其他第三方Python模块。你可以修改`play.py`来自行测试。

## [提示](https://cs50.harvard.edu/ai/2024/projects/4/nim/#hints)

- 如果`lst`是一个列表，那么`tuple(lst)`可以用来将`lst`转换为元组。

## 