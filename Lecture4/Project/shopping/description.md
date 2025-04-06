
## [购物预测](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#shopping)

编写一个AI来预测在线购物用户是否会完成购买。

```yaml
$ python shopping.py shopping.csv
正确: 4088
错误: 844
真正例率: 41.02%
真负例率: 90.55%
```

## [截止时间](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#when-to-do-it)

北京时间 [2026年1月1日 星期四 上午7:59](https://time.cs50.io/20251231T235900Z)

## [获取帮助](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#how-to-get-help)

1. 通过[Ed](https://cs50.edx.org/ed)提问
2. 通过CS50的[社区](https://cs50.harvard.edu/ai/2024/communities/)提问

## [项目背景](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#background)

当用户在线购物时，并非所有人最终都会完成购买。事实上，大多数访问在线购物网站的用户很可能不会在当次浏览会话中完成购买。然而，如果网站能够预测用户是否有购买意图可能会很有用：例如向用户显示不同的内容，如果网站认为用户不打算完成购买，可以向用户显示折扣优惠。网站如何确定用户的购买意图？这就是机器学习发挥作用的地方。

你在这个问题中的任务是构建一个最近邻分类器来解决这个问题。给定关于用户的信息——他们访问了多少页面、是否在周末购物、使用什么网络浏览器等——你的分类器将预测用户是否会进行购买。你的分类器不会完全准确——完美建模人类行为远远超出了本课程的范围——但它应该比随机猜测更好。为了训练你的分类器，我们将为你提供来自约12,000个用户会话的购物网站数据。

我们如何衡量这样一个系统的准确性？如果我们有一个测试数据集，我们可以在数据上运行我们的分类器，并计算我们正确分类用户意图的时间比例。这将给我们一个单一的准确率百分比。但这个数字可能有点误导。例如，假设大约15%的用户最终完成了购买。一个总是预测用户不会完成购买的分类器，我们会测量其准确率为85%：它唯一分类错误的用户是那15%确实完成购买的用户。虽然85%的准确率听起来不错，但这似乎不是一个非常有用的分类器。

相反，我们将测量两个值：敏感性（也称为"真正例率"）和特异性（也称为"真负例率"）。敏感性指的是正确识别的正例比例：换句话说，确实完成购买的用户中被正确识别的比例。特异性指的是正确识别的负例比例：在这种情况下，没有完成购买的用户中被正确识别的比例。因此，我们之前的"总是猜测否"的分类器将具有完美的特异性（1.0）但没有敏感性（0.0）。我们的目标是构建一个在这两个指标上都表现合理的分类器。

## [开始项目](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#getting-started)

- 从[https://cdn.cs50.net/ai/2023/x/projects/4/shopping.zip](https://cdn.cs50.net/ai/2023/x/projects/4/shopping.zip)下载分发代码并解压
- 运行`pip3 install scikit-learn`安装`scikit-learn`包（如果尚未安装）

## [理解项目](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#understanding)

首先打开`shopping.csv`，这是为本项目提供的数据集。你可以在文本编辑器中打开它，但在电子表格应用程序（如Microsoft Excel、Apple Numbers或Google Sheets）中查看可能更容易理解。

这个电子表格中大约有12,000个用户会话：每个用户会话表示为一行。前六列测量用户在会话中访问的不同类型页面：`Administrative`、`Informational`和`ProductRelated`列测量用户访问的这些类型页面的数量，它们对应的`_Duration`列测量用户在这些页面上花费的时间。`BounceRates`、`ExitRates`和`PageValues`列测量来自Google Analytics的关于用户访问页面的信息。`SpecialDay`是一个值，测量用户会话日期与特殊日子（如情人节或母亲节）的接近程度。`Month`是用户访问月份的缩写。`OperatingSystems`、`Browser`、`Region`和`TrafficType`都是描述用户自身信息的整数。`VisitorType`对于回头客将取值`Returning_Visitor`，对于非回头客将取其他字符串值。`Weekend`根据用户是否在周末访问而取值`TRUE`或`FALSE`。

也许最重要的列是最后一列：`Revenue`列。这列指示用户最终是否进行了购买：`TRUE`表示进行了购买，`FALSE`表示没有。这是我们希望基于所有其他列（"证据"）的值来预测的列（"标签"）。

接下来，看看`shopping.py`。`main`函数通过调用`load_data`函数从CSV电子表格加载数据，并将数据拆分为训练集和测试集。然后调用`train_model`函数在训练数据上训练机器学习模型。接着，该模型用于在测试数据集上进行预测。最后，`evaluate`函数确定模型的敏感性和特异性，然后将结果打印到终端。

函数`load_data`、`train_model`和`evaluate`是空白的。这就是你需要完成的部分！

## [项目规范](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#specification)

完成`shopping.py`中`load_data`、`train_model`和`evaluate`的实现。

`load_data`函数应接受一个CSV文件名作为参数，打开该文件，并返回一个元组`(evidence, labels)`。`evidence`应该是所有数据点的证据列表，`labels`应该是所有数据点的标签列表。

- 由于电子表格的每一行都有一个证据和一个标签，`evidence`列表和`labels`列表的长度最终应等于CSV电子表格中的行数（不包括标题行）。列表应按用户在电子表格中出现的顺序排序。也就是说，`evidence[0]`应该是第一个用户的证据，`labels[0]`应该是第一个用户的标签。
- `evidence`列表中的每个元素本身应该是一个列表。该列表的长度应为17：电子表格中不包括最后一列（标签列）的列数。
- 每个`evidence`列表中的值应与证据电子表格中列出现的顺序相同。你可以假设`shopping.csv`中列的顺序将始终保持该顺序。
- 注意，要构建最近邻分类器，我们所有的数据都需要是数字的。确保你的值具有以下类型：
  - `Administrative`、`Informational`、`ProductRelated`、`Month`、`OperatingSystems`、`Browser`、`Region`、`TrafficType`、`VisitorType`和`Weekend`都应为`int`类型
  - `Administrative_Duration`、`Informational_Duration`、`ProductRelated_Duration`、`BounceRates`、`ExitRates`、`PageValues`和`SpecialDay`都应为`float`类型
  - `Month`应为`0`（一月）到`11`（十二月）
  - `VisitorType`应为`1`（回头客）和`0`（非回头客）
  - `Weekend`应为`1`（用户在周末访问）和`0`（否则）
- 每个`labels`值应为整数`1`（如果用户确实完成了购买）或`0`（否则）
- 例如，第一个证据列表的值应为`[0, 0.0, 0, 0.0, 1, 0.0, 0.2, 0.2, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0]`，第一个标签的值应为`0`

`train_model`函数应接受一个证据列表和一个标签列表，并返回一个`scikit-learn`最近邻分类器（k=1的k最近邻分类器）在该训练数据上的拟合结果。

- 注意我们已经为你导入了`from sklearn.neighbors import KNeighborsClassifier`。你将需要在这个函数中使用`KNeighborsClassifier`。

`evaluate`函数应接受一个`labels`列表（测试集中用户的真实标签）和一个`predictions`列表（你的分类器预测的标签），并返回两个浮点值`(sensitivity, specificity)`。

- `sensitivity`应为0到1的浮点值，表示"真正例率"：实际正标签中被准确识别的比例
- `specificity`应为0到1的浮点值，表示"真负例率"：实际负标签中被准确识别的比例
- 你可以假设每个标签对于正结果（确实完成购买的用户）为`1`，对于负结果（没有完成购买的用户）为`0`
- 你可以假设真实标签列表将包含至少一个正标签和至少一个负标签

你不应修改`shopping.py`中规范要求你实现的函数之外的任何内容，尽管你可以编写额外的函数和/或导入其他Python标准库模块。你也可以导入`numpy`或`pandas`或`scikit-learn`中的任何内容，但不应使用任何其他第三方Python模块。你不应修改`shopping.csv`。

## [提示](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#hints)

- 有关如何从CSV文件加载数据的信息和示例，请参阅Python的[CSV文档](https://docs.python.org/3/library/csv.html)

## [测试](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#testing)

如果你愿意，可以在[设置`check50`](https://cs50.readthedocs.io/projects/check50/en/latest/index.html)后执行以下命令来评估代码的正确性。这不是强制性的；你可以简单地按照本规范末尾的步骤提交，这些相同的测试将在我们的服务器上运行。无论哪种方式，请务必自己编译和测试！

```bash
check50 ai50/projects/2024/x/shopping
```

执行以下命令使用`style50`评估代码风格

## [如何提交](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#how-to-submit)

1. 访问[此链接](https://submit.cs50.io/invites/d03c31aef1984c29b5e7b268c3a87b7b)，使用GitHub帐户登录，然后点击**Authorize cs50**。然后，选中表示你希望授予课程工作人员访问你提交内容的复选框，并点击**Join course**
2. [安装Git](https://git-scm.com/downloads)和（可选）[安装`submit50`](https://cs50.readthedocs.io/submit50/)
3. 如果你已安装`submit50`，执行
    
    ```bash
    submit50 ai50/projects/2024/x/shopping
    ```
    
    否则，使用Git将你的工作推送到`https://github.com/me50/USERNAME.git`，其中`USERNAME`是你的GitHub用户名，分支名为`ai50/projects/2024/x/shopping`
    
工作应在五分钟内被评分。然后你可以访问[https://cs50.me/cs50ai](https://cs50.me/cs50ai)查看当前进度！

## [致谢](https://cs50.harvard.edu/ai/2024/projects/4/shopping/#acknowledgements)

数据集由[Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018)](https://link.springer.com/article/10.1007%2Fs00521-018-3523-0)提供
