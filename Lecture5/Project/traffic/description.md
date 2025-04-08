## [交通标志识别](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#traffic)

编写一个 AI 来识别照片中出现的交通标志。

```yaml
$ python traffic.py gtsrb
Epoch 1/10
500/500 [==============================] - 5s 9ms/step - loss: 3.7139 - accuracy: 0.1545
Epoch 2/10
500/500 [==============================] - 6s 11ms/step - loss: 2.0086 - accuracy: 0.4082
Epoch 3/10
500/500 [==============================] - 6s 12ms/step - loss: 1.3055 - accuracy: 0.5917
Epoch 4/10
500/500 [==============================] - 5s 11ms/step - loss: 0.9181 - accuracy: 0.7171
Epoch 5/10
500/500 [==============================] - 7s 13ms/step - loss: 0.6560 - accuracy: 0.7974
Epoch 6/10
500/500 [==============================] - 9s 18ms/step - loss: 0.5078 - accuracy: 0.8470
Epoch 7/10
500/500 [==============================] - 9s 18ms/step - loss: 0.4216 - accuracy: 0.8754
Epoch 8/10
500/500 [==============================] - 10s 20ms/step - loss: 0.3526 - accuracy: 0.8946
Epoch 9/10
500/500 [==============================] - 10s 21ms/step - loss: 0.3016 - accuracy: 0.9086
Epoch 10/10
500/500 [==============================] - 10s 20ms/step - loss: 0.2497 - accuracy: 0.9256
333/333 - 5s - loss: 0.1616 - accuracy: 0.9535
```

## [何时完成](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#when-to-do-it)

截止于 [2026 年 1 月 1 日星期四上午 7:59 GMT+8](https://time.cs50.io/20251231T235900Z)

## [如何获得帮助](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#how-to-get-help)

1.  通过 [Ed](https://cs50.edx.org/ed) 提问！
2.  通过 CS50 的任何 [社区](https://cs50.harvard.edu/ai/2024/communities/) 提问！

## [背景](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#background)

随着自动驾驶汽车研究的不断深入，其中的一个关键挑战是 [计算机视觉](https://en.wikipedia.org/wiki/Computer_vision)，它使这些汽车能够从数字图像中理解周围环境。特别是，这涉及到识别和区分道路标志的能力——停车标志、限速标志、让行标志等等。

在这个项目中，你将使用 [TensorFlow](https://www.tensorflow.org/) 构建一个神经网络，根据道路标志的图像对其进行分类。为此，你需要一个标记的数据集：一组已经按照道路标志类型进行分类的图像。

已经存在几个这样的数据集，但对于这个项目，我们将使用 [德国交通标志识别基准](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) (GTSRB) 数据集，它包含数千张 43 种不同类型道路标志的图像。

## [开始](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#getting-started)

-   从 [https://cdn.cs50.net/ai/2023/x/projects/5/traffic.zip](https://cdn.cs50.net/ai/2023/x/projects/5/traffic.zip) 下载分发代码并解压。
-   下载此项目的 [数据集](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb.zip) 并解压。 将生成的 `gtsrb` 目录移动到你的 `traffic` 目录中。
-   在 `traffic` 目录中，运行 `pip3 install -r requirements.txt` 来安装此项目的依赖项：`opencv-python` 用于图像处理，`scikit-learn` 用于 ML 相关功能，`tensorflow` 用于神经网络。

## [理解](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#understanding)

首先，打开 `gtsrb` 目录来查看数据集。 你会注意到该数据集中有 43 个子目录，编号为 `0` 到 `42`。 每个编号的子目录代表一个不同的类别（一种不同类型的道路标志）。 在每个交通标志的目录中，都包含该类型交通标志的图像集合。

接下来，查看 `traffic.py`。 在 `main` 函数中，我们将包含数据的目录（以及可选的用于保存训练模型的文件名）作为命令行参数。 然后，数据和相应的标签从数据目录加载（通过 `load_data` 函数）并分成训练集和测试集。 之后，调用 `get_model` 函数以获得一个编译好的神经网络，然后将其拟合到训练数据上。 然后在测试数据上评估该模型。 最后，如果提供了模型文件名，则将训练后的模型保存到磁盘。

`load_data` 和 `get_model` 函数留给你来实现。

## [规范](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#specification)

完成 `traffic.py` 中 `load_data` 和 `get_model` 的实现。

-   `load_data` 函数应接受一个参数 `data_dir`，表示数据存储目录的路径，并返回数据集中每个图像的图像数组和标签。
    -   你可以假设 `data_dir` 将包含一个以每个类别命名的目录，编号为 `0` 到 `NUM_CATEGORIES - 1`。 每个类别目录中都会有一些图像文件。
    -   使用 OpenCV-Python 模块 (`cv2`) 将每个图像读取为 `numpy.ndarray`（一个 `numpy` 多维数组）。 要将这些图像传递到神经网络中，图像需要大小相同，因此请务必将每个图像的大小调整为宽度 `IMG_WIDTH` 和高度 `IMG_HEIGHT`。
    -   该函数应返回一个元组 `(images, labels)`。 `images` 应该是数据集中所有图像的列表，其中每个图像都表示为具有适当大小的 `numpy.ndarray`。 `labels` 应该是一个整数列表，表示 `images` 列表中每个对应图像的类别编号。
    -   你的函数应该是平台独立的：也就是说，它应该可以在任何操作系统上工作。 请注意，在 macOS 上，`/` 字符用于分隔路径组成部分，而在 Windows 上，则使用 `\` 字符。 根据需要使用 [`os.sep`](https://docs.python.org/3/library/os.html) 和 [`os.path.join`](https://docs.python.org/3/library/os.path.html#os.path.join)，而不是使用你的平台特定的分隔符字符。
-   `get_model` 函数应返回一个编译好的神经网络模型。
    -   你可以假设神经网络的输入将是 `(IMG_WIDTH, IMG_HEIGHT, 3)` 的形状（即，一个数组，表示宽度为 `IMG_WIDTH`，高度为 `IMG_HEIGHT` 的图像，并且每个像素有 `3` 个值，分别代表红色、绿色和蓝色）。
    -   神经网络的输出层应具有 `NUM_CATEGORIES` 个单元，每个单元对应一个交通标志类别。
    -   层数和层类型由你决定。 你可能希望尝试：
        -   不同数量的卷积层和池化层
        -   卷积层使用不同数量和大小的过滤器
        -   池化层使用不同的池化大小
        -   不同数量和大小的隐藏层
        -   dropout
-   在一个名为 _README.md_ 的单独文件中，记录（至少一段或两段）你的实验过程。 你尝试了什么？ 什么效果好？ 什么效果不好？ 你注意到了什么？

最终，这个项目的大部分内容是关于探索文档并研究 `cv2` 和 `tensorflow` 中的不同选项，并查看尝试它们时得到的结果！

除了规范要求你实现的函数之外，你不应修改 `traffic.py` 中的任何其他内容，尽管你可以编写其他函数和/或导入其他 Python 标准库模块。 如果你熟悉 `numpy` 或 `pandas`，也可以导入它们，但不应使用任何其他第三方 Python 模块。 你可以修改文件顶部定义的全局变量，以便使用其他值测试你的程序。

## [提示](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#hints)

-   查看官方 [Tensorflow Keras 概述](https://www.tensorflow.org/guide/keras/overview)，了解构建神经网络层的语法的一些指南。 你可能会发现讲座的源代码也很有用。
-   [OpenCV-Python](https://docs.opencv.org/4.5.2/d2/d96/tutorial_py_table_of_contents_imgproc.html) 文档可能有助于将图像读取为数组，然后调整其大小。
-   调整图像 `img` 的大小后，你可以通过打印 `img.shape` 的值来验证其维度。 如果你正确地调整了图像的大小，则其形状应为 `(30, 30, 3)`（假设 `IMG_WIDTH` 和 `IMG_HEIGHT` 均为 `30`）。
-   如果你想使用较小的数据集进行练习，则可以下载一个 [修改后的数据集](https://cdn.cs50.net/ai/2023/x/projects/5/gtsrb-small.zip)，其中仅包含 3 种不同类型的道路标志，而不是 43 种。

## [测试](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#testing)

如果你愿意，你可以执行以下命令（在你的系统上 [设置 `check50`](https://cs50.readthedocs.io/projects/check50/en/latest/index.html) 之后）来评估你的代码的正确性。 这不是强制性的； 你可以按照本规范末尾的步骤进行提交，这些相同的测试将在我们的服务器上运行。 无论哪种方式，请务必自己编译和测试它！

```bash
check50 ai50/projects/2024/x/traffic
```

执行以下命令以使用 `style50` 评估你的代码的样式。

## [如何提交](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#how-to-submit)

1.  访问 [此链接](https://submit.cs50.io/invites/d03c31aef1984c29b5e7b268c3a87b7b)，使用你的 GitHub 帐户登录，然后单击 **授权 cs50**。 然后，选中指示你希望授予课程工作人员访问你的提交内容的复选框，然后单击 **加入课程**。
2.  [安装 Git](https://git-scm.com/downloads)，并可以选择 [安装 `submit50`](https://cs50.readthedocs.io/submit50/)。
3.  如果你已安装 `submit50`，请执行
    
    ```bash
    submit50 ai50/projects/2024/x/traffic
    ```
    
    否则，使用 Git 将你的工作推送到 `https://github.com/me50/USERNAME.git`，其中 `USERNAME` 是你的 GitHub 用户名，分支名称为 `ai50/projects/2024/x/traffic`。
    

工作将在五分钟内评分。 然后，你可以转到 [https://cs50.me/cs50ai](https://cs50.me/cs50ai) 查看你当前的进度！

## [致谢](https://cs50.harvard.edu/ai/2024/projects/5/traffic/#acknowledgements)

数据由 [J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=dataset#Acknowledgements) 提供。
