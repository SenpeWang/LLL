**项目介绍**
论文《Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting》的原始 PyTorch 实现版本，该论文获得了 AAAI'21 最佳论文奖。
论文链接：[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)。

ProbSparse 注意力机制
   自注意力分数呈长尾分布，其中“活跃”的查询位于“头部”分数中，“懒惰”的查询位于“尾部”区域。ProbSparse 注意力机制旨在选择“活跃”的查询，而不是“懒惰”的查询，通过概率分布形成稀疏 Transformer。
   选择 Top-u 查询而不是 Top-u 键的原因是：自注意力层的输出是输入的重新表示，是根据点积对的分数加权组合而成的。选择顶部查询并使用完整的键可以鼓励对输入中主要成分的完整重新表示，这相当于在所有点积对中选择“头部”分数。

**环境依赖**
  - Python 3.6。
  - matplotlib == 3.1.1。
  - numpy == 1.19.4。
  - pandas == 0.25.1。
  - scikit_learn == 0.21.3。
  - torch == 1.8.0。
  - 可以通过运行命令 `pip install -r requirements.txt` 来安装依赖。

**数据**
论文中使用的 ETT 数据集可以在 [ETDataset](https://github.com/zhouhaoyi/ETDataset) 仓库中下载，所需的数据文件应放入 `data/ETT/` 文件夹。
ECL 数据和 Weather 数据可以从以下链接下载：
 [Google Drive](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR?usp=sharing)。
[BaiduPan](https://pan.baidu.com/s/1wyaGUisUICYHnfkZzWCwyA)，密码：6gan。

**可复现性**
  - 可以按照以下步骤轻松复现结果：
    1. 使用命令 `make init` 初始化 Docker 镜像。
    2. 使用命令 `make dataset` 下载数据集。
    3. 使用命令 `make run_module module="bash ETTh1.sh"` 运行 `scripts/` 文件夹中的每个脚本。
    4. 或者，一次性运行所有脚本：
       ```bash
       for file in `ls scripts`; do make run_module module="bash scripts/$script"; done
       ```。
**使用方法**
提供了谷歌 Colab 示例，帮助复现和自定义仓库，包括实验（训练和测试）、预测、可视化和自定义数据。
训练和测试模型的命令示例：
    ```bash
    # ETTh1
    python -u main_informer.py --model informer --data ETTh1 --attn prob --freq h

    # ETTh2
    python -u main_informer.py --model informer --data ETTh2 --attn prob --freq h

    # ETTm1
    python -u main_informer.py --model informer --data ETTm1 --attn prob --freq t
    ```。

**实验结果**
由于数据缩放方式的改变，实验结果已更新，Informer 的性能得到了提升。
每个数据集的实验参数都格式化在 `./scripts/` 文件夹中的 `.sh` 文件中。
提供了单变量和多变量预测结果的图表。

**常见问题**
如果遇到类似 `RuntimeError: The size of tensor a (98) must match the size of tensor b (96) at non-singleton dimension 1` 的问题，可以检查 PyTorch 版本，或者修改 `models/embed.py` 中 `TokenEmbedding` 的 `Conv1d` 代码，因为不同版本的 PyTorch 中循环填充模式在 `Conv1d` 中的实现方式有所变化。
