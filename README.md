# AEA: Listening to the Brain

Auditory-Inspired Electrode Attention for Cross-Subject EEG Drowsiness Recognition（BIBM 2025）。

本仓库提供AEA-ICNN模型及**测试被试严格隔离的嵌套LOSO训练、评估和结果导出代码**。原始Notebook保留供核查，维护代码位于`aea/`。

**复现状态（2026-10-21）：**旧README报告11名被试平均准确率约83.31%、召回率约81.97%、F1约83.59%，但没有提供逐折预测、训练记录、完整训练脚本或可核验的配置。这些属于历史报告值，并非本修订的运行结果。当前未取得原始实验数据和环境，**未复现或验证上述指标，也没有据此认定论文结果有误**。本次仅通过合成数据与代码测试验证实现流程。新默认协议与Notebook存在明确差异，不能直接将其结果与历史数字作等价比较。

![AEA architecture](Fig.2.png)

## 安装

Python >= 3.10。建议使用独立环境：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

CPU和CUDA版PyTorch应按本机环境安装；`--device cpu`为默认值。依赖范围见`requirements.txt`，本次CPU测试的直接依赖版本见`requirements-tested.txt`，不是原论文环境。可选安装项目：`python -m pip install -e .`。

## 数据接口

数据集来源由原README指向[EEG driver drowsiness dataset](https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687)。本仓库不附带数据，不自动下载，也不假定已核验其预处理。

支持MAT（SciPy可读的非v7.3格式）和NPZ，必须含以下键：

| 键 | 形状 | 约定 |
|---|---|---|
| `EEGsample` | `(N, C, T)` | 有限实数；原README约定`C=30, T=384` |
| `substate` | `(N,)`、`(N,1)`或`(1,N)` | 整数标签：`0=清醒, 1=疲劳` |
| `subindex` | 同上 | 整数被试ID，可不连续；同一人的所有记录必须使用同一ID |

不猜测轴顺序、不自动转换标签。其他布局必须由使用者明确转为`(N,C,T)`。至少需要3名被试；每个实际训练分区必须包含两个类别，否则报错。验证/测试分区允许单类，并按下述规则报告指标。旧README描述11人、128 Hz、3秒窗口；脚本按实际ID枚举折，不强行假定11人。正式实验前核对输入确有预定被试和样本。

NPZ保存示例（`x/y/subject_ids`来自已核查的数据）：

```python
import numpy as np
np.savez_compressed('dataset.npz', EEGsample=x, substate=y, subindex=subject_ids)
```

默认`--normalization channel`：每折仅用该折训练样本，在样本和时间维度上估计每电极均值/标准差；内层各训练折和外层重新训练均独立拟合。常量通道的缩放因子设为1。`--normalization none`可禁用。**传入的数据必须避免事先用全体被试拟合标准化、特征选择、伪迹阈值等参数**；已经发生的上游泄漏无法由脚本消除。分段、滤波、标签生成、试次/会话归属及电极顺序仍需核查并记录。

## 训练与超参数选择

在仓库根目录执行完整外层LOSO：

```bash
python train.py --data data/dataset.mat --output results/seed42 --seed 42 --device cpu
```

默认候选：batch size=24、lr=0.001、weight decay=0、N1=16、d=2、时间卷积核=64、电极注意力核=3；最多50轮、早停耐心26轮、Adam、余弦退火、NLLLoss。时间卷积核必须小于T，以支持训练中的单样本尾批。输出是LogSoftmax，评估取指数得到概率。所有训练样本都会使用，不丢弃尾批。

搜索候选（只在外层训练被试内部选择）：

```bash
python train.py --data data/dataset.mat --output results/search42 --grid configs/search.json --seed 42 --device cuda:0
```

`configs/search.json`是候选字典组成的JSON列表，未填写的字段使用上述默认值；可配置`batch_size, lr, weight_decay, N1, d, kernelLength, attention_kernel`。不会自动搜索20–59，也不会按外层测试准确率选出“最佳batch size”。

每个外层测试被试的流程：

1. 将该被试全部样本留作测试，其余被试作为源数据；参数选择函数只接收源数据。
2. 默认在源被试内再做完整LOSO：每次留一个源被试作验证，其余作训练；各折重新初始化网络、优化器和标准化器。
3. 每候选在每内层折按验证`balanced_accuracy`选择最佳轮次并早停；同分保留最早轮次。以各内层折最佳分数的算术平均选择候选；候选同分按JSON顺序取首个。
4. 对获选候选的各内层最佳轮次取中位数，四舍五入（0.5向上），作为外层重新训练的固定轮数。重新初始化后用全部源被试训练，调度器的周期保持为`--epochs`，与内层学习率进程一致。
5. 固定模型后仅评估一次测试被试。测试标签仅用于计算最终指标，不用于早停、选择参数或选择训练轮数；不做测试时适应。

内层最佳分数本身用于调参，不是无偏性能估计；最终性能来自外层折。即使仅一个候选，内层仍用于选择训练轮数。11人、3候选、完整内层LOSO需要330次内层训练和11次外层重新训练。

其他选项：

```bash
# 限定外层折，适合检查流程；结果会标记为非完整LOSO
python train.py --data data/dataset.mat --output results/partial --test-subjects 1 2 --inner-folds 2 --epochs 5 --patience 2

# 查看全部选项
python train.py --help
```

`--inner-folds 0`表示完整内层LOSO；正数K表示仅使用按ID排序的前K个源被试作为验证折，每折的训练仍使用其余全部源被试。这是较便宜的源内验证方案，正式报告必须说明K及ID选择规则。`--selection-metric`可选`balanced_accuracy`（默认）、`macro_f1`或`accuracy`。应预先确定候选、种子、协议，避免根据外层结果反复修改配置或挑选种子。

## 模型核查与兼容性

维护版保留Notebook的**有效注意力公式**：

```text
r = max_time(x) - min_time(x)
w = 2 * sigmoid(Conv2(ReLU(Conv1(r))))
output = LogSoftmax(Linear(GAP(BN(ReLU(DepthwiseConv(SpatialConv(x*w)))))))
```

注意力权重范围为`[0,2]`，不是归一化为和1的概率。原代码的均值、标准差、`temporal_weigh`未进入输出；已从维护代码移除。时间维已缩为1后，平均/最大池化为相同的恒等操作，两路共享卷积，所以合并为上式。**没有擅自接入时间权重或加入新统计特征。**图2绘有时间权重乘法，Notebook却没有使用，这是尚需原作者结合论文公式/原训练代码确认的差异。

默认`--batchnorm-mode running`启用训练集运行统计，测试`eval()`使用固定统计，避免测试批次组成影响预测。原Notebook的`track_running_stats=False`即使`eval()`也使用当前测试批次统计；如需研究其有效前向行为，可显式使用`--batchnorm-mode batch --normalization none`。该模式具有测试批次依赖，属于使用目标批统计的行为，**不是默认的严格归纳式协议**；必须固定并报告评估batch size和样本顺序，且不能据此称已恢复原论文实验。原Notebook未使用层的删除还会改变同种子初始化时的随机数消耗；前向等价测试使用相同有效权重，而非声称初始化轨迹完全相同。

```python
from model import AEA_ICNN
import torch
model = AEA_ICNN(classes=2)
model.eval()
with torch.inference_mode():
    log_probabilities = model(torch.randn(10, 1, 30, 384))
```

`N1`与`d`现真正控制卷积宽度和深度倍率，默认值保持原有效尺寸；电极数传入AEA，注意力使用奇数核的保形填充。原AEA构造函数中未使用的`sampleLength/classes`参数不再接受。旧Notebook权重中含未使用的`AEA.temporal_fc.*`键，且缺少运行统计，不能直接作为默认维护版checkpoint加载。详见[核查记录](docs/AUDIT.md)。

## 指标、导出和重新评估

`results/seed42/`包含：

```text
manifest.json            # 参数、数据SHA256、依赖版本、源码提交/脏状态、完成状态
fold_metrics.csv/json    # 每测试被试指标
summary.json             # 被试等权均值/总体标准差 + 全样本汇总指标
predictions.csv          # 每样本原始0起始索引、被试ID、真实/预测标签、两类概率
subject_<ID>/
  selection.json         # 源/测试索引、候选、内层被试划分、种子、历史、获选轮数
  refit_history.json     # 用全部源被试重新训练的逐轮损失和学习率
  model.pt               # 最终权重、模型配置、标准化参数、测试被试、数据hash
  metrics.json
  predictions.csv
```

Accuracy、balanced accuracy、macro precision/recall/F1、疲劳类precision/recall/F1、ROC-AUC和固定顺序`[0,1]`的混淆矩阵均导出；指标以0–1计。Macro指标固定对两类取平均，未定义precision/recall/F1记0；balanced accuracy仅平均真实出现类别的recall，单类折需谨慎解释。单类折ROC-AUC写为JSON `null`；跨折AUC只统计可定义折并报告数量。

`subject_macro`对被试等权，`std_population`采用`ddof=0`；`pooled_samples`合并所有外层测试预测，按样本加权。两者含义不同。每折混淆矩阵在JSON中。仅跑部分外层折时`complete_loso=false`，不会标为完整LOSO。

输出目录必须为空，避免覆盖旧实验。失败中断时`manifest.status`保持`running`，不应当作完成结果；目前不自动续跑。`model.pt`是推理checkpoint，不包含恢复优化器训练的状态。

重新评估某折，无任何训练或参数选择：

```bash
python evaluate.py --checkpoint results/seed42/subject_1/model.pt --data data/dataset.mat --output results/recheck_subject1 --device cpu
```

仅评估checkpoint记录的外层测试被试，并校验数据文件SHA256一致。字节不同（包括重打包）的文件会被拒绝。保持相同设备/软件/线程设置有利于数值一致。`torch.load(weights_only=True)`只加载维护版checkpoint；不要加载不可信来源的模型。

## 随机性与测试

默认种子42，分别固定Python、NumPy、PyTorch和DataLoader随机性；折种子由主种子和排序后的被试位置派生，不使用标签。使用单进程DataLoader、固定CPU线程数（默认1），启用确定性算法、关闭cuDNN benchmark。CUDA工作区配置在模型创建前设置；不支持确定性的操作会报错，`--allow-nondeterministic`可显式放宽。相同种子不保证跨硬件、PyTorch/CUDA版本完全一致，原README中“不同服务器差异较大”不能替代可核验的配置和预测记录。

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试包括原Notebook有效前向/梯度等价性、默认BN评估批次独立性、被试隔离、训练集标准化、MAT/NPZ输入检查、固定指标定义、合成数据完整嵌套LOSO、checkpoint重新评估、固定种子重复性，以及更改测试标签和信号后选择记录/最终权重不变。合成数据测试仅验证工程流程，不能验证论文准确率或临床有效性。

## 文件结构

```text
aea/{model,data,experiment,metrics}.py   # 模型、数据、训练流程、指标
model.py                               # 兼容模型导入
train.py / evaluate.py                 # 命令行训练 / 重新评估
configs/search.json                    # 候选参数示例
requirements*.txt / pyproject.toml     # 依赖与包配置
tests/ / .github/workflows/tests.yml   # CPU测试与CI定义
docs/AUDIT.md                          # 模型与复现缺口核查
AEA.ipynb                              # 历史模型代码（不含完整实验）
Fig.2.png / LICENSE
```

## 引用与许可证

以下作者信息沿用旧README并规范BibTeX格式；页码和DOI尚未核验，不补造。

```bibtex
@inproceedings{li2025listening,
  author = {Li, Songquan and Yuan, Jun and Li, Mengyao and Wang, Hanming and Zhu, Rongbo},
  title = {Listening to the Brain: An Auditory-Inspired Electrode Attention for Cross-Subject EEG Drowsiness Recognition},
  booktitle = {2025 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year = {2025}
}
```

MIT License。仅供研究。
