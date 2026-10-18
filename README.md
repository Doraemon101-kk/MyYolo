# 机器人比赛视觉组：深度学习与 YOLO 入门

本项目面向 **YOLO 目标检测**，包括图片重命名、train/val 划分、模型训练和视频推理。使用 Ultralytics，默认模型为 `yolo11n.pt`，也可换成 `yolov8n.pt` 等兼容的检测模型。分割、姿态、分类任务的标签格式不同，不适用本数据划分脚本。

已有的 `original_image/` 是原始照片目录。所有数据整理操作都复制到新目录，原文件不会被修改；输出目录已存在时会报错，避免覆盖。脚本只扫描指定目录的当前层，不递归扫描。

## 1. 用 uv 安装环境

在项目根目录打开 PowerShell；以下命令均从这里运行。

```powershell
# 尚未安装 uv 时运行；安装后重新打开终端
winget install --id astral-sh.uv -e

# 根据 .python-version 创建 .venv，并按 uv.lock 安装依赖
uv sync --locked

# 检查 PyTorch 和 CUDA
uv run python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

项目使用 Python 3.13，uv 在缺少该版本时会自动下载。首次安装依赖需要联网；首次使用官方预训练权重也会下载。无需手动激活虚拟环境，`uv run` 会使用本项目的 `.venv`。

若当前环境不允许写入系统 uv 缓存，可在当前 PowerShell 会话中设置：

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
uv sync --locked
```

### NVIDIA GPU（可选）

先运行上面的 CUDA 检查。若输出为 `False`，但机器有 NVIDIA 显卡，按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择与驱动匹配的 CUDA wheel。以下是 CUDA 12.8 的示例，不是所有显卡和驱动的通用配置：

```powershell
uv pip install --python .venv/Scripts/python.exe --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv run --no-sync python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

这是对当前虚拟环境的硬件特定替换，不会修改锁文件。**后续 GPU 命令将 `uv run` 换为 `uv run --no-sync`**，避免 uv 同步回锁定环境，并使用 `--device 0`。重新执行 `uv sync --locked` 可恢复锁定依赖。CPU 可完成流程教学；实际比赛数据训练建议使用 GPU。

### CPU 课堂演示：跑 5 轮或 10 轮

本项目当前 Windows 环境已验证安装的是 `torch 2.14.0+cpu`，无需再安装 CUDA。可用下面的命令确认；CPU 版的 `torch.version.cuda` 应为 `None`（仅 `cuda.is_available()` 为 False 并不足以证明安装的是 CPU 版）：

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

`tutorial_show/images` 和 `tutorial_show/labels` 中已有 33 张标注图片。先划分数据集（只需执行一次，已有输出时不要重复执行）：

```powershell
uv run python scripts/split_dataset.py --mode yolo --images tutorial_show/images --labels tutorial_show/labels --output datasets/tutorial_cpu --names class_0 class_1 --val-ratio 0.2 --seed 42
```

`class_0`、`class_1` 是按标注 ID 对应的占位名称；可以将生成的 `data.yaml` 中的名称改为真实类别名，不要改变 ID 对应关系。

```powershell
# 5 轮 CPU 演示：小模型、320 分辨率、batch=4
uv run python scripts/train_yolo.py --config configs/train_cpu_demo.yaml

# 改为 10 轮，作为一个新实验运行
uv run python scripts/train_yolo.py --config configs/train_cpu_demo.yaml --epochs 10 --name cpu_demo_10
```

演示配置使用自带的 `yolo11n.yaml` 从随机参数开始训练，无需下载预训练权重。若已有 `yolo11n.pt`，可追加 `--model yolo11n.pt --set pretrained=true` 使用预训练模型。

同名实验会自动递增目录。训练曲线在输出目录的 `results.png`，权重在 `weights/best.pt`。这套参数用于课堂演示训练流程，从随机参数训练 5～10 轮很可能还不能有效检测；33 张照片也不能代表比赛场景的泛化效果。若 CPU 仍较慢，可追加 `--imgsz 224`。

## 2. 图片重命名

将不规则名字统一为 `robot_000001.jpg`、`robot_000002.jpg`……，保留图片格式，不重新编码。按原文件名排序后编号，编号并不代表拍摄时间。

```powershell
# 先预览（不创建任何文件）
uv run python scripts/rename_images.py --images original_image --output data/renamed --dry-run

# 正式复制并重命名
uv run python scripts/rename_images.py --images original_image --output data/renamed

# 已经标注的数据：图片与同名 txt 一起重命名，另选新输出目录
uv run python scripts/rename_images.py --images raw/images --labels raw/labels --output data/renamed_labeled --prefix armor_ --start 1 --digits 6
```

输出：

```text
data/renamed_labeled/
├── images/             # 重命名后的图片
├── labels/             # 指定 --labels 时生成
└── rename_map.csv      # 原图片、新图片和标签路径的对应表
```

同一张图片与标签必须同名，例如 `abc.jpg` 对应 `abc.txt`。可将 `--images` 与 `--labels` 指向同一个混合目录，但其中的 `.txt` 必须都是标注文件，不能混入 `classes.txt` 等文件。图片存在相同主文件名（如 `a.jpg` 与 `a.png`）时会拒绝处理。

## 3. 标注与划分 train / val

### YOLO 检测标签

使用标注工具导出 YOLO 检测格式，每张图片对应一个 `.txt`，每个目标一行：

```text
class_id x_center y_center width height
```

例如图片中有一个类别 0 的目标，其中心在图片正中，宽占整张图的 20%，高占 30%：

```text
0 0.5 0.5 0.2 0.3
```

类别从 **0** 开始；坐标除以图片宽高后归一化到 `[0, 1]`。真正没有目标的背景图使用空 `.txt`；有目标却未标注的图片是漏标，不能当背景图。

### 模式 A：纯图片划分

适合先分目录、再分别标注的流程，不生成可以直接训练的 `data.yaml`。

```powershell
uv run python scripts/split_dataset.py --mode images --images data/renamed/images --output datasets/images_only --val-ratio 0.2 --seed 42
```

只生成 `images/train/`、`images/val/` 和 `split_manifest.csv`。之后须完成标注，将标签放入对应的 `labels/train/` 与 `labels/val/`，并手动创建 `data.yaml`；新手建议用下面的模式 B 自动完成。

### 模式 B：图片与标签同步划分（推荐）

假设已将重命名图片的标签导出到 `data/renamed/labels/`：

```powershell
# --names 的顺序必须与标注工具的类别 ID 一致：0=armor，1=robot
uv run python scripts/split_dataset.py --mode yolo --images data/renamed/images --labels data/renamed/labels --output datasets/robot --names armor robot --val-ratio 0.2 --seed 42 --dry-run

# 检查通过后，去掉 --dry-run
uv run python scripts/split_dataset.py --mode yolo --images data/renamed/images --labels data/renamed/labels --output datasets/robot --names armor robot --val-ratio 0.2 --seed 42
```

`armor robot` 只是示例，请替换成实际类别；只有一个类别时用 `--names armor`。脚本检查配对、类别范围、五列格式和归一化坐标。缺失标签默认报错；只有确认缺失标签的图片全是背景图时，才加 `--allow-missing-labels` 自动生成空文件。

```text
datasets/robot/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
└── split_manifest.csv
```

生成的 `data.yaml` 类似：

```yaml
path: F:/WTR/27/tutorial/DL/datasets/robot
train: images/train
val: images/val
names:
  0: armor
  1: robot
```

`path` 是自动生成的绝对路径，移动项目或换电脑后要修改。随机划分结果在输入文件列表和 `seed` 一致时可复现；至少需要两张图片，并保证 train/val 都非空。脚本不保证类别比例一致，划分后应检查两边是否覆盖所有类别。

**比赛视频抽帧的数据尤其要避免泄漏：** 同一段视频的相邻帧不要随机散落到训练集和验证集，否则验证成绩会虚高。此脚本按图片随机划分，适合相对独立的照片；视频数据应按整段视频、采集场次或场地手动分组，再分别整理对应目录。最终评估最好另留一批从未用于调参的视频或测试集。

## 4. 训练与超参数调节

深度学习训练是让模型根据带标签样本调整参数；train 用来更新参数，val 用来评估泛化能力和选模型。YOLO 一次前向计算预测目标框、类别和分数。通常从预训练权重微调，比直接从随机参数开始更适合小型比赛数据集。

```powershell
# 检查最终参数，不启动训练
uv run python scripts/train_yolo.py --config configs/train.yaml --dry-run

# CPU 上先跑 1 轮，确认数据与环境正常；结果不代表模型已训练好
uv run python scripts/train_yolo.py --config configs/train.yaml --epochs 1 --batch 2 --device cpu --close-mosaic 0 --name smoke_test

# GPU 正式训练（若手动替换了 CUDA wheel，此处用 uv run --no-sync）
uv run python scripts/train_yolo.py --config configs/train.yaml --device 0 --epochs 100 --batch 8 --name robot_v1

# 不使用配置文件也可以
uv run python scripts/train_yolo.py --model yolo11n.pt --data datasets/robot/data.yaml --epochs 100 --imgsz 640 --batch 8 --device 0

# 改用 AdamW，并传入其他 Ultralytics 参数
uv run python scripts/train_yolo.py --config configs/train.yaml --device 0 --optimizer AdamW --lr0 0.001 --set cos_lr=true --set mosaic=0.5 --name adamw_trial
```

参数优先级：**脚本默认值 < 配置文件 < 常用命令行参数 < `--set KEY=VALUE`**。`--set` 可以重复，值按 YAML 解析成数字、布尔值或列表；未知参数由 Ultralytics 报错。所有相对路径以运行命令的工作目录为准。

| 参数 | 含义与调节建议 |
| --- | --- |
| `model` | `n` 模型更轻；`s/m` 等通常需要更多显存与推理时间 |
| `epochs` | 训练轮数；结合验证指标观察，不是越多越好 |
| `imgsz` | 输入边长；小装甲板可尝试 960/1280，但显存和延迟上升 |
| `batch` | 每批图片数；显存不足先减小，单 GPU 可试 `-1` 自动估算 |
| `optimizer` | 入门可用 SGD；AdamW 可从更小的学习率开始 |
| `lr0` / `lrf` | 初始学习率 / 最终学习率系数；`optimizer=auto` 会忽略手动 `lr0` 和 `momentum` |
| `momentum` / `weight_decay` | 动量 / 权重衰减；先保持基线配置 |
| `warmup_epochs` / `cos_lr` | 学习率预热 / 余弦调度 |
| `patience` | 早停等待轮数；设为 0 关闭 |
| `mosaic` / `mixup` / `close_mosaic` | 拼图增强概率 / 混合增强概率 / 最后多少轮关闭拼图 |
| `hsv_h/s/v` | 颜色增强；红蓝阵营分类应谨慎改变色相 |
| `degrees` / `scale` / `fliplr` | 旋转 / 缩放 / 左右翻转；依据真实场景设置 |
| `box` / `cls` / `dfl` | 不同损失项的权重；初学阶段保持默认即可 |
| `amp` / `workers` / `cache` | 混合精度 / 数据加载进程数 / 数据缓存；影响速度与资源使用 |
| `seed` / `deterministic` | 尽量控制随机性，不保证跨硬件完全一致 |

先建立基线，每次只改变一个因素，记录验证指标和机器人实际推理延迟。训练 loss 降低不代表真实比赛效果一定提高；应一起看 Precision、Recall、mAP50、mAP50-95，以及误检、漏检样例。

训练结果默认在 `runs/detect/<name>/`，重名时框架自动递增目录；以脚本最终打印的目录为准：

- `weights/best.pt`：验证表现最好的权重，用于测试与部署。
- `weights/last.pt`：最近一轮权重；恢复中断训练时使用。
- `results.csv`、`results.png`：指标记录与曲线。
- `args.yaml`：实际训练参数，方便复现实验。

### 中断续训与继续微调

```powershell
# 恢复尚未完成的训练，沿用原来的轮数、优化器状态等
uv run python scripts/train_yolo.py --resume runs/detect/robot_v1/weights/last.pt --device 0

# 已完成的模型再训练 50 轮：创建新实验，不使用 --resume
uv run python scripts/train_yolo.py --config configs/train.yaml --model runs/detect/robot_v1/weights/best.pt --epochs 50 --lr0 0.001 --device 0 --name robot_finetune
```

`--resume` 只允许额外指定 `--device` 和 `--dry-run`，防止新参数意外覆盖原实验。完整训练结束后，检查点通常已移除优化器状态，应使用第二种方式微调。

## 5. 读取视频查看权重效果

```powershell
# 逐帧检测，保存带框视频；默认不弹窗
uv run python scripts/predict_video.py --weights runs/detect/robot_v1/weights/best.pt --source videos/test.mp4 --output runs/video/robot_v1.mp4 --device 0

# 只看类别 0，提高置信度阈值，弹窗显示；按 Q 或 Esc 结束
uv run python scripts/predict_video.py --weights runs/detect/robot_v1/weights/best.pt --source videos/test.mp4 --output runs/video/robot_v1_preview.mp4 --device 0 --conf 0.4 --iou 0.6 --classes 0 --show --max-frames 300
```

无 GPU 时用 `--device cpu`。输出文件必须是新的 `.mp4`，保存原视频尺寸与可读取到的帧率，不保留音轨；源帧率不可读时使用 `--fallback-fps`（默认 25）。逐帧处理，不将整段视频载入内存。日志中的处理速度含推理、绘图和编码开销，并不是输出播放帧率。

`--conf` 提高通常会减少误检，也可能增加漏检；`--iou` 控制 NMS 对重叠预测框的去重，并不是评估时的匹配 IoU。视频推理适合直观观察，**没有真实标注时不能计算准确率或 mAP**。需要量化指标时，在标注好的 val 集运行：

```powershell
uv run yolo detect val model=runs/detect/robot_v1/weights/best.pt data=datasets/robot/data.yaml imgsz=640 device=0
```

## 6. 培训顺序与检查

建议顺序：理解检测框与归一化坐标 → 重命名 → 标注十几张图片 → 配对检查与划分 → 跑通一轮训练 → 正式训练并看曲线 → 使用独立视频观察漏检与误检 → 每次改变一个参数比较。

```powershell
# 每个脚本都有中文参数帮助
uv run python scripts/rename_images.py --help
uv run python scripts/split_dataset.py --help
uv run python scripts/train_yolo.py --help
uv run python scripts/predict_video.py --help

# 自动检查文件配对、无重叠划分、参数优先级等，不下载权重
uv run python -m unittest discover -s tests -v
```

常见问题：

- **CUDA 不可用**：先检查 PyTorch CUDA 支持和显卡驱动；课堂流程可先用 CPU。
- **显存不足**：先减小 `batch`，再降低 `imgsz` 或改用更小模型。
- **Windows 加载数据出错**：使用 `--workers 0`；脚本已经添加主入口保护。
- **找不到标签 / 类别越界**：检查同名配对、类别从 0 开始以及 `--names` 顺序；不要把漏标当成背景图。
- **验证成绩很高，实际视频很差**：检查相邻帧泄漏、光照与距离差异、运动模糊、小目标样本不足。
- **输出已存在**：换一个新目录或文件名；脚本不会替你删除旧数据。

更多参数见 [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/) 和 [配置文档](https://docs.ultralytics.com/usage/cfg/)。
