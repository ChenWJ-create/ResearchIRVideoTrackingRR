# TFW YOLOv5n-Face 热成像人脸检测基线

本目录已接入 TFW 官方发布的 `YOLOv5n-Face` 热成像权重。你不需要再手动下载。

注意：该模型完成的是**热成像人脸检测 + 5 个关键点定位**，还不能判断“这个人是谁”。身份识别需要在检测和对齐之后，再接一个特征模型以及人员特征库。

## 已准备的内容

- 权重：`weights/yolov5n-face-tfw.pt`
- YOLOv5-Face 源码：`vendor/yolov5-face`
- 推理入口：`run_inference.py`
- BIN 逐帧读取模块：`thermal_bin.py`
- 待测图片目录：`input_images`
- 本机验证环境：`.venv`（Python 3.12、PyTorch 2.2.2 CPU）

源码和权重版本信息见 [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md)。`vendor` 目录只作为固定依赖，不要直接修改；本项目的适配均放在当前目录。

## 直接读取 16UC1 BIN 视频

不需要先把整个 BIN 转换成图片。`thermal_bin.py` 按以下结构读取：

- 文件头：4 个 little-endian `int32`；前两个字段依次为 `height`、`width`。
- 帧数据：连续存放的 `height × width` 个 little-endian `uint16`。
- 温度换算：`raw / 64.0 - 273.15`。

随机读取一帧：

```python
from thermal_bin import read_bin_header, read_bin_frame

bin_path = r"F:\你的目录\nir_temp_frame_16uc1.bin"

info = read_bin_header(bin_path)
print(info.height, info.width, info.total_frames)

raw_frame = read_bin_frame(bin_path, frame_index=0, output="raw")
temp_frame = read_bin_frame(bin_path, frame_index=0, output="temperature")

print(raw_frame.shape, raw_frame.dtype)   # (height, width), uint16
print(temp_frame.min(), temp_frame.max()) # 摄氏温度
```

也提供了与你要求一致的函数别名：

```python
from thermal_bin import readbinframe

raw_frame = readbinframe(bin_path, frame_index=0)
```

循环读取视频时不要反复调用单帧函数，使用只打开一次文件的迭代器：

```python
from thermal_bin import iter_bin_frames

for frame_index, raw_frame in iter_bin_frames(
    bin_path,
    start_frame=0,
    frame_step=30,
    max_frames=100,
    output="raw",
):
    # 在这里直接调用检测、显示或其他处理
    print(frame_index, raw_frame.shape)
```

你原代码中的：

```python
ir_frame.reshape((width, height), order="F").T
```

与这里使用的 `ir_frame.reshape((height, width))` 数值和方向完全相同。读取器一次只保留当前帧，不会把整个大文件加载进内存，也不会向 BIN 原文件写入内容。

## 第一步：先验证少量 BIN 帧

建议先只运行一个指定帧：

```powershell
cd D:\major\IRsegementation\IRrecognize\python_code\thermal_face_baseline

.\.venv\Scripts\python.exe .\run_inference.py `
  --input "F:\你的目录\nir_temp_frame_16uc1.bin" `
  --output .\output\bin_single_frame `
  --start-frame 0 `
  --end-frame 1 `
  --max-frames 1
```

确认画面方向、灰度和人脸框正常后，可以生成实时检测预览和完整 AVI。`--frame-step 1` 表示每个原始帧都进行检测并写入 AVI；实时窗口通过 `--show-every 5` 每 5 帧刷新一次，不会造成 AVI 丢帧：

```powershell
.\.venv\Scripts\python.exe .\run_inference.py `
  --input "F:\你的目录\nir_temp_frame_16uc1.bin" `
  --output .\output\bin_video_preview `
  --start-frame 0 `
  --frame-step 1 `
  --max-frames 600 `
  --source-fps 30 `
  --show `
  --show-every 5 `
  --save-video
```

实时窗口按 `Q` 或 `Esc` 可以提前停止，程序仍会正常关闭 AVI 并写出 CSV/JSON。该功能是逐帧检测预览，不包含跨帧人员 ID 跟踪。

参数含义：

- `--start-frame`：起始帧号，从 0 开始。
- `--end-frame`：终止帧号，不包含该帧。
- `--frame-step`：原始帧处理间隔；完整 AVI 必须使用默认值 `1`，会处理每个原始帧。大于 `1` 时 AVI 也会相应抽帧。
- `--max-frames`：本次最多处理帧数，默认 100；设置为 `0` 才会不限制，不建议第一次就处理完整视频。
- `--show`：打开实时检测窗口；不添加该参数时可以在后台处理。
- `--show-every`：每多少个已处理帧刷新一次窗口，默认 `5`；它只影响实时窗口，不影响检测、CSV、漏检图片或 AVI。
- `--save-video`：生成 `detection_preview.avi`，使用兼容性较好的 MJPG 编码。
- `--source-fps`：原始视频帧率，默认 `30`；当 `frame-step=1` 时 AVI 保持该帧率和完整时间长度。
- `--reliable-conf-thres`：可独立建立或重新建立主脸的可靠阈值，默认 `0.40`。
- `--max-interpolation-seconds`：低分主脸候选可沿用上一位置进行关联的最长时间，默认 `0.20` 秒。
- `--save-images auto`：BIN 输入默认只把没有关联到主脸的帧保存到 `missed/`，不再保存每一张成功帧。

输出 AVI 会包含帧号、候选数、检测框、关键点和 `PRIMARY` 主脸标记。低于 `0.40` 的候选为橙色、`0.40～0.59` 为黄色、`0.60` 及以上为绿色；`missed/` 中只保留未关联到主脸的帧。`detections.csv` 会记录所有已处理帧的 `frame_index`、原始温度最小值、最大值和平均值。

如果需要保存所有标注帧用于逐张核查，可以添加 `--save-images all`；完全不保存静态图则使用 `--save-images none`。

## 图片文件验证方式

1. 把 30～100 张有代表性的原始热像复制到 `input_images`。支持 JPG、PNG、BMP、TIF、TIFF，以及 8 位/16 位灰度图。不要转换或覆盖原图。
2. 在 PowerShell 中执行：

```powershell
cd D:\major\IRsegementation\IRrecognize\python_code\thermal_face_baseline
.\.venv\Scripts\python.exe .\run_inference.py --input .\input_images --output .\output\first_test
```

默认参数沿用 TFW 的主要设置：输入尺寸 `800`，置信度阈值 `0.60`，NMS IoU 阈值 `0.50`。如果漏检明显，可以额外测试一次较低阈值：

```powershell
.\.venv\Scripts\python.exe .\run_inference.py --input .\input_images --output .\output\conf_040 --conf-thres 0.40
```

Notebook 的运动人脸实验使用候选阈值 `0.10` 和可靠阈值 `0.40`。`0.10～0.39` 的候选只有在位置和面积与近期 `PRIMARY` 主脸连续时才会被选中，不能单独初始化主脸轨迹。检测始终使用原始热像生成的 8 位检测视图，不做 CLAHE 或边缘增强；后续 ROI 温度数据必须从原始 `uint16` 帧换算后裁剪，不能从检测视图或标注图中取值。

若输出目录已经包含结果，程序会自动新建 `_2`、`_3` 后缀目录，不会清空旧结果。

## 需要检查什么

运行结果位于指定输出目录：

- `annotated/`：人脸框和 5 个关键点叠加图。首先人工检查框是否覆盖完整人脸，以及眼、鼻、嘴关键点是否落在正确位置。
- `detections.csv`：每个框的置信度、坐标、关键点、输入位深和推理耗时。
- `summary.json`：本次模型、阈值、图片数、检出图片数和平均耗时。

建议样本至少覆盖：近/中/远距离、正脸/侧脸、眼镜、口罩、不同环境温度、不同相机或伪彩设置。重点关注以下变量：

- `face_count`：每张图检出人脸数；如果每张输入本来都有一张脸，`0` 就是漏检。
- `confidence`：正确框与错误框的分数分布，用来确定后续阈值。
- `right_eye_*` 到 `left_mouth_*`：5 个关键点是否稳定；命名采用被拍摄者自身左右方向。
- `source_dtype`、`normalization`、`scale_low/high`：确认 16 位热像采用了合理的灰度拉伸。
- `inference_ms`：当前环境是 CPU，只能作为功能验证，不能代表最终设备速度。

把 `output/first_test/annotated` 中有代表性的正确、漏检、误检图片保留下来，并把 `detections.csv` 一起给我。下一步再根据结果决定是调预处理和阈值，还是制作少量标注进行微调，不建议现在直接开始身份识别训练。

## 16 位热像说明

原始 16 位图片不会被改写。程序只在内存中创建一个 8 位检测视图，默认截取 1%～99% 灰度百分位并拉伸。可以对比以下模式：

```powershell
# 使用实际最小值和最大值拉伸
.\.venv\Scripts\python.exe .\run_inference.py --input .\input_images --output .\output\minmax --normalization minmax

# 按数据类型完整范围映射，通常只适合已经定标的数据
.\.venv\Scripts\python.exe .\run_inference.py --input .\input_images --output .\output\full_range --normalization none
```

## 重建 Python 环境（仅在 `.venv` 不可用时）

当前 `.venv` 已经完成最低限度验证，不必重复安装。如果迁移到另一台机器，可以用 Python 3.10～3.12 新建环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

`requirements.txt` 是本机验证过的 CPU 组合。若最终设备使用 NVIDIA GPU、Jetson、RKNN、OpenVINO 或其他 NPU，不要直接照搬其中的 PyTorch 包；先按设备安装对应框架，再进行 ONNX/设备格式导出和速度验证。
