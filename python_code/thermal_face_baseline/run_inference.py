"""TFW YOLOv5n-Face thermal face detection baseline.

This script intentionally keeps the vendored YOLOv5-Face source unchanged. It adds
thermal-image normalization, Unicode path I/O, result visualization, and CSV/JSON
reports around the official TFW checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
import types
import warnings
from pathlib import Path
from typing import Any, Iterator

BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "vendor" / "yolov5-face"
DEFAULT_WEIGHTS = BASE_DIR / "weights" / "yolov5n-face-tfw.pt"
DEFAULT_INPUT = BASE_DIR / "input_images"
DEFAULT_OUTPUT = BASE_DIR / "output" / "run"

if not VENDOR_DIR.is_dir():
    raise FileNotFoundError(
        f"未找到 YOLOv5-Face 源码：{VENDOR_DIR}\n"
        "请先按 README.md 的下载说明准备 vendor/yolov5-face。"
    )
sys.path.insert(0, str(VENDOR_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression_face, scale_coords  # noqa: E402
from thermal_bin import (  # noqa: E402
    ThermalBinInfo,
    iter_bin_frames,
    raw_to_celsius,
    read_bin_header,
    select_frame_indices,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LANDMARK_NAMES = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")
LANDMARK_COLORS = (
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
)
HIGH_CONFIDENCE_THRESHOLD = 0.60
MEDIUM_CONFIDENCE_THRESHOLD = 0.40
HIGH_CONFIDENCE_COLOR = (0, 200, 0)
MEDIUM_CONFIDENCE_COLOR = (0, 255, 255)
LOW_CONFIDENCE_COLOR = (0, 165, 255)
PRIMARY_FACE_COLOR = (255, 255, 0)
MAX_CENTER_DISTANCE_RATIO = 0.75
MIN_AREA_RATIO = 0.50
MAX_AREA_RATIO = 2.00
VENDOR_COMMIT = "152c688d551aefb973b7b589fb0691c93dab3564"
MODEL_SOURCE = "https://github.com/IS2AI/TFW"
TRUSTED_TFW_SHA256 = "5596275882839ab6e21177cc15572dd56c71c3fcafd2b0ea3b3ffa45d2c2677a"

warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release.*",
    category=UserWarning,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 TFW YOLOv5n-Face 检测热成像人脸和 5 个关键点。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="单张图片、图片文件夹，或含 4 个 int32 文件头的 16UC1 BIN 视频。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="TFW YOLOv5n-Face .pt 权重路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出目录；若目录非空会自动创建 run_2、run_3 等新目录。",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=800,
        help="网络输入长边尺寸，TFW 官方训练配置为 800。",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.10,
        help="候选人脸置信度下限；低于可靠阈值的候选还必须通过时序位置和面积门控。",
    )
    parser.add_argument(
        "--reliable-conf-thres",
        type=float,
        default=0.40,
        help="可独立建立或重新建立主脸轨迹的可靠置信度阈值。",
    )
    parser.add_argument("--iou-thres", type=float, default=0.50, help="NMS IoU 阈值。")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto、cpu、cuda 或 cuda:0；auto 会优先使用 CUDA。",
    )
    parser.add_argument(
        "--normalization",
        choices=("percentile", "minmax", "none"),
        default="percentile",
        help="非 8 位热像转 8 位检测视图的归一化方式。",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="percentile 模式的低百分位。",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.0,
        help="percentile 模式的高百分位。",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="BIN 起始帧号，从 0 开始。")
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="BIN 终止帧号，不包含该帧；默认到文件末尾。",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="BIN 处理间隔；默认 1，即每个原始帧都检测并写入 AVI。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="BIN 最多处理多少帧；默认 100，设为 0 表示不限制。",
    )
    parser.add_argument(
        "--save-images",
        choices=("auto", "all", "missed", "none"),
        default="auto",
        help="静态图片保存策略；auto 对 BIN 只保存漏检帧，对图片输入保存全部。",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="把所有已处理帧及检测结果保存为 AVI。",
    )
    parser.add_argument(
        "--video-name",
        default="detection_preview.avi",
        help="AVI 文件名，使用 MJPG 编码。",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="原始 BIN 帧率，用于计算抽帧后 AVI 的播放帧率。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="运行时显示检测窗口；按 Q 或 Esc 可提前停止并正常保存结果。",
    )
    parser.add_argument(
        "--show-every",
        type=int,
        default=5,
        help="每多少个已处理帧刷新一次窗口；默认每 5 帧显示一次，不影响 AVI。",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="每多少个已处理帧打印一次进度；漏检帧始终打印。",
    )
    parser.add_argument(
        "--max-interpolation-seconds",
        type=float,
        default=0.20,
        help="主脸关联允许沿用上一检测位置的最长时间；后续也用于短缺失插值。",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.conf_thres <= 1.0:
        raise ValueError("--conf-thres 必须在 0 到 1 之间。")
    if not 0.0 <= args.reliable_conf_thres <= 1.0:
        raise ValueError("--reliable-conf-thres 必须在 0 到 1 之间。")
    if not 0.0 <= args.iou_thres <= 1.0:
        raise ValueError("--iou-thres 必须在 0 到 1 之间。")
    if not 0.0 <= args.lower_percentile < args.upper_percentile <= 100.0:
        raise ValueError("百分位参数必须满足 0 <= lower < upper <= 100。")
    if args.img_size <= 0:
        raise ValueError("--img-size 必须大于 0。")
    if args.start_frame < 0:
        raise ValueError("--start-frame 不能小于 0。")
    if args.end_frame is not None and args.end_frame < 0:
        raise ValueError("--end-frame 不能小于 0。")
    if args.frame_step <= 0:
        raise ValueError("--frame-step 必须大于 0。")
    if args.max_frames < 0:
        raise ValueError("--max-frames 不能小于 0。")
    if args.source_fps <= 0:
        raise ValueError("--source-fps 必须大于 0。")
    if args.show_every <= 0:
        raise ValueError("--show-every 必须大于 0。")
    if args.log_every <= 0:
        raise ValueError("--log-every 必须大于 0。")
    if args.max_interpolation_seconds < 0:
        raise ValueError("--max-interpolation-seconds 不能小于 0。")
    video_name = Path(args.video_name)
    if video_name.suffix.lower() != ".avi":
        raise ValueError("--video-name 必须使用 .avi 扩展名。")
    if video_name.name != args.video_name:
        raise ValueError("--video-name 只能是文件名，不能包含目录。")


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图片格式：{input_path.suffix}")
        return [input_path]
    if input_path.is_dir():
        images = sorted(
            path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            return images
        raise FileNotFoundError(f"输入目录中没有支持的图片：{input_path}")
    raise FileNotFoundError(f"输入路径不存在：{input_path}")


def iter_inputs(
    input_path: Path,
    image_paths: list[Path],
    args: argparse.Namespace,
) -> Iterator[tuple[Path, int | None, np.ndarray]]:
    if input_path.is_file() and input_path.suffix.lower() == ".bin":
        max_frames = args.max_frames if args.max_frames > 0 else None
        for frame_index, raw_frame in iter_bin_frames(
            input_path,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_step=args.frame_step,
            max_frames=max_frames,
            output="raw",
        ):
            yield input_path, frame_index, raw_frame
        return

    for image_path in image_paths:
        yield image_path, None, read_image_unicode(image_path)


def choose_output_dir(requested: Path) -> Path:
    candidate = requested
    index = 2
    while candidate.exists() and any(candidate.iterdir()):
        candidate = requested.with_name(f"{requested.name}_{index}")
        index += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def read_image_unicode(path: Path) -> np.ndarray:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV 无法读取图片：{path}")
    return image


def write_image_unicode(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"OpenCV 无法编码输出图片：{path}")
    encoded.tofile(str(path))


def create_avi_writer(path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"无法创建 AVI：{path}")
    return writer


def draw_frame_status(
    image: np.ndarray,
    frame_index: int | None,
    face_count: int,
    primary_confidence: float | None,
) -> None:
    frame_text = f"frame {frame_index}" if frame_index is not None else "image"
    primary_text = "none" if primary_confidence is None else f"{primary_confidence:.2f}"
    status = f"{frame_text} | faces {face_count} | primary {primary_text}"
    color = (
        confidence_color(primary_confidence)
        if primary_confidence is not None
        else (0, 0, 255)
    )
    cv2.rectangle(image, (0, 0), (440, 32), (0, 0, 0), -1)
    cv2.putText(image, status, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)


def scale_to_uint8(
    image: np.ndarray,
    method: str,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, float, float]:
    values = image[np.isfinite(image)] if np.issubdtype(image.dtype, np.floating) else image.reshape(-1)
    if values.size == 0:
        raise ValueError("图片不包含有限数值。")

    if method == "percentile":
        low, high = np.percentile(values, (lower_percentile, upper_percentile))
    elif method == "minmax":
        low, high = float(values.min()), float(values.max())
    elif np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        low, high = float(limits.min), float(limits.max)
    else:
        low = 0.0
        high = 1.0 if float(values.max()) <= 1.0 else 255.0

    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(image.shape, dtype=np.uint8), low, high

    scaled = (image.astype(np.float32) - low) * (255.0 / (high - low))
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8), low, high


def make_detection_view(
    raw: np.ndarray,
    method: str,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "source_dtype": str(raw.dtype),
        "source_shape": list(raw.shape),
        "normalization": "uint8_passthrough" if raw.dtype == np.uint8 else method,
        "scale_low": 0.0,
        "scale_high": 255.0,
    }

    if raw.dtype == np.uint8:
        view = raw
    else:
        view, low, high = scale_to_uint8(raw, method, lower_percentile, upper_percentile)
        metadata["scale_low"] = low
        metadata["scale_high"] = high

    if view.ndim == 2:
        bgr = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    elif view.ndim == 3 and view.shape[2] == 1:
        bgr = cv2.cvtColor(view[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif view.ndim == 3 and view.shape[2] == 3:
        bgr = view
    elif view.ndim == 3 and view.shape[2] == 4:
        bgr = cv2.cvtColor(view, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"不支持的图片维度：{raw.shape}")
    return np.ascontiguousarray(bgr), metadata


def select_device(device_arg: str) -> torch.device:
    requested = device_arg.strip().lower()
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 检测不到可用 CUDA。请改用 --device cpu。")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def preprocess(image_bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    resized = letterbox(image_bgr, new_shape=image_size)[0]
    chw_rgb = resized[:, :, ::-1].transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(chw_rgb)).to(device)
    tensor = tensor.float() / 255.0
    return tensor.unsqueeze(0)


def scale_landmarks(
    input_shape: tuple[int, int],
    landmarks: torch.Tensor,
    original_shape: tuple[int, int, int],
) -> torch.Tensor:
    gain = min(input_shape[0] / original_shape[0], input_shape[1] / original_shape[1])
    pad_x = (input_shape[1] - original_shape[1] * gain) / 2
    pad_y = (input_shape[0] - original_shape[0] * gain) / 2
    landmarks[:, 0::2] -= pad_x
    landmarks[:, 1::2] -= pad_y
    landmarks[:, :10] /= gain
    landmarks[:, 0::2].clamp_(0, original_shape[1])
    landmarks[:, 1::2].clamp_(0, original_shape[0])
    return landmarks


def confidence_color(confidence: float) -> tuple[int, int, int]:
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return HIGH_CONFIDENCE_COLOR
    if confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        return MEDIUM_CONFIDENCE_COLOR
    return LOW_CONFIDENCE_COLOR


def draw_detection(image: np.ndarray, detection: np.ndarray) -> None:
    x1, y1, x2, y2, confidence = detection[:5]
    box_color = confidence_color(float(confidence))
    cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), box_color, 2)
    label = f"face {confidence:.2f}"
    text_y = max(round(y1) - 8, 18)
    cv2.putText(image, label, (round(x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
    for index, color in enumerate(LANDMARK_COLORS):
        point_x, point_y = detection[5 + 2 * index : 7 + 2 * index]
        cv2.circle(image, (round(point_x), round(point_y)), 3, color, -1)


def bbox_area(detection: np.ndarray) -> float:
    width = max(float(detection[2] - detection[0]), 0.0)
    height = max(float(detection[3] - detection[1]), 0.0)
    return width * height


def is_position_consistent(candidate: np.ndarray, previous: np.ndarray) -> bool:
    previous_width = max(float(previous[2] - previous[0]), 1.0)
    previous_height = max(float(previous[3] - previous[1]), 1.0)
    previous_diagonal = float(np.hypot(previous_width, previous_height))
    previous_center = np.array(
        ((previous[0] + previous[2]) / 2.0, (previous[1] + previous[3]) / 2.0)
    )
    candidate_center = np.array(
        ((candidate[0] + candidate[2]) / 2.0, (candidate[1] + candidate[3]) / 2.0)
    )
    center_distance = float(np.linalg.norm(candidate_center - previous_center))
    previous_area = max(bbox_area(previous), 1.0)
    area_ratio = bbox_area(candidate) / previous_area
    return (
        center_distance <= MAX_CENTER_DISTANCE_RATIO * previous_diagonal
        and MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO
    )


def select_primary_detection(
    detections: np.ndarray,
    previous: np.ndarray | None,
    reliable_conf_thres: float,
) -> np.ndarray | None:
    """选择单个主脸；低分候选必须与近期主脸在位置和面积上连续。"""

    if len(detections) == 0:
        return None

    if previous is not None:
        consistent = [
            detection for detection in detections if is_position_consistent(detection, previous)
        ]
        if consistent:
            return max(consistent, key=lambda detection: float(detection[4])).copy()

    reliable = [
        detection
        for detection in detections
        if float(detection[4]) >= reliable_conf_thres
    ]
    if reliable:
        return max(reliable, key=lambda detection: float(detection[4])).copy()
    return None


def draw_primary_detection(image: np.ndarray, detection: np.ndarray) -> None:
    x1, y1, x2, y2 = (round(float(value)) for value in detection[:4])
    height, width = image.shape[:2]
    outer_top_left = (max(x1 - 4, 0), max(y1 - 4, 0))
    outer_bottom_right = (min(x2 + 4, width - 1), min(y2 + 4, height - 1))
    cv2.rectangle(image, outer_top_left, outer_bottom_right, PRIMARY_FACE_COLOR, 2)
    label_y = y2 + 20 if y2 + 20 < height else max(y1 + 20, 18)
    cv2.putText(
        image,
        "PRIMARY",
        (max(x1, 0), label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        PRIMARY_FACE_COLOR,
        2,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yolov5_face_model(
    weights_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    """兼容 PyTorch 2.6+，且只对已校验的官方 TFW 权重启用完整反序列化。"""

    weights_hash = sha256(weights_path)
    if weights_hash != TRUSTED_TFW_SHA256:
        raise RuntimeError(
            "权重 SHA-256 与项目记录的 TFW 官方权重不一致，已拒绝使用 "
            "weights_only=False 加载。\n"
            f"实际 SHA-256：{weights_hash}\n"
            f"预期 SHA-256：{TRUSTED_TFW_SHA256}"
        )

    # 上游 models/yolo.py 在模块顶层导入 thop，但推理并不使用它。某些
    # Notebook 环境没有安装 thop，此兼容模块只允许模型反序列化正常导入。
    if "thop" not in sys.modules:
        try:
            import thop  # noqa: F401
        except ModuleNotFoundError:
            thop_compat = types.ModuleType("thop")

            def thop_not_installed(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("当前环境未安装 thop，无法执行 FLOPs/参数量分析。")

            thop_compat.profile = thop_not_installed
            thop_compat.clever_format = thop_not_installed
            sys.modules["thop"] = thop_compat

    original_torch_load = torch.load

    def trusted_checkpoint_load(*args: Any, **kwargs: Any) -> Any:
        # YOLOv5-Face 旧检查点保存了完整 nn.Module。PyTorch 2.6+ 默认
        # weights_only=True，无法加载该格式；这里只对哈希已验证的官方文件关闭限制。
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = trusted_checkpoint_load
    try:
        model = attempt_load(str(weights_path), map_location=device)
    finally:
        torch.load = original_torch_load
    return model, weights_hash


def detection_row(
    image_path: Path,
    frame_index: int | None,
    metadata: dict[str, Any],
    width: int,
    height: int,
    face_count: int,
    face_index: int | str,
    detection: np.ndarray | None,
    is_primary: bool,
    timings: dict[str, float],
    device: torch.device,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image": str(image_path),
        "frame_index": "" if frame_index is None else frame_index,
        "source_dtype": metadata["source_dtype"],
        "width": width,
        "height": height,
        "normalization": metadata["normalization"],
        "scale_low": metadata["scale_low"],
        "scale_high": metadata["scale_high"],
        "temperature_min_c": metadata.get("temperature_min_c", ""),
        "temperature_max_c": metadata.get("temperature_max_c", ""),
        "temperature_mean_c": metadata.get("temperature_mean_c", ""),
        "face_count": face_count,
        "face_index": face_index,
        "is_primary": int(is_primary),
        "confidence": "",
        "x1": "",
        "y1": "",
        "x2": "",
        "y2": "",
        "preprocess_ms": round(timings["preprocess_ms"], 3),
        "inference_ms": round(timings["inference_ms"], 3),
        "nms_ms": round(timings["nms_ms"], 3),
        "total_ms": round(timings["total_ms"], 3),
        "device": str(device),
    }
    for name in LANDMARK_NAMES:
        row[f"{name}_x"] = ""
        row[f"{name}_y"] = ""
    if detection is not None:
        row.update(
            {
                "confidence": round(float(detection[4]), 6),
                "x1": round(float(detection[0]), 3),
                "y1": round(float(detection[1]), 3),
                "x2": round(float(detection[2]), 3),
                "y2": round(float(detection[3]), 3),
            }
        )
        for index, name in enumerate(LANDMARK_NAMES):
            row[f"{name}_x"] = round(float(detection[5 + 2 * index]), 3)
            row[f"{name}_y"] = round(float(detection[6 + 2 * index]), 3)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image",
        "frame_index",
        "source_dtype",
        "width",
        "height",
        "normalization",
        "scale_low",
        "scale_high",
        "temperature_min_c",
        "temperature_max_c",
        "temperature_mean_c",
        "face_count",
        "face_index",
        "is_primary",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    for name in LANDMARK_NAMES:
        fieldnames.extend((f"{name}_x", f"{name}_y"))
    fieldnames.extend(("preprocess_ms", "inference_ms", "nms_ms", "total_ms", "device"))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    validate_args(args)
    input_path = resolve_path(args.input)
    weights_path = resolve_path(args.weights)
    requested_output = resolve_path(args.output)
    if not weights_path.is_file():
        raise FileNotFoundError(f"未找到权重文件：{weights_path}")

    is_bin = input_path.is_file() and input_path.suffix.lower() == ".bin"
    bin_info: ThermalBinInfo | None = None
    if is_bin:
        bin_info = read_bin_header(input_path)
        max_frames = args.max_frames if args.max_frames > 0 else None
        frame_indices = select_frame_indices(
            bin_info.total_frames,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_step=args.frame_step,
            max_frames=max_frames,
        )
        input_count = len(frame_indices)
        image_paths: list[Path] = []
        if input_count == 0:
            raise ValueError("当前 BIN 帧范围没有可处理的完整帧。")
        print(
            f"BIN 信息：{bin_info.width}x{bin_info.height}，"
            f"完整帧 {bin_info.total_frames}，本次抽取 {input_count} 帧"
        )
        if bin_info.trailing_bytes:
            print(f"提示：文件尾有 {bin_info.trailing_bytes} 字节不足一帧，已安全忽略。")
    else:
        image_paths = collect_images(input_path)
        input_count = len(image_paths)
    output_dir = choose_output_dir(requested_output)
    device = select_device(args.device)

    print(f"加载模型：{weights_path}")
    print(f"运行设备：{device}")
    model, weights_hash = load_yolov5_face_model(weights_path, device)
    model.eval()
    stride = int(model.stride.max())
    image_size = int(check_img_size(args.img_size, s=stride))

    with torch.inference_mode():
        warmup = torch.zeros((1, 3, image_size, image_size), device=device)
        model(warmup)[0]
        synchronize(device)

    csv_rows: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    total_faces = 0
    primary_frame_count = 0
    low_confidence_primary_count = 0
    previous_primary: np.ndarray | None = None
    previous_primary_frame: int | None = None
    max_primary_gap_frames = max(
        0, round(args.max_interpolation_seconds * args.source_fps)
    )
    print(
        f"主脸关联：候选阈值 {args.conf_thres:.2f}，"
        f"可靠阈值 {args.reliable_conf_thres:.2f}，"
        f"最长短缺失 {max_primary_gap_frames} 帧"
    )
    effective_save_images = args.save_images
    if effective_save_images == "auto":
        effective_save_images = "missed" if is_bin else "all"
    still_dir: Path | None = None
    if effective_save_images == "all":
        still_dir = output_dir / "annotated"
        still_dir.mkdir(parents=True, exist_ok=True)
    elif effective_save_images == "missed":
        still_dir = output_dir / "missed"
        still_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / args.video_name if args.save_video else None
    video_fps = max(args.source_fps / args.frame_step, 0.1) if is_bin else args.source_fps
    video_writer: cv2.VideoWriter | None = None
    show_enabled = args.show
    stopped_by_user = False
    saved_image_count = 0

    for image_index, (image_path, frame_index, raw) in enumerate(
        iter_inputs(input_path, image_paths, args), start=1
    ):
        total_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        image_bgr, metadata = make_detection_view(
            raw,
            args.normalization,
            args.lower_percentile,
            args.upper_percentile,
        )
        if frame_index is not None:
            temperature = raw_to_celsius(raw)
            metadata.update(
                {
                    "temperature_min_c": round(float(temperature.min()), 3),
                    "temperature_max_c": round(float(temperature.max()), 3),
                    "temperature_mean_c": round(float(temperature.mean()), 3),
                }
            )
        tensor = preprocess(image_bgr, image_size, device)
        synchronize(device)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

        inference_start = time.perf_counter()
        with torch.inference_mode():
            prediction = model(tensor)[0]
        synchronize(device)
        inference_ms = (time.perf_counter() - inference_start) * 1000

        nms_start = time.perf_counter()
        detections = non_max_suppression_face(
            prediction,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
        )[0]
        if len(detections):
            detections[:, :4] = scale_coords(tensor.shape[2:], detections[:, :4], image_bgr.shape).round()
            detections[:, 5:15] = scale_landmarks(
                (tensor.shape[2], tensor.shape[3]), detections[:, 5:15], image_bgr.shape
            )
            detection_array = detections.detach().cpu().numpy()
        else:
            detection_array = np.empty((0, 16), dtype=np.float32)
        synchronize(device)
        nms_ms = (time.perf_counter() - nms_start) * 1000

        previous_for_selection: np.ndarray | None = None
        if is_bin and previous_primary is not None and previous_primary_frame is not None:
            assert frame_index is not None
            missing_since_primary = max(frame_index - previous_primary_frame - 1, 0)
            if missing_since_primary <= max_primary_gap_frames:
                previous_for_selection = previous_primary
        selected_primary = select_primary_detection(
            detection_array,
            previous_for_selection,
            args.reliable_conf_thres,
        )
        if selected_primary is not None:
            previous_primary = selected_primary.copy() if is_bin else None
            previous_primary_frame = frame_index if is_bin else None
            primary_frame_count += 1
            if float(selected_primary[4]) < args.reliable_conf_thres:
                low_confidence_primary_count += 1
        primary_confidence = (
            float(selected_primary[4]) if selected_primary is not None else None
        )

        annotated = image_bgr.copy()
        for detection in detection_array:
            draw_detection(annotated, detection)
        if selected_primary is not None:
            draw_primary_detection(annotated, selected_primary)
        face_count = len(detection_array)
        draw_frame_status(annotated, frame_index, face_count, primary_confidence)
        if frame_index is None:
            output_name = f"{image_index:04d}_{image_path.stem}.png"
            display_name = image_path.name
        else:
            output_name = f"{image_path.stem}_frame_{frame_index:06d}.png"
            display_name = f"{image_path.name} frame={frame_index}"

        should_save_image = effective_save_images == "all" or (
            effective_save_images == "missed" and selected_primary is None
        )
        output_image: Path | None = None
        if should_save_image and still_dir is not None:
            output_image = still_dir / output_name
            write_image_unicode(output_image, annotated)
            saved_image_count += 1

        if video_path is not None:
            if video_writer is None:
                video_writer = create_avi_writer(
                    video_path,
                    width=annotated.shape[1],
                    height=annotated.shape[0],
                    fps=video_fps,
                )
                print(f"AVI 输出：{video_path}，播放帧率 {video_fps:.3f} FPS")
            video_writer.write(annotated)

        if show_enabled and image_index % args.show_every == 0:
            try:
                cv2.imshow("TFW thermal face detection - Q/Esc to stop", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    stopped_by_user = True
            except cv2.error as error:
                print(f"实时窗口不可用，后续仅保存结果：{error}")
                show_enabled = False

        total_ms = (time.perf_counter() - total_start) * 1000
        timings = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "nms_ms": nms_ms,
            "total_ms": total_ms,
        }
        total_faces += face_count
        per_image.append(
            {
                "image": str(image_path),
                "frame_index": frame_index,
                "saved_image": str(output_image) if output_image is not None else None,
                "face_count": face_count,
                "primary_detected": selected_primary is not None,
                "primary_confidence": (
                    round(primary_confidence, 6)
                    if primary_confidence is not None
                    else None
                ),
                **{key: round(value, 3) for key, value in timings.items()},
            }
        )

        if face_count:
            for face_index, detection in enumerate(detection_array, start=1):
                is_primary = selected_primary is not None and np.allclose(
                    detection[:5], selected_primary[:5], rtol=1e-5, atol=1e-4
                )
                csv_rows.append(
                    detection_row(
                        image_path,
                        frame_index,
                        metadata,
                        image_bgr.shape[1],
                        image_bgr.shape[0],
                        face_count,
                        face_index,
                        detection,
                        is_primary,
                        timings,
                        device,
                    )
                )
        else:
            csv_rows.append(
                detection_row(
                    image_path,
                    frame_index,
                    metadata,
                    image_bgr.shape[1],
                    image_bgr.shape[0],
                    0,
                    "",
                    None,
                    False,
                    timings,
                    device,
                )
            )
        should_log = (
            image_index == 1
            or image_index == input_count
            or image_index % args.log_every == 0
            or selected_primary is None
        )
        if should_log:
            primary_log = (
                f"主脸 {primary_confidence:.2f}"
                if primary_confidence is not None
                else "未关联主脸"
            )
            print(
                f"[{image_index}/{input_count}] {display_name}: "
                f"{face_count} 个候选，{primary_log}，推理 {inference_ms:.1f} ms"
            )
        if stopped_by_user:
            print("收到 Q/Esc，提前停止并保存已完成结果。")
            break

    if video_writer is not None:
        video_writer.release()
    if args.show:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    write_csv(output_dir / "detections.csv", csv_rows)
    summary = {
        "task": "thermal_face_detection_and_5_landmarks",
        "note": "这是检测模型，不负责人员身份识别。",
        "input_type": "thermal_bin" if is_bin else "images",
        "input": str(input_path),
        "output": str(output_dir),
        "weights": str(weights_path),
        "weights_sha256": weights_hash,
        "model_source": MODEL_SOURCE,
        "vendor_commit": VENDOR_COMMIT,
        "device": str(device),
        "torch_version": torch.__version__,
        "image_size": image_size,
        "conf_threshold": args.conf_thres,
        "reliable_conf_threshold": args.reliable_conf_thres,
        "max_primary_gap_frames": max_primary_gap_frames,
        "max_interpolation_seconds": args.max_interpolation_seconds,
        "iou_threshold": args.iou_thres,
        "normalization": args.normalization,
        "selected_frame_count": input_count,
        "processed_frame_count": len(per_image),
        "stopped_by_user": stopped_by_user,
        "save_images": effective_save_images,
        "saved_image_count": saved_image_count,
        "video": str(video_path) if video_path is not None else None,
        "video_fps": video_fps if video_path is not None else None,
        "log_every": args.log_every,
        "bin_info": bin_info.to_dict() if bin_info is not None else None,
        "frame_selection": {
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "frame_step": args.frame_step,
            "max_frames": args.max_frames,
        }
        if is_bin
        else None,
        "images_with_faces": sum(item["face_count"] > 0 for item in per_image),
        "images_without_faces": sum(item["face_count"] == 0 for item in per_image),
        "frames_with_primary": primary_frame_count,
        "frames_without_primary": len(per_image) - primary_frame_count,
        "low_confidence_primary_frames": low_confidence_primary_count,
        "total_faces": total_faces,
        "mean_inference_ms": round(statistics.mean(item["inference_ms"] for item in per_image), 3),
        "mean_total_ms": round(statistics.mean(item["total_ms"] for item in per_image), 3),
        "images": per_image,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)

    print(
        f"完成：共处理 {len(per_image)} 个输入帧，"
        f"关联主脸 {primary_frame_count} 帧，未关联 {len(per_image) - primary_frame_count} 帧；"
        f"共保留 {total_faces} 个候选框。"
    )
    if still_dir is not None:
        print(f"静态图片（{effective_save_images}）：{still_dir}，共 {saved_image_count} 张")
    if video_path is not None:
        print(f"AVI 视频：{video_path}")
    print(f"检测表：{output_dir / 'detections.csv'}")
    print(f"汇总表：{output_dir / 'summary.json'}")
    return output_dir


if __name__ == "__main__":
    main()
