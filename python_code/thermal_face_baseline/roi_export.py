"""主脸温度 ROI、轨迹坐标和 MATLAB v7.3 导出。"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from thermal_bin import ThermalBinReader, raw_to_celsius


ROI_STATUS_INVALID = np.uint8(0)
ROI_STATUS_DETECTED = np.uint8(1)
ROI_STATUS_INTERPOLATED = np.uint8(2)
ROI_STATUS_NAMES = ("invalid", "detected", "interpolated")


def require_hdf5storage() -> Any:
    """按需导入 MATLAB v7.3 写入依赖，避免普通检测强制安装。"""

    try:
        import hdf5storage
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "启用 ROI MAT 导出需要 hdf5storage。请在当前 Notebook 内核执行：\n"
            "%pip install 'hdf5storage>=0.2,<0.3'\n"
            "安装完成后重启内核并重新运行。"
        ) from error
    return hdf5storage


def _segments(frame_indices: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """把布尔状态压缩为首尾均包含的原始帧区间。"""

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_valid in enumerate(valid):
        if bool(is_valid) and start is None:
            start = index
        is_last = index == len(valid) - 1
        if start is not None and (not bool(is_valid) or is_last):
            end = index if bool(is_valid) and is_last else index - 1
            ranges.append((int(frame_indices[start]), int(frame_indices[end])))
            start = None
    return np.asarray(ranges, dtype=np.int64).reshape(-1, 2)


def _expanded_bbox(bbox: np.ndarray, padding: float) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = x2 - x1
    height = y2 - y1
    return np.asarray(
        [
            x1 - padding * width,
            y1 - padding * height,
            x2 + padding * width,
            y2 + padding * height,
        ],
        dtype=np.float32,
    )


def _landmark_small_bbox(
    bbox: np.ndarray,
    landmarks: np.ndarray,
    side_ratio: float,
) -> np.ndarray:
    """由鼻尖和嘴角估计口鼻区域；异常关键点退回人脸中心偏下。"""

    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    side = side_ratio * min(width, height)

    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    nose = points[2]
    mouth_midpoint = (points[3] + points[4]) / 2.0
    tolerance_x = 0.25 * width
    tolerance_y = 0.25 * height
    relevant = np.vstack((nose, points[3], points[4]))
    landmarks_valid = bool(
        np.all(np.isfinite(relevant))
        and np.all(relevant[:, 0] >= x1 - tolerance_x)
        and np.all(relevant[:, 0] <= x2 + tolerance_x)
        and np.all(relevant[:, 1] >= y1 - tolerance_y)
        and np.all(relevant[:, 1] <= y2 + tolerance_y)
        and float(np.linalg.norm(points[3] - points[4])) >= 1.0
    )
    if landmarks_valid:
        center = (nose + mouth_midpoint) / 2.0
    else:
        center = np.asarray(((x1 + x2) / 2.0, y1 + 0.65 * height), dtype=np.float32)

    half = side / 2.0
    return np.asarray(
        [center[0] - half, center[1] - half, center[0] + half, center[1] + half],
        dtype=np.float32,
    )


def build_roi_track(
    frame_indices: Sequence[int],
    primary_detections: Sequence[np.ndarray | None],
    max_interpolation_frames: int,
    roi_padding: float = 0.05,
    small_roi_ratio: float = 0.50,
) -> dict[str, np.ndarray]:
    """补全短缺失几何并生成逐帧 ROI 轨迹数组。"""

    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    frame_count = len(frames)
    if frame_count == 0:
        raise ValueError("ROI 导出没有可处理帧。")
    if len(primary_detections) != frame_count:
        raise ValueError("帧号数量与主脸检测数量不一致。")
    if max_interpolation_frames < 0:
        raise ValueError("max_interpolation_frames 不能小于 0。")
    if frame_count > 1 and not np.all(np.diff(frames) == 1):
        raise ValueError("ROI MAT 导出要求原始帧连续，请使用 frame_step=1。")

    status = np.zeros(frame_count, dtype=np.uint8)
    confidence = np.full(frame_count, np.nan, dtype=np.float32)
    bbox = np.full((frame_count, 4), np.nan, dtype=np.float32)
    landmarks = np.full((frame_count, 5, 2), np.nan, dtype=np.float32)

    for index, detection in enumerate(primary_detections):
        if detection is None:
            continue
        values = np.asarray(detection, dtype=np.float32).reshape(-1)
        if values.size < 15 or not np.all(np.isfinite(values[:15])):
            continue
        status[index] = ROI_STATUS_DETECTED
        confidence[index] = values[4]
        bbox[index] = values[:4]
        landmarks[index] = values[5:15].reshape(5, 2)

    index = 0
    while index < frame_count:
        if status[index] != ROI_STATUS_INVALID:
            index += 1
            continue
        gap_start = index
        while index < frame_count and status[index] == ROI_STATUS_INVALID:
            index += 1
        gap_end = index - 1
        gap_length = gap_end - gap_start + 1
        left = gap_start - 1
        right = gap_end + 1
        bounded = left >= 0 and right < frame_count
        if bounded and gap_length <= max_interpolation_frames:
            for missing_index in range(gap_start, gap_end + 1):
                alpha = (missing_index - left) / (right - left)
                bbox[missing_index] = (1.0 - alpha) * bbox[left] + alpha * bbox[right]
                landmarks[missing_index] = (
                    (1.0 - alpha) * landmarks[left] + alpha * landmarks[right]
                )
                status[missing_index] = ROI_STATUS_INTERPOLATED

    roi_bbox = np.full_like(bbox, np.nan)
    small_bbox = np.full_like(bbox, np.nan)
    roi_center = np.full((frame_count, 2), np.nan, dtype=np.float32)
    valid = status != ROI_STATUS_INVALID
    for index in np.flatnonzero(valid):
        roi_bbox[index] = _expanded_bbox(bbox[index], roi_padding)
        small_bbox[index] = _landmark_small_bbox(
            bbox[index], landmarks[index], small_roi_ratio
        )
        roi_center[index] = (
            (roi_bbox[index, 0] + roi_bbox[index, 2]) / 2.0,
            (roi_bbox[index, 1] + roi_bbox[index, 3]) / 2.0,
        )

    return {
        "frameIndices": frames,
        "roiStatus": status,
        "confidence": confidence,
        "bbox": bbox,
        "landmarks": landmarks,
        "roiBbox": roi_bbox,
        "smallBbox": small_bbox,
        "roiCenter": roi_center,
        "validSegments": _segments(frames, valid),
        "invalidSegments": _segments(frames, ~valid),
    }


def extract_temperature_roi(
    temperature: np.ndarray,
    bbox: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, float]:
    """按浮点框双线性采样，图像外区域保持 NaN。"""

    output = np.full((output_size, output_size), np.nan, dtype=np.float32)
    values = np.asarray(bbox, dtype=np.float32).reshape(4)
    if output_size <= 0 or not np.all(np.isfinite(values)):
        return output, 0.0
    x1, y1, x2, y2 = (float(value) for value in values)
    if x2 <= x1 or y2 <= y1:
        return output, 0.0

    x_coordinates = np.linspace(x1, x2, output_size, dtype=np.float32)
    y_coordinates = np.linspace(y1, y2, output_size, dtype=np.float32)
    map_x, map_y = np.meshgrid(x_coordinates, y_coordinates)
    source = np.asarray(temperature, dtype=np.float32)
    sampled = cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    valid_weight = cv2.remap(
        np.ones(source.shape, dtype=np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    sampled[valid_weight < 0.999] = np.nan
    return sampled.astype(np.float32, copy=False), float(np.mean(valid_weight))


def _mat_column(values: np.ndarray, dtype: np.dtype[Any] | type) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape(-1, 1)


def _save_v73_mat(path: Path, data: dict[str, Any]) -> None:
    hdf5storage = require_hdf5storage()
    hdf5storage.savemat(
        str(path),
        data,
        appendmat=False,
        format="7.3",
        oned_as="column",
        store_python_metadata=False,
    )


def _draw_box(image: np.ndarray, bbox: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    if not np.all(np.isfinite(bbox)):
        return
    x1, y1, x2, y2 = (round(float(value)) for value in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (max(x1, 0), max(y1 - 7, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
    )


def _draw_tracking_preview(image: np.ndarray, track: dict[str, np.ndarray], index: int) -> None:
    status = int(track["roiStatus"][index])
    if status == int(ROI_STATUS_INVALID):
        color = (0, 0, 255)
        label = "invalid"
    elif status == int(ROI_STATUS_INTERPOLATED):
        color = (255, 255, 0)
        label = "interpolated"
    else:
        confidence = float(track["confidence"][index])
        color = (0, 200, 0) if confidence >= 0.60 else (0, 255, 255) if confidence >= 0.40 else (0, 165, 255)
        label = f"detected {confidence:.2f}"

    if status != int(ROI_STATUS_INVALID):
        _draw_box(image, track["roiBbox"][index], color, f"ROI {label}")
        _draw_box(image, track["smallBbox"][index], (255, 0, 255), "mouth-nose")
        center = track["roiCenter"][index]
        cv2.circle(image, (round(float(center[0])), round(float(center[1]))), 4, color, -1)

    frame_index = int(track["frameIndices"][index])
    cv2.rectangle(image, (0, 0), (560, 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        f"frame {frame_index} | {label}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        color,
        2,
    )


def _csv_value(value: float, digits: int = 6) -> float | str:
    return round(float(value), digits) if np.isfinite(value) else ""


def _write_track_csv(
    path: Path,
    track: dict[str, np.ndarray],
    source_fps: float,
    roi_coverage: np.ndarray,
    small_coverage: np.ndarray,
    mat_part_names: Sequence[str],
) -> None:
    fields = [
        "frame_index", "time_seconds", "status", "status_name", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2",
        "small_x1", "small_y1", "small_x2", "small_y2",
        "roi_center_x", "roi_center_y", "roi_coverage", "small_coverage",
        "mat_file",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, frame_index in enumerate(track["frameIndices"]):
            status = int(track["roiStatus"][index])
            bbox = track["bbox"][index]
            roi_bbox = track["roiBbox"][index]
            small_bbox = track["smallBbox"][index]
            center = track["roiCenter"][index]
            writer.writerow(
                {
                    "frame_index": int(frame_index),
                    "time_seconds": round(float(frame_index) / source_fps, 9),
                    "status": status,
                    "status_name": ROI_STATUS_NAMES[status],
                    "confidence": _csv_value(track["confidence"][index]),
                    "bbox_x1": _csv_value(bbox[0], 3), "bbox_y1": _csv_value(bbox[1], 3),
                    "bbox_x2": _csv_value(bbox[2], 3), "bbox_y2": _csv_value(bbox[3], 3),
                    "roi_x1": _csv_value(roi_bbox[0], 3), "roi_y1": _csv_value(roi_bbox[1], 3),
                    "roi_x2": _csv_value(roi_bbox[2], 3), "roi_y2": _csv_value(roi_bbox[3], 3),
                    "small_x1": _csv_value(small_bbox[0], 3), "small_y1": _csv_value(small_bbox[1], 3),
                    "small_x2": _csv_value(small_bbox[2], 3), "small_y2": _csv_value(small_bbox[3], 3),
                    "roi_center_x": _csv_value(center[0], 3), "roi_center_y": _csv_value(center[1], 3),
                    "roi_coverage": _csv_value(roi_coverage[index]),
                    "small_coverage": _csv_value(small_coverage[index]),
                    "mat_file": mat_part_names[index],
                }
            )


def export_roi_artifacts(
    bin_path: Path,
    output_dir: Path,
    frame_indices: Sequence[int],
    primary_detections: Sequence[np.ndarray | None],
    source_fps: float,
    max_interpolation_frames: int,
    roi_size: int = 128,
    small_roi_size: int = 64,
    mat_chunk_frames: int = 3000,
    save_mat: bool = True,
    save_preview: bool = True,
    preview_view: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """从原始热像二次读取并导出温度 Tensor、轨迹表和预览。"""

    if roi_size <= 0 or small_roi_size <= 0 or mat_chunk_frames <= 0:
        raise ValueError("ROI 尺寸和 MAT 分片帧数必须大于 0。")
    if source_fps <= 0:
        raise ValueError("source_fps 必须大于 0。")
    if save_mat:
        require_hdf5storage()
    if save_preview and preview_view is None:
        raise ValueError("启用 ROI 预览时必须提供 preview_view。")

    track = build_roi_track(
        frame_indices,
        primary_detections,
        max_interpolation_frames=max_interpolation_frames,
    )
    frame_count = len(track["frameIndices"])
    roi_coverage = np.zeros(frame_count, dtype=np.float32)
    small_coverage = np.zeros(frame_count, dtype=np.float32)
    part_count = max(1, math.ceil(frame_count / mat_chunk_frames))
    mat_paths: list[Path] = []
    mat_part_names = [""] * frame_count
    preview_path = output_dir / "roi_tracking_preview.avi" if save_preview else None
    preview_writer: cv2.VideoWriter | None = None

    with ThermalBinReader(bin_path) as reader:
        if save_preview:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            preview_writer = cv2.VideoWriter(
                str(preview_path), fourcc, source_fps, (reader.info.width, reader.info.height)
            )
            if not preview_writer.isOpened():
                preview_writer.release()
                raise RuntimeError(f"无法创建 ROI 预览 AVI：{preview_path}")

        try:
            for part_index in range(part_count):
                start = part_index * mat_chunk_frames
                stop = min(start + mat_chunk_frames, frame_count)
                length = stop - start
                roi_tensor = (
                    np.full((roi_size, roi_size, length), np.nan, dtype=np.float32)
                    if save_mat
                    else None
                )
                small_tensor = (
                    np.full(
                        (small_roi_size, small_roi_size, length),
                        np.nan,
                        dtype=np.float32,
                    )
                    if save_mat
                    else None
                )
                if save_mat:
                    if part_count == 1:
                        mat_path = output_dir / "roiTensor.mat"
                    else:
                        mat_path = output_dir / f"roiTensor_part{part_index + 1:03d}.mat"
                    mat_paths.append(mat_path)
                    for index in range(start, stop):
                        mat_part_names[index] = mat_path.name

                for index in range(start, stop):
                    raw = reader.read_frame(int(track["frameIndices"][index]), output="raw")
                    if track["roiStatus"][index] != ROI_STATUS_INVALID:
                        temperature = raw_to_celsius(raw)
                        roi, roi_valid = extract_temperature_roi(
                            temperature, track["roiBbox"][index], roi_size
                        )
                        small, small_valid = extract_temperature_roi(
                            temperature, track["smallBbox"][index], small_roi_size
                        )
                        if roi_tensor is not None and small_tensor is not None:
                            roi_tensor[:, :, index - start] = roi
                            small_tensor[:, :, index - start] = small
                        roi_coverage[index] = roi_valid
                        small_coverage[index] = small_valid

                    if preview_writer is not None:
                        assert preview_view is not None
                        preview = np.ascontiguousarray(preview_view(raw).copy())
                        _draw_tracking_preview(preview, track, index)
                        preview_writer.write(preview)

                if save_mat:
                    assert roi_tensor is not None and small_tensor is not None
                    part_slice = slice(start, stop)
                    mat_data: dict[str, Any] = {
                        "roiTensor": roi_tensor,
                        "smallTensor": small_tensor,
                        "frameIndices": _mat_column(track["frameIndices"][part_slice], np.int64),
                        "timeSeconds": _mat_column(
                            track["frameIndices"][part_slice] / source_fps, np.float64
                        ),
                        "roiStatus": _mat_column(track["roiStatus"][part_slice], np.uint8),
                        "confidence": _mat_column(track["confidence"][part_slice], np.float32),
                        "bbox": track["bbox"][part_slice].astype(np.float32),
                        "roiBbox": track["roiBbox"][part_slice].astype(np.float32),
                        "landmarks": track["landmarks"][part_slice].astype(np.float32),
                        "smallBbox": track["smallBbox"][part_slice].astype(np.float32),
                        "roiCenter": track["roiCenter"][part_slice].astype(np.float32),
                        "roiCoverage": _mat_column(roi_coverage[part_slice], np.float32),
                        "smallCoverage": _mat_column(small_coverage[part_slice], np.float32),
                        "sourceFPS": np.asarray([[source_fps]], dtype=np.float64),
                        "partIndex": np.asarray([[part_index + 1]], dtype=np.int32),
                        "partCount": np.asarray([[part_count]], dtype=np.int32),
                    }
                    if part_count == 1:
                        mat_data["validSegments"] = track["validSegments"]
                        mat_data["invalidSegments"] = track["invalidSegments"]
                    _save_v73_mat(mat_paths[-1], mat_data)
        finally:
            if preview_writer is not None:
                preview_writer.release()

    manifest_path: Path | None = None
    if save_mat and part_count > 1:
        manifest_path = output_dir / "roiManifest.mat"
        part_ranges = np.asarray(
            [
                (
                    int(track["frameIndices"][part_index * mat_chunk_frames]),
                    int(
                        track["frameIndices"][
                            min((part_index + 1) * mat_chunk_frames, frame_count) - 1
                        ]
                    ),
                )
                for part_index in range(part_count)
            ],
            dtype=np.int64,
        )
        _save_v73_mat(
            manifest_path,
            {
                "dataFileNames": np.asarray(
                    [path.name for path in mat_paths], dtype=object
                ).reshape(-1, 1),
                "partFrameRanges": part_ranges,
                "validSegments": track["validSegments"],
                "invalidSegments": track["invalidSegments"],
                "sourceFPS": np.asarray([[source_fps]], dtype=np.float64),
                "roiSize": np.asarray([[roi_size]], dtype=np.int32),
                "smallRoiSize": np.asarray([[small_roi_size]], dtype=np.int32),
            },
        )

    csv_path = output_dir / "roi_track.csv"
    _write_track_csv(
        csv_path,
        track,
        source_fps,
        roi_coverage,
        small_coverage,
        mat_part_names,
    )
    status = track["roiStatus"]
    return {
        "frame_count": frame_count,
        "detected_frames": int(np.count_nonzero(status == ROI_STATUS_DETECTED)),
        "interpolated_frames": int(np.count_nonzero(status == ROI_STATUS_INTERPOLATED)),
        "invalid_frames": int(np.count_nonzero(status == ROI_STATUS_INVALID)),
        "mat_files": [str(path) for path in mat_paths],
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "track_csv": str(csv_path),
        "preview_video": str(preview_path) if preview_path is not None else None,
        "valid_segments": track["validSegments"].tolist(),
        "invalid_segments": track["invalidSegments"].tolist(),
        "roi_size": roi_size,
        "small_roi_size": small_roi_size,
        "mat_chunk_frames": mat_chunk_frames,
    }
