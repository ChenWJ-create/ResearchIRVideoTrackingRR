"""按帧读取 16UC1 热成像 BIN 文件。

文件布局：4 个 little-endian int32 头字段，随后连续存放 height * width
个 little-endian uint16 像素的图像帧。整个视频不会一次性载入内存。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Literal

import numpy as np

HEADER_ITEMS = 4
HEADER_DTYPE = np.dtype("<i4")
FRAME_DTYPE = np.dtype("<u2")
HEADER_BYTES = HEADER_ITEMS * HEADER_DTYPE.itemsize
TEMPERATURE_SCALE = 64.0
KELVIN_OFFSET = 273.15
OutputMode = Literal["raw", "temperature"]


@dataclass(frozen=True)
class ThermalBinInfo:
    path: str
    height: int
    width: int
    header_values: tuple[int, int, int, int]
    total_bytes: int
    frame_bytes: int
    total_frames: int
    trailing_bytes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_bin_header(bin_path: str | Path) -> ThermalBinInfo:
    """读取文件头并计算可完整读取的帧数。"""

    path = Path(bin_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BIN 文件不存在：{path}")

    with path.open("rb", buffering=0) as stream:
        header = np.fromfile(stream, dtype=HEADER_DTYPE, count=HEADER_ITEMS)
    if header.size != HEADER_ITEMS:
        raise ValueError(f"BIN 文件头不足 {HEADER_BYTES} 字节：{path}")

    height, width = int(header[0]), int(header[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"无效图像尺寸：height={height}, width={width}")

    frame_bytes = height * width * FRAME_DTYPE.itemsize
    total_bytes = path.stat().st_size
    payload_bytes = max(total_bytes - HEADER_BYTES, 0)
    total_frames, trailing_bytes = divmod(payload_bytes, frame_bytes)
    if total_frames == 0:
        raise ValueError(f"BIN 文件不包含完整图像帧：{path}")

    return ThermalBinInfo(
        path=str(path),
        height=height,
        width=width,
        header_values=tuple(int(value) for value in header),
        total_bytes=total_bytes,
        frame_bytes=frame_bytes,
        total_frames=total_frames,
        trailing_bytes=trailing_bytes,
    )


def raw_to_celsius(raw_frame: np.ndarray) -> np.ndarray:
    """按相机公式 raw / 64 - 273.15 转为摄氏温度（float32）。"""

    return raw_frame.astype(np.float32) / TEMPERATURE_SCALE - KELVIN_OFFSET


class ThermalBinReader:
    """单次打开文件并支持随机跳帧，适合长视频循环读取。"""

    def __init__(self, bin_path: str | Path):
        self.info = read_bin_header(bin_path)
        self._stream: BinaryIO | None = None

    def __enter__(self) -> "ThermalBinReader":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __len__(self) -> int:
        return self.info.total_frames

    def open(self) -> None:
        if self._stream is None or self._stream.closed:
            self._stream = Path(self.info.path).open("rb", buffering=0)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def read_frame(self, frame_index: int, output: OutputMode = "raw") -> np.ndarray:
        """读取指定帧；output='raw' 返回 uint16，'temperature' 返回摄氏度。"""

        if not 0 <= frame_index < self.info.total_frames:
            raise IndexError(
                f"帧号越界：{frame_index}，有效范围为 0 到 {self.info.total_frames - 1}"
            )
        if output not in ("raw", "temperature"):
            raise ValueError("output 只能是 'raw' 或 'temperature'。")

        self.open()
        assert self._stream is not None
        offset = HEADER_BYTES + frame_index * self.info.frame_bytes
        self._stream.seek(offset, 0)
        frame = np.fromfile(
            self._stream,
            dtype=FRAME_DTYPE,
            count=self.info.height * self.info.width,
        )
        if frame.size != self.info.height * self.info.width:
            raise EOFError(f"第 {frame_index} 帧数据不完整：只读到 {frame.size} 个像素。")

        # 等价于用户代码：reshape((width, height), order='F').T
        frame = frame.reshape((self.info.height, self.info.width))
        return raw_to_celsius(frame) if output == "temperature" else frame


def read_bin_frame(
    bin_path: str | Path,
    frame_index: int = 0,
    output: OutputMode = "raw",
) -> np.ndarray:
    """方便随机读取单帧；循环处理大量帧时优先使用 iter_bin_frames。"""

    with ThermalBinReader(bin_path) as reader:
        return reader.read_frame(frame_index, output=output)


# 保留用户习惯的无下划线函数名。
readbinframe = read_bin_frame


def select_frame_indices(
    total_frames: int,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    max_frames: int | None = None,
) -> range:
    """生成待处理帧号；end_frame 使用 Python 惯例，不包含终止帧。"""

    if start_frame < 0:
        raise ValueError("start_frame 不能小于 0。")
    if frame_step <= 0:
        raise ValueError("frame_step 必须大于 0。")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames 不能小于 0。")

    stop = total_frames if end_frame is None else min(end_frame, total_frames)
    if stop < 0:
        raise ValueError("end_frame 不能小于 0。")
    indices = range(min(start_frame, total_frames), max(min(stop, total_frames), 0), frame_step)
    if max_frames:
        indices = indices[:max_frames]
    return indices


def iter_bin_frames(
    bin_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_step: int = 1,
    max_frames: int | None = None,
    output: OutputMode = "raw",
) -> Iterator[tuple[int, np.ndarray]]:
    """只打开一次文件，按指定间隔逐帧产生 ``(帧号, 图像)``。"""

    with ThermalBinReader(bin_path) as reader:
        indices = select_frame_indices(
            len(reader),
            start_frame=start_frame,
            end_frame=end_frame,
            frame_step=frame_step,
            max_frames=max_frames,
        )
        for frame_index in indices:
            yield frame_index, reader.read_frame(frame_index, output=output)
