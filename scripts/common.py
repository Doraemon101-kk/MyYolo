"""数据脚本共用的文件与 YOLO 检测标签检查；仅依赖 Python 标准库。"""

import math
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"图片目录不存在：{directory}")
    images = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: (p.name.casefold(), p.name),
    )
    if not images:
        raise ValueError(f"目录中没有支持的图片：{directory}（不递归扫描子目录）")
    stems = [p.stem.casefold() for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError("图片存在重复主文件名（如 a.jpg 与 a.png），无法唯一匹配标签，请先整理。")
    return images


def collect_labels(images: list[Path], directory: Path, allow_missing: bool = False) -> dict[Path, Path | None]:
    if not directory.is_dir():
        raise ValueError(f"标签目录不存在：{directory}")
    labels: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() == ".txt":
            key = path.stem.casefold()
            if key in labels:
                raise ValueError(f"重复标签：{path.name}")
            labels[key] = path
    image_stems = {p.stem.casefold() for p in images}
    orphans = set(labels) - image_stems
    if orphans:
        raise ValueError(f"发现没有对应图片的标签：{', '.join(sorted(orphans)[:5])}；标签目录只应放 YOLO 标注。")
    missing = [p.name for p in images if p.stem.casefold() not in labels]
    if missing and not allow_missing:
        raise ValueError(f"有 {len(missing)} 张图片缺少标签，例如 {missing[:5]}。确认是背景图后才使用 --allow-missing-labels。")
    return {p: labels.get(p.stem.casefold()) for p in images}


def validate_label(path: Path, num_classes: int) -> None:
    """只接受检测任务的 class x_center y_center width height 格式。空文件是背景图。"""
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        location = f"{path}:{line_no}"
        if len(parts) != 5:
            raise ValueError(f"{location}：检测标签每行必须有 5 列。")
        try:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{location}：类别须为整数，坐标须为数字。") from exc
        if not 0 <= class_id < num_classes:
            raise ValueError(f"{location}：类别 {class_id} 超出 0..{num_classes - 1}。")
        if not all(math.isfinite(v) for v in (x, y, w, h)) or not (
            0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
        ):
            raise ValueError(f"{location}：坐标必须归一化到 [0, 1]，宽高必须大于 0。")


def check_output(output: Path, sources: list[Path]) -> None:
    resolved = output.resolve()
    if output.exists():
        raise ValueError(f"输出路径已存在，拒绝覆盖：{output}；请换一个新目录。")
    for source in sources:
        source = source.resolve()
        if resolved == source or source in resolved.parents or resolved in source.parents:
            raise ValueError("输出目录与输入目录不能互相包含，请使用独立的同级目录。")


def copy_label(source: Path | None, destination: Path) -> None:
    if source is None:
        destination.write_text("", encoding="utf-8")
    else:
        shutil.copy2(source, destination)
