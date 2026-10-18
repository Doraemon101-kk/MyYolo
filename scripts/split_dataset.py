"""将图片或图片+YOLO 检测标签复制为可复现的 train/val 数据集。"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

try:
    from .common import check_output, collect_labels, copy_label, list_images, validate_label
except ImportError:
    from common import check_output, collect_labels, copy_label, list_images, validate_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["images", "yolo"], required=True, help="images：纯图片；yolo：图片+检测标签")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", type=Path, help="yolo 模式必填")
    parser.add_argument("--output", required=True, type=Path, help="不存在的新目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例，默认 0.2")
    parser.add_argument("--seed", type=int, default=42, help="固定随机种子，默认 42")
    parser.add_argument("--names", nargs="+", help="按类别 ID 顺序提供名称，例如 armor robot；含空格的名称加引号")
    parser.add_argument("--allow-missing-labels", action="store_true", help="将缺失的标签作为背景图，生成空 txt")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    if not 0 < args.val_ratio < 1:
        raise ValueError("val-ratio 必须在 0 和 1 之间。")
    if args.mode == "yolo" and (args.labels is None or not args.names):
        raise ValueError("yolo 模式必须指定 --labels 和 --names。")
    if args.mode == "images" and (args.labels or args.names or args.allow_missing_labels):
        raise ValueError("images 模式不接受标签、类别或缺失标签选项。")
    if args.names and (len(set(args.names)) != len(args.names) or any(not n.strip() for n in args.names)):
        raise ValueError("类别名不能重复或为空。")
    images = list_images(args.images)
    if len(images) < 2:
        raise ValueError("至少需要 2 张图片才能保证训练集和验证集都非空。")
    check_output(args.output, [args.images] + ([args.labels] if args.labels else []))
    labels = collect_labels(images, args.labels, args.allow_missing_labels) if args.mode == "yolo" else {}
    for label in labels.values():
        if label is not None:
            validate_label(label, len(args.names))
    random.Random(args.seed).shuffle(images)
    val_count = min(len(images) - 1, max(1, round(len(images) * args.val_ratio)))
    splits = {"train": images[val_count:], "val": images[:val_count]}
    print(f"train={len(splits['train'])}, val={val_count}, seed={args.seed}；按图片随机划分，不保证类别分层。")
    if args.dry_run:
        print("检查通过，未写入文件。")
        return
    args.output.mkdir(parents=True)
    with (args.output / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "original_image", "output_image", "original_label"])
        for split, subset in splits.items():
            image_dir = args.output / "images" / split
            image_dir.mkdir(parents=True)
            label_dir = args.output / "labels" / split
            if labels:
                label_dir.mkdir(parents=True)
            for source in subset:
                destination = image_dir / source.name
                shutil.copy2(source, destination)
                if labels:
                    copy_label(labels[source], label_dir / f"{source.stem}.txt")
                writer.writerow([split, str(source.resolve()), str(destination.resolve()),
                                 str(labels[source].resolve()) if labels.get(source) else ""])
    if args.mode == "yolo":
        # JSON 字符串也是合法 YAML 标量，可正确处理中文、空格和 Windows 路径。
        dump = lambda value: json.dumps(value, ensure_ascii=False)
        lines = [f"path: {dump(args.output.resolve().as_posix())}", "train: images/train", "val: images/val", "names:"]
        lines.extend(f"  {i}: {dump(name)}" for i, name in enumerate(args.names))
        (args.output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"划分完成：{args.output.resolve()}；原文件已保留。")


def main() -> None:
    parser = build_parser()
    try:
        run(parser.parse_args())
    except (ValueError, OSError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
