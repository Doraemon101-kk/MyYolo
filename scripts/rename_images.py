"""按稳定顺序复制并重命名图片，可同步重命名同名 YOLO 标签。"""

import argparse
import csv
import shutil
from pathlib import Path

try:
    from .common import check_output, collect_labels, copy_label, list_images
except ImportError:
    from common import check_output, collect_labels, copy_label, list_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="输入图片目录（只扫描当前层）")
    parser.add_argument("--labels", type=Path, help="可选：同名 YOLO txt 标签目录")
    parser.add_argument("--output", required=True, type=Path, help="不存在的输出目录；保留原文件")
    parser.add_argument("--prefix", default="robot_", help="文件名前缀，默认 robot_")
    parser.add_argument("--start", type=int, default=1, help="编号起点，默认 1")
    parser.add_argument("--digits", type=int, default=6, help="编号最小位数，默认 6")
    parser.add_argument("--allow-missing-labels", action="store_true", help="明确将缺失标签的图片当背景图，生成空 txt")
    parser.add_argument("--dry-run", action="store_true", help="只检查和预览，不写文件")
    return parser


def run(args: argparse.Namespace) -> None:
    if args.start < 0 or args.digits < 1:
        raise ValueError("start 必须非负，digits 必须大于 0。")
    if any(c in args.prefix for c in '<>:"/\\|?*') or any(ord(c) < 32 for c in args.prefix):
        raise ValueError("prefix 包含不能用于文件名的字符。")
    if args.allow_missing_labels and args.labels is None:
        raise ValueError("--allow-missing-labels 需要同时指定 --labels。")
    images = list_images(args.images)
    labels = collect_labels(images, args.labels, args.allow_missing_labels) if args.labels else {}
    check_output(args.output, [args.images] + ([args.labels] if args.labels else []))
    plan = [(p, f"{args.prefix}{i:0{args.digits}d}{p.suffix.lower()}") for i, p in enumerate(images, args.start)]
    for source, name in plan[:10]:
        print(f"{source.name} -> {name}")
    print(f"共 {len(plan)} 张图片；输出：{args.output.resolve()}")
    if args.dry_run:
        print("预览完成，未写入文件。")
        return
    (args.output / "images").mkdir(parents=True)
    if args.labels:
        (args.output / "labels").mkdir()
    with (args.output / "rename_map.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_image", "new_image", "original_label", "new_label"])
        for source, name in plan:
            destination = args.output / "images" / name
            shutil.copy2(source, destination)
            new_label = args.output / "labels" / f"{destination.stem}.txt" if args.labels else None
            if new_label:
                copy_label(labels[source], new_label)
            writer.writerow([str(source.resolve()), str(destination.resolve()),
                             str(labels[source].resolve()) if labels.get(source) else "",
                             str(new_label.resolve()) if new_label else ""])
    print("完成，原文件已保留；名称映射见 rename_map.csv。")


def main() -> None:
    parser = build_parser()
    try:
        run(parser.parse_args())
    except (ValueError, OSError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
