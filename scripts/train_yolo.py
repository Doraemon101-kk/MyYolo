"""训练 YOLO 检测模型：YAML 配置、常用参数、自定义超参数与断点续训。"""

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="训练参数 YAML；命令行参数优先")
    parser.add_argument("--model", help="预训练权重或模型 YAML，默认 yolo11n.pt")
    parser.add_argument("--data", help="数据集 data.yaml")
    parser.add_argument("--resume", type=Path, help="从中断训练的 last.pt 恢复；只允许额外指定 --device")
    parser.add_argument("--dry-run", action="store_true", help="打印最终配置，不加载模型或训练")
    parser.add_argument("--device", help="cpu、0 或 0,1；省略时由框架选择")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int, help="输入边长，通常取 32 的倍数")
    parser.add_argument("--batch", type=int, help="批大小；-1 表示单 GPU 自动估算")
    parser.add_argument("--workers", type=int, help="Windows 新手建议设为 0")
    parser.add_argument("--patience", type=int, help="早停等待轮数，0 表示禁用")
    parser.add_argument("--optimizer", choices=["auto", "SGD", "Adam", "AdamW", "RMSProp", "NAdam", "RAdam"])
    for key in ("lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "close_mosaic"):
        parser.add_argument("--" + key.replace("_", "-"), dest=key, type=int if key == "close_mosaic" else float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--project", help="实验输出根目录")
    parser.add_argument("--name", help="实验名称")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="--amp 开启 / --no-amp 关闭混合精度")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="其他 Ultralytics 参数，可重复；如 --set mosaic=0.5 --set cos_lr=true")
    return parser


def resolve_config(args: argparse.Namespace) -> tuple[str, dict]:
    import yaml

    regular_keys = [key for key in vars(args) if key not in {"config", "model", "resume", "dry_run", "overrides"}]
    if args.resume:
        if args.config or args.model or args.overrides or any(
            getattr(args, key) is not None for key in regular_keys if key != "device"
        ):
            raise ValueError("断点续训恢复原实验参数，只允许 --resume、--device 和 --dry-run；微调请用 --model。")
        if not args.resume.is_file():
            raise ValueError(f"续训权重不存在：{args.resume}")
        options = {"resume": True}
        if args.device is not None:
            options["device"] = args.device
        return str(args.resume), options

    options = {"epochs": 100, "imgsz": 640, "batch": 8, "workers": 0,
               "seed": 42, "project": "runs/detect", "name": "robot_baseline"}
    config = {}
    if args.config:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or not all(isinstance(key, str) for key in config):
            raise ValueError("配置文件顶层必须是参数名与参数值组成的字典。")
    model = args.model or config.pop("model", "yolo11n.pt")
    config.pop("model", None)
    options.update(config)
    for key in regular_keys:
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    for override in args.overrides:
        key, separator, value = override.partition("=")
        key = key.strip()
        if not separator or not key or not value.strip():
            raise ValueError(f"自定义参数格式应为 KEY=VALUE：{override}")
        if key in {"model", "data", "resume", "task", "mode", "cfg"}:
            raise ValueError(f"请使用对应脚本选项设置 {key}；本脚本仅支持检测训练。")
        options[key] = yaml.safe_load(value)
    if any(key in options for key in ("resume", "task", "mode", "cfg")):
        raise ValueError("配置中不要设置 resume/task/mode/cfg；续训使用 --resume，本脚本用于检测训练。")
    if not isinstance(model, str) or not model:
        raise ValueError("model 必须是权重名或模型配置文件路径。")
    if not isinstance(options.get("data"), str) or not Path(options["data"]).is_file():
        raise ValueError("请通过 --data 或配置文件指定存在的 data.yaml。")
    options["data"] = str(Path(options["data"]).resolve())
    return model, options


def run(args: argparse.Namespace) -> None:
    import yaml

    model_path, options = resolve_config(args)
    print("最终训练配置（相对路径以当前工作目录为基准）：")
    print(yaml.safe_dump({"model": model_path, **options}, allow_unicode=True, sort_keys=False))
    if args.dry_run:
        print("预览完成，未加载模型或启动训练；参数合法性还会由 Ultralytics 检查。")
        return
    from ultralytics import YOLO

    model = YOLO(model_path, task="detect")
    if model.task != "detect":
        raise ValueError("请使用目标检测权重，本脚本不接受分割、分类或姿态权重。")
    model.train(**options)
    print(f"训练结束，实验目录：{model.trainer.save_dir}")
    print("weights/best.pt 用于部署与验证，weights/last.pt 用于恢复中断的训练。")


def main() -> None:
    parser = build_parser()
    try:
        run(parser.parse_args())
    except (ValueError, OSError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    # Windows 多进程加载数据必须有入口保护。
    main()
