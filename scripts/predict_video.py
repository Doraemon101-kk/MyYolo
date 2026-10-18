"""逐帧读取视频并用 YOLO 检测，保存标注视频，可选实时显示。"""

import argparse
import math
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="本地检测权重，如 best.pt")
    parser.add_argument("--source", required=True, type=Path, help="本地输入视频")
    parser.add_argument("--output", type=Path, default=Path("runs/video/result.mp4"), help="新的输出 mp4 文件，拒绝覆盖")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值，默认 0.7")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", help="cpu、0 等")
    parser.add_argument("--classes", type=int, nargs="+", help="只显示指定类别 ID")
    parser.add_argument("--max-det", type=int, default=300, help="单帧最大检测数")
    parser.add_argument("--half", action="store_true", help="在支持的 GPU 上使用 FP16")
    parser.add_argument("--show", action="store_true", help="显示窗口；按 Q 或 Esc 提前结束")
    parser.add_argument("--max-frames", type=int, help="最多处理帧数，适合快速检查")
    parser.add_argument("--fallback-fps", type=float, default=25.0, help="仅在无法读取原视频帧率时使用")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.weights, args.source):
        if not path.is_file():
            raise ValueError(f"文件不存在：{path}")
    if args.output.resolve() in {args.source.resolve(), args.weights.resolve()} or args.output.exists():
        raise ValueError("输出不能覆盖输入、权重或任何已存在文件，请换一个输出路径。")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("输出文件扩展名必须为 .mp4。")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("conf 和 iou 必须在 [0, 1] 内。")
    if args.imgsz <= 0 or args.max_det <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        raise ValueError("imgsz、max-det 和 max-frames 必须为正整数。")
    if not math.isfinite(args.fallback_fps) or args.fallback_fps <= 0:
        raise ValueError("fallback-fps 必须是有限正数。")
    if args.classes is not None and any(c < 0 for c in args.classes):
        raise ValueError("类别 ID 必须非负。")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    if args.half and (args.device == "cpu" or not torch.cuda.is_available()):
        raise ValueError("--half 需要支持 CUDA 的 GPU；CPU 请省略该选项。")
    model = YOLO(str(args.weights), task="detect")
    if model.task != "detect":
        raise ValueError("请使用 YOLO 目标检测权重。")
    if args.classes is not None and any(c not in model.names for c in args.classes):
        raise ValueError(f"类别 ID 超出模型范围：{model.names}")
    # 新版 Ultralytics 用 quantize 指定精度，旧版仍使用 half。
    precision = {}
    if args.half:
        precision = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
    capture = cv2.VideoCapture(str(args.source))
    writer = None
    processed = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"无法打开视频：{args.source}，请检查文件与解码器。")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = args.fallback_fps
            print(f"无法读取帧率，输出使用 {fps:g} FPS。")
        start = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(
                source=frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                device=args.device, classes=args.classes,
                max_det=args.max_det, verbose=False, **precision,
            )[0]
            annotated = result.plot()
            if writer is None:
                height, width = annotated.shape[:2]
                args.output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not writer.isOpened():
                    raise ValueError("无法创建输出视频，请检查 mp4v 编码器及输出路径。")
            writer.write(annotated)
            processed += 1
            if processed % 100 == 0:
                print(f"已处理 {processed} 帧，平均处理速度 {processed / (time.perf_counter() - start):.1f} 帧/秒")
            if args.show:
                cv2.imshow("YOLO - Q / Esc to quit", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            if args.max_frames is not None and processed >= args.max_frames:
                break
        if processed == 0:
            raise ValueError("视频没有可解码的帧。")
        elapsed = time.perf_counter() - start
        print(f"完成：{processed} 帧，处理速度 {processed / elapsed:.1f} 帧/秒；输出播放帧率 {fps:g} FPS。")
        print(f"结果：{args.output.resolve()}（不保留音轨）")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


def main() -> None:
    parser = build_parser()
    try:
        run(parser.parse_args())
    except (ValueError, OSError) as exc:
        parser.exit(1, f"错误：{exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\n已停止；已写入的帧会保留在输出文件中。\n")


if __name__ == "__main__":
    main()
