"""验证数据对应关系、可复现划分和参数优先级，不需要下载权重。"""

import csv
import tempfile
import unittest
from pathlib import Path

from scripts import predict_video, rename_images, split_dataset, train_yolo


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "source_images"
        self.labels = self.root / "source_labels"
        self.images.mkdir()
        self.labels.mkdir()
        for index in range(10):
            # 数据整理只应按字节复制，不依赖图片解码。
            (self.images / f"不规则 图 {index}.JPG").write_bytes(f"image-{index}".encode())
            (self.labels / f"不规则 图 {index}.txt").write_text(f"{index % 2} 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    def rename_args(self, output="renamed", extra=()):
        return rename_images.build_parser().parse_args([
            "--images", str(self.images), "--labels", str(self.labels),
            "--output", str(self.root / output), *extra,
        ])

    def split_args(self, output="dataset", mode="yolo", extra=()):
        argv = ["--mode", mode, "--images", str(self.images), "--output", str(self.root / output)]
        if mode == "yolo":
            argv += ["--labels", str(self.labels), "--names", "装甲板", "机器人"]
        return split_dataset.build_parser().parse_args([*argv, *extra])

    def test_rename_keeps_pairs_and_originals(self):
        args = self.rename_args()
        rename_images.run(args)
        with (args.output / "rename_map.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 10)
        self.assertEqual(Path(rows[0]["new_image"]).name, "robot_000001.jpg")
        for row in rows:
            self.assertEqual(Path(row["original_image"]).read_bytes(), Path(row["new_image"]).read_bytes())
            self.assertEqual(Path(row["original_label"]).read_bytes(), Path(row["new_label"]).read_bytes())
            self.assertEqual(Path(row["new_image"]).stem, Path(row["new_label"]).stem)
        with self.assertRaises(ValueError):
            rename_images.run(args)

    def test_dry_run_creates_nothing(self):
        args = self.rename_args(extra=["--dry-run"])
        rename_images.run(args)
        self.assertFalse(args.output.exists())
        args = self.split_args(extra=["--dry-run"])
        split_dataset.run(args)
        self.assertFalse(args.output.exists())

    def test_split_is_disjoint_complete_and_reproducible(self):
        import yaml

        first, second = self.split_args("first"), self.split_args("second")
        split_dataset.run(first)
        split_dataset.run(second)
        subsets = {}
        for split, expected_count in [("train", 8), ("val", 2)]:
            paths = list((first.output / "images" / split).iterdir())
            subsets[split] = {p.name for p in paths}
            self.assertEqual(len(paths), expected_count)
            self.assertEqual(subsets[split], {p.name for p in (second.output / "images" / split).iterdir()})
            for path in paths:
                self.assertEqual(path.read_bytes(), (self.images / path.name).read_bytes())
                label = first.output / "labels" / split / f"{path.stem}.txt"
                self.assertEqual(label.read_bytes(), (self.labels / label.name).read_bytes())
        self.assertFalse(subsets["train"] & subsets["val"])
        self.assertEqual(subsets["train"] | subsets["val"], {p.name for p in self.images.iterdir()})
        data = yaml.safe_load((first.output / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["names"], {0: "装甲板", 1: "机器人"})
        self.assertEqual(Path(data["path"]), first.output.resolve())

    def test_images_only_has_no_labels_or_training_yaml(self):
        args = self.split_args(mode="images")
        split_dataset.run(args)
        self.assertFalse((args.output / "labels").exists())
        self.assertFalse((args.output / "data.yaml").exists())
        self.assertEqual(len(list((args.output / "images" / "val").iterdir())), 2)

    def test_missing_labels_require_explicit_background_option(self):
        missing = next(self.labels.iterdir())
        missing.unlink()
        args = self.split_args()
        with self.assertRaises(ValueError):
            split_dataset.run(args)
        self.assertFalse(args.output.exists())
        args = self.split_args(extra=["--allow-missing-labels"])
        split_dataset.run(args)
        copied = list((args.output / "labels").rglob(missing.name))
        self.assertEqual(len(copied), 1)
        self.assertEqual(copied[0].read_bytes(), b"")

    def test_bad_labels_fail_before_writing(self):
        label = next(self.labels.iterdir())
        for contents in ["2 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2", "0 1 2"]:
            with self.subTest(contents=contents):
                label.write_text(contents)
                args = self.split_args()
                with self.assertRaises(ValueError):
                    split_dataset.run(args)
                self.assertFalse(args.output.exists())

    def test_orphan_label_is_rejected(self):
        (self.labels / "orphan.txt").write_text("")
        with self.assertRaises(ValueError):
            split_dataset.run(self.split_args())

    def test_duplicate_stems_are_rejected(self):
        (self.images / "不规则 图 0.png").write_bytes(b"different")
        with self.assertRaises(ValueError):
            rename_images.run(self.rename_args())

    def test_nested_output_is_rejected(self):
        args = self.rename_args()
        args.output = self.images / "nested"
        with self.assertRaises(ValueError):
            rename_images.run(args)

    def test_tiny_ratio_still_keeps_nonempty_validation(self):
        args = self.split_args(extra=["--val-ratio", "0.001"])
        split_dataset.run(args)
        self.assertEqual(len(list((args.output / "images" / "val").iterdir())), 1)

    def test_train_config_precedence_and_typed_overrides(self):
        data = self.root / "data.yaml"
        data.write_text("names: [robot]\n")
        config = self.root / "train.yaml"
        config.write_text("model: yolov8n.pt\nepochs: 30\nbatch: 4\nlr0: 0.02\n")
        args = train_yolo.build_parser().parse_args([
            "--config", str(config), "--data", str(data), "--epochs", "2", "--lr0", "0.01",
            "--set", "lr0=0.001", "--set", "cos_lr=true", "--no-amp",
        ])
        model, options = train_yolo.resolve_config(args)
        self.assertEqual(model, "yolov8n.pt")
        self.assertEqual(options["epochs"], 2)
        self.assertEqual(options["batch"], 4)
        self.assertEqual(options["lr0"], 0.001)
        self.assertIs(options["cos_lr"], True)
        self.assertIs(options["amp"], False)

    def test_resume_does_not_override_saved_parameters(self):
        checkpoint = self.root / "last.pt"
        checkpoint.write_bytes(b"test")
        parser = train_yolo.build_parser()
        args = parser.parse_args(["--resume", str(checkpoint), "--device", "cpu"])
        self.assertEqual(train_yolo.resolve_config(args), (str(checkpoint), {"resume": True, "device": "cpu"}))
        with self.assertRaises(ValueError):
            train_yolo.resolve_config(parser.parse_args(["--resume", str(checkpoint), "--epochs", "200"]))

    def test_video_rejects_overwrite_and_invalid_threshold(self):
        weights, source = self.root / "best.pt", self.root / "input.mp4"
        weights.write_bytes(b"test")
        source.write_bytes(b"test")
        parser = predict_video.build_parser()
        args = parser.parse_args(["--weights", str(weights), "--source", str(source), "--output", str(source)])
        with self.assertRaises(ValueError):
            predict_video.validate_args(args)
        args.output = self.root / "output.mp4"
        args.conf = float("nan")
        with self.assertRaises(ValueError):
            predict_video.validate_args(args)


if __name__ == "__main__":
    unittest.main()
