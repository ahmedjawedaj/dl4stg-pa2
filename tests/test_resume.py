"""An interrupted run must resume with the same windows, optimizer and final weights."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from pa2 import train


class TinyForecaster(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        y = self.output(x[:, -1]).sigmoid()
        return y[:, None].expand(-1, 5, -1, -1, -1)


class Interrupted(Exception):
    pass


class ResumeTests(unittest.TestCase):
    def test_interrupted_training_matches_uninterrupted(self):
        data = torch.randint(0, 256, (5, 20, 1, 4, 4), dtype=torch.uint8,
                             generator=torch.Generator().manual_seed(17))
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(train, "load_moving_mnist", return_value=data))
            stack.enter_context(patch.object(train, "split", return_value=(data[:4], data[4:], None)))
            stack.enter_context(patch.object(train, "default_device", return_value=torch.device("cpu")))
            stack.enter_context(patch.object(train, "build", side_effect=lambda _: TinyForecaster()))
            stack.enter_context(patch.object(train, "BATCH", 2))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            direct = Path(folder) / "direct"
            resumed = Path(folder) / "resumed"

            def run(out, seed=0):
                args = ["train", "--model", "A", "--seed", str(seed), "--epochs", "2",
                        "--tag", "audit", "--out", str(out)]
                with patch("sys.argv", args):
                    train.main()

            run(direct)
            original_save = train.save_json_atomic

            def interrupt_after_epoch(value, path):
                original_save(value, path)
                raise Interrupted()

            with patch.object(train, "save_json_atomic", side_effect=interrupt_after_epoch):
                with self.assertRaises(Interrupted):
                    run(resumed)
            self.assertTrue((resumed / "audit.resume.pt").exists())
            self.assertFalse((resumed / "audit.pt").exists())
            with self.assertRaisesRegex(ValueError, "changed seed"):
                run(resumed, seed=1)
            run(resumed)
            self.assertFalse((resumed / "audit.resume.pt").exists())
            a = torch.load(direct / "audit.pt", weights_only=True)["state_dict"]
            b = torch.load(resumed / "audit.pt", weights_only=True)["state_dict"]
            for name in a:
                self.assertTrue(torch.equal(a[name], b[name]), name)
            log_a = json.loads((direct / "audit.json").read_text())
            log_b = json.loads((resumed / "audit.json").read_text())
            self.assertEqual([e["epoch"] for e in log_b["epochs"]], [1, 2])
            for ea, eb in zip(log_a["epochs"], log_b["epochs"]):
                self.assertEqual(ea["train_loss"], eb["train_loss"])
                self.assertEqual(ea["val_E5_1x"], eb["val_E5_1x"])


if __name__ == "__main__":
    unittest.main()
