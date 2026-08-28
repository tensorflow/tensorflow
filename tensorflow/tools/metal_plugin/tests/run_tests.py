"""On-device checks for the plugin, against the TensorFlow it loads into.

Every numeric check compares a GPU result to the CPU kernel for the same op
with soft placement off, so an op with no GPU kernel raises rather than
quietly producing the right answer on the wrong device. One check exists only
to prove that soft placement really is off, because without it every other
check could be passing on the CPU.

Deliberately a plain script rather than a pytest suite: TensorFlow holds a
device open for the life of the process, and a test harness that tears down
between cases has nowhere useful to put that.
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import load_library

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN = os.environ.get(
    "METAL_PLUGIN",
    os.path.join(os.path.dirname(HERE), "build", "libmetal_plugin.dylib"))

_failures = 0


def check(name, condition, detail=""):
  global _failures
  print(f"  {name:38s} {'ok' if condition else 'FAILED'}"
        f"{('  ' + detail) if detail else ''}")
  if not condition:
    _failures += 1


def close(name, got, want, rtol=1e-4, atol=2e-4):
  worst = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
  check(name, np.allclose(got, want, rtol=rtol, atol=atol),
        f"max diff {worst:.2e}")


def main():
  print(f"tensorflow {tf.__version__}")
  before = [d.name for d in tf.config.list_physical_devices("GPU")]
  load_library.load_pluggable_device_library(PLUGIN)
  after = [d.name for d in tf.config.list_physical_devices("GPU")]
  print(f"GPU devices before {before}, after {after}")
  check("the plugin adds a GPU device", after == ["/physical_device:GPU:0"])
  if not after:
    print("\nno device, nothing further to check")
    return 1

  tf.config.set_soft_device_placement(False)

  rng = np.random.default_rng(0)
  a = rng.standard_normal((64, 48), dtype=np.float32)
  x = rng.standard_normal((2, 16, 16, 3), dtype=np.float32)
  k = rng.standard_normal((3, 3, 3, 8), dtype=np.float32)

  cases = {
      "MatMul": lambda: tf.matmul(tf.constant(a), tf.constant(a.T)),
      "Softmax": lambda: tf.nn.softmax(tf.constant(a)),
      "Conv2D": lambda: tf.nn.conv2d(x, k, strides=1, padding="SAME"),
      "Relu": lambda: tf.nn.relu(tf.constant(a)),
      "MaxPool2D": lambda: tf.nn.max_pool2d(x, 2, 2, "VALID"),
      "ReduceSum": lambda: tf.reduce_sum(tf.constant(a), axis=1),
      "Transpose": lambda: tf.transpose(tf.constant(a)),
      "BiasAdd": lambda: tf.nn.bias_add(tf.constant(a),
                                        tf.constant(a[0].copy())),
  }
  print("\nGPU results against the CPU kernel for the same op:")
  for name, build in cases.items():
    with tf.device("/GPU:0"):
      got = build()
    if not got.device.endswith("GPU:0"):
      check(name, False, f"placed on {got.device}")
      continue
    with tf.device("/CPU:0"):
      want = build()
    close(name, got.numpy(), want.numpy())

  print("\ncontrols:")
  # Without this the checks above prove nothing: they would all pass on the
  # CPU if soft placement were quietly moving them there.
  try:
    with tf.device("/GPU:0"):
      tf.raw_ops.MatrixDeterminant(input=tf.eye(4))
    check("an op with no GPU kernel is refused", False,
          "it ran, so soft placement is on")
  except Exception:  # pylint: disable=broad-except
    check("an op with no GPU kernel is refused", True)

  data = rng.standard_normal((257, 33), dtype=np.float32)
  with tf.device("/GPU:0"):
    on_device = tf.constant(data)
  check("host round trip is exact",
        np.array_equal(on_device.numpy(), data))

  print(f"\n{'all checks passed' if _failures == 0 else str(_failures) + ' FAILED'}")
  return 0 if _failures == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
