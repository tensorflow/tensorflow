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

  # skip_input and a recurrent projection change the parameter buffer's
  # layout, so a mismatch between what the size op reports and what the graph
  # reads shows up here as a wrong size rather than as a wrong answer.
  def params_size(num_layers=1, **kwargs):
    return int(tf.raw_ops.CudnnRNNParamsSize(
        num_layers=num_layers, num_units=4, input_size=4, T=tf.float32,
        S=tf.int32, rnn_mode="lstm", **kwargs).numpy())

  check("skip_input drops the input matrices",
        params_size(input_mode="linear_input") -
        params_size(input_mode="skip_input") == 4 * 4 * 4)
  # Recurrent matrices narrow from 4 to 2 columns, and a 2x4 projection joins
  # them.
  check("a projection resizes the parameter buffer",
        params_size(num_proj=2) ==
        params_size() - 4 * 4 * 2 + 2 * 4)

  with tf.device("/GPU:0"):
    projected = tf.raw_ops.CudnnRNNV3(
        input=tf.zeros([3, 2, 4]), input_h=tf.zeros([1, 2, 2]),
        input_c=tf.zeros([1, 2, 4]),
        params=tf.zeros([params_size(num_proj=2)]),
        sequence_lengths=tf.constant([3, 3], dtype=tf.int32),
        rnn_mode="lstm", num_proj=2, is_training=True)
  check("a projection narrows the hidden state but not the cell",
        projected.output.shape[2] == 2 and projected.output_h.shape[2] == 2
        and projected.output_c.shape[2] == 4,
        f"y {projected.output.shape} h {projected.output_h.shape} "
        f"c {projected.output_c.shape}")

  # Dropout has to change the answer, has to leave the draws behind for the
  # gradient, and has to do nothing at all outside training. Two layers,
  # because dropout applies at the boundary between them and a single layer
  # has none.
  weights = tf.constant(rng.standard_normal(params_size(num_layers=2)) * 0.3,
                        dtype=tf.float32)
  series = tf.constant(rng.standard_normal((3, 2, 4)), dtype=tf.float32)

  def recurrent(dropout, training):
    with tf.device("/GPU:0"):
      return tf.raw_ops.CudnnRNNV3(
          input=series, input_h=tf.zeros([2, 2, 4]),
          input_c=tf.zeros([2, 2, 4]), params=weights,
          sequence_lengths=tf.constant([3, 3], dtype=tf.int32),
          rnn_mode="lstm", dropout=dropout, seed=7, seed2=0,
          is_training=training)

  dropped = recurrent(0.5, True)
  check("dropout leaves its draws in the reserve space",
        int(dropped.reserve_space.shape[0]) == 3 * 2 * 4,
        f"reserve {int(dropped.reserve_space.shape[0])}")
  check("no dropout leaves the reserve space empty",
        int(recurrent(0.0, True).reserve_space.shape[0]) == 0)
  check("dropout changes the output",
        not np.allclose(dropped.output.numpy(),
                        recurrent(0.0, True).output.numpy()))
  check("dropout is inactive outside training",
        np.allclose(recurrent(0.5, False).output.numpy(),
                    recurrent(0.0, False).output.numpy()))

  # RandomUniform promises [0, 1). In float16 that is not free: a float draw
  # above 1 - 2^-12 rounds up to exactly one, and a caller computing
  # log(1 - u) would get -inf from a generator that can return one.
  with tf.device("/GPU:0"):
    draws = tf.raw_ops.RandomUniform(shape=[20000], dtype=tf.float16,
                                     seed=1, seed2=2).numpy()
  check("RandomUniform float16 stays below one",
        bool(draws.max() < 1.0 and draws.min() >= 0.0),
        f"range [{draws.min():.4f}, {draws.max():.4f}]")

  # The float16 max-pooling gradient scatters through an atomic, which Metal
  # has only for float, so it accumulates into a float32 temporary and is
  # narrowed afterwards. Overlapping windows are what make that path matter:
  # without them no two gradients ever land on the same element.
  pooled = rng.standard_normal((1, 4, 4, 2)).astype(np.float16)
  image = rng.standard_normal((1, 8, 8, 2)).astype(np.float16)
  pool_args = dict(ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
  with tf.device("/GPU:0"):
    _, argmax = tf.raw_ops.MaxPoolWithArgmax(
        input=tf.constant(image), Targmax=tf.int64,
        include_batch_in_index=True, **pool_args)
    scattered = tf.raw_ops.MaxPoolGradWithArgmax(
        input=tf.constant(image), grad=tf.constant(pooled), argmax=argmax,
        include_batch_in_index=True, **pool_args).numpy()
  expected = np.zeros(image.size, dtype=np.float64)
  for value, index in zip(pooled.reshape(-1).astype(np.float64),
                          argmax.numpy().reshape(-1)):
    expected[index] += value
  close("MaxPoolGradWithArgmax float16", scattered.reshape(-1), expected,
        rtol=2e-2, atol=2e-2)

  # float16 with an odd column count gives a row stride MPSMatrix will not
  # accept, so these shapes take a different path through the kernel than the
  # even ones and are the reason to check both.
  for m, kk, n in [(3, 5, 7), (4, 6, 8)]:
    lhs = rng.standard_normal((m, kk)).astype(np.float16)
    rhs = rng.standard_normal((kk, n)).astype(np.float16)
    with tf.device("/GPU:0"):
      got = tf.matmul(tf.constant(lhs), tf.constant(rhs))
    with tf.device("/CPU:0"):
      want = tf.matmul(tf.constant(lhs), tf.constant(rhs))
    close(f"MatMul float16 {m}x{kk}x{n}", got.numpy().astype(np.float32),
          want.numpy().astype(np.float32), rtol=2e-2, atol=2e-2)

  # CheckNumerics is the identity plus a promise to fail, and a version of it
  # that only forwarded would pass every value check above while breaking the
  # one thing the op exists for.
  inf, nan = float("inf"), float("nan")
  numerics = [
      ("CheckNumerics passes finite values", [1.0, 2.0], False, None),
      ("CheckNumerics reports a NaN", [1.0, nan], False, "Tensor had NaN"),
      ("CheckNumerics reports an Inf", [1.0, inf], False, "Tensor had Inf"),
      ("CheckNumericsV2 names the sign", [1.0, -inf], True, "Tensor had -Inf"),
  ]
  for name, values, v2, want in numerics:
    op = tf.raw_ops.CheckNumericsV2 if v2 else tf.raw_ops.CheckNumerics
    try:
      with tf.device("/GPU:0"):
        op(tensor=tf.constant(values, dtype=tf.float32), message="check")
      check(name, want is None, "" if want is None else "it did not raise")
    except tf.errors.InvalidArgumentError as error:
      matched = want is not None and want in str(error)
      check(name, matched, "" if matched else str(error)[:70])

  data = rng.standard_normal((257, 33), dtype=np.float32)
  with tf.device("/GPU:0"):
    on_device = tf.constant(data)
  check("host round trip is exact",
        np.array_equal(on_device.numpy(), data))

  print(f"\n{'all checks passed' if _failures == 0 else str(_failures) + ' FAILED'}")
  return 0 if _failures == 0 else 1


if __name__ == "__main__":
  sys.exit(main())
