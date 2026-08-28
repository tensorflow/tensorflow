"""Times the same work on the GPU and on the CPU.

The point of a Metal backend on Apple silicon is not that it can run an op but
that it is worth running there, so the number that matters is the ratio, and
the honest way to get it is to run the identical graph on both devices in the
same process with the same data.

Every measurement waits for the device before stopping the clock, since a
Metal command buffer is asynchronous and an unwaited timing measures how fast
work can be enqueued.
"""

import argparse
import os
import statistics
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import load_library


def sync(tensor):
  """Forces the work to have happened before the clock stops."""
  if isinstance(tensor, (list, tuple)):
    for item in tensor:
      sync(item)
    return
  # Reading one element brings the device to a stop for this stream.
  np.asarray(tensor).reshape(-1)[:1]


def timed(fn, device, warmup=3, runs=10):
  with tf.device(device):
    for _ in range(warmup):
      sync(fn())
    samples = []
    for _ in range(runs):
      start = time.perf_counter()
      sync(fn())
      samples.append(time.perf_counter() - start)
  return statistics.median(samples)


def build_model():
  return tf.keras.Sequential([
      tf.keras.layers.Input((32, 32, 3)),
      tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dense(10),
  ])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--plugin", default=None)
  args = parser.parse_args()
  if args.plugin:
    load_library.load_pluggable_device_library(args.plugin)
  print(f"tensorflow {tf.__version__}")
  devices = [d.name for d in tf.config.list_physical_devices()]
  print(f"devices {devices}")
  if not any("GPU" in d for d in devices):
    print("no GPU device, nothing to compare")
    return 1
  tf.config.set_soft_device_placement(True)

  rng = np.random.default_rng(0)
  cases = []

  for n in (512, 1024, 2048):
    a = tf.constant(rng.standard_normal((n, n), dtype=np.float32))
    cases.append((f"MatMul {n}x{n}", lambda a=a: tf.matmul(a, a)))

  for batch in (16, 64):
    image = tf.constant(rng.standard_normal((batch, 64, 64, 32),
                                            dtype=np.float32))
    filt = tf.constant(rng.standard_normal((3, 3, 32, 64), dtype=np.float32))
    cases.append((f"Conv2D batch {batch}",
                  lambda i=image, f=filt: tf.nn.conv2d(i, f, 1, "SAME")))

  big = tf.constant(rng.standard_normal((4096, 4096), dtype=np.float32))
  cases.append(("Elementwise 4096x4096", lambda: tf.nn.relu(big * 2.0 + 1.0)))
  cases.append(("ReduceSum 4096x4096", lambda: tf.reduce_sum(big, axis=1)))

  model = build_model()
  model.build((None, 32, 32, 3))
  for batch in (32, 128):
    x = tf.constant(rng.standard_normal((batch, 32, 32, 3), dtype=np.float32))
    cases.append((f"CNN forward batch {batch}",
                  lambda x=x, m=model: m(x, training=False)))

  # A training step with plain SGD, which needs no fused optimiser.
  y = tf.constant(rng.integers(0, 10, 128).astype(np.int32))
  x = tf.constant(rng.standard_normal((128, 32, 32, 3), dtype=np.float32))
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  optimiser = tf.keras.optimizers.SGD(0.01)

  def train_step():
    with tf.GradientTape() as tape:
      loss = loss_fn(y, model(x, training=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimiser.apply_gradients(zip(grads, model.trainable_variables))
    return loss

  cases.append(("CNN train step, SGD, batch 128", train_step))

  print(f"\n{'case':34s} {'GPU ms':>10s} {'CPU ms':>10s} {'speedup':>9s}")
  print("-" * 68)
  for name, fn in cases:
    try:
      gpu = timed(fn, "/GPU:0")
      cpu = timed(fn, "/CPU:0")
    except Exception as error:  # pylint: disable=broad-except
      print(f"{name:34s} {str(error).splitlines()[0][:28]}")
      continue
    ratio = cpu / gpu if gpu > 0 else float("inf")
    print(f"{name:34s} {gpu*1e3:10.2f} {cpu*1e3:10.2f} {ratio:8.2f}x")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
