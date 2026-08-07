"""End-to-end smoke test for the KDNN integration.

Run on a Kunpeng 920 host with --define=enable_kdnn=true to verify that
tf.math.sigmoid in a real GraphDef gets rewritten to _KdnnSigmoid by the
Grappler remapper. Documented as Step 7 of
third_party/KDNN/HARDWARE_VERIFICATION.md.

Exits 0 if the rewrite fires (i.e. _KdnnSigmoid appears in the optimized
graph) and the rewritten op executes without numerical drift greater than
the remapper's internal tolerance. Exits 1 otherwise.

Usage:
    TF_ENABLE_KDNN_OPTS=1 python3 third_party/KDNN/kdnn_e2e_smoke.py
"""

import os
import sys

import numpy as np


def main():
  if os.environ.get("TF_ENABLE_KDNN_OPTS") != "1":
    print(
        "ERROR: TF_ENABLE_KDNN_OPTS=1 is required to enable the KDNN "
        "rewrite path. Re-run with that env var set.",
        file=sys.stderr,
    )
    return 1

  import tensorflow as tf  # noqa: E402  (import after env var check)
  from tensorflow.core.protobuf import config_pb2  # noqa: E402

  # Build a small graph: y = sigmoid(W @ x + b).
  x = tf.compat.v1.placeholder(tf.float32, shape=(4,), name="x")
  w = tf.constant(np.random.RandomState(0).randn(4, 4).astype(np.float32))
  b = tf.constant(np.random.RandomState(1).randn(4).astype(np.float32))
  y = tf.nn.sigmoid(tf.linalg.matvec(w, x) + b, name="y")

  # Run Grappler with the optimizer turned on at L1. We capture the
  # post-optimization graph via RunMetadata.partitioned_graphs, which is
  # the documented way to inspect what Grappler produced for a given run.
  config = tf.compat.v1.ConfigProto()
  config.graph_options.optimizer_options.opt_level = (
      tf.compat.v1.training.OptimizerOptions.L1
  )
  run_options = config_pb2.RunOptions(
      trace_level=config_pb2.RunOptions.FULL_TRACE
  )
  run_metadata = config_pb2.RunMetadata()
  test_x = np.random.RandomState(2).randn(4).astype(np.float32)
  with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    actual = sess.run(
        y,
        feed_dict={x: test_x},
        options=run_options,
        run_metadata=run_metadata,
    )

  # Walk the executed graph and look for _KdnnSigmoid. RunMetadata captures
  # one GraphDef per partition; on a single-CPU graph there is exactly one.
  has_kdnn_sigmoid = False
  for partition in run_metadata.partitioned_graphs:
    for node in partition.node:
      if node.op == "_KdnnSigmoid":
        has_kdnn_sigmoid = True
        break
  if not has_kdnn_sigmoid:
    print(
        "FAIL: optimized graph does not contain _KdnnSigmoid. The Grappler "
        "remapper did not fire; either IsKDNNEnabled() returned false, "
        "the rewrite's dtype/shape constraints excluded this graph, or "
        "the rewrite was eliminated by a later pass.",
        file=sys.stderr,
    )
    return 1

  # Compare the rewritten op's output to the eager-mode reference.
  expected = 1.0 / (1.0 + np.exp(-(test_x @ w.numpy() + b.numpy())))
  max_abs_err = float(np.max(np.abs(actual - expected)))
  if max_abs_err > 1e-4:
    print(
        f"FAIL: numerical drift exceeds 1e-4: max_abs_err={max_abs_err:.3e}",
        file=sys.stderr,
    )
    return 1

  print(
      "PASS: optimized graph contains _KdnnSigmoid, "
      f"max_abs_err={max_abs_err:.3e}"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())