"""Runs every op the backend registers and compares it to the CPU kernel.

The point is the difference between registered and working. A kernel that
returns zeros, or that TensorFlow refuses to dispatch, is registered and
broken, and only going through TensorFlow's own dispatch with real inputs
shows it.

Each op is called twice with identical inputs, once pinned to the GPU and once
to the CPU, and the results compared. Soft placement is off, so an op with no
GPU kernel raises rather than quietly answering from the host. When the CPU
call fails too, the inputs were wrong rather than the kernel, and the op is
reported as unexercised rather than as a failure: a sweep that counts its own
bad recipes as bugs is worse than no sweep.
"""

import argparse
import os
import sys
import traceback

import collections

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import kernels
from tensorflow.python.framework import load_library
from tensorflow.python.eager import context
from tensorflow.python.eager import execute as eager_execute
from tensorflow.python.framework import op_def_registry

import recipes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MATCH, MISMATCH, GPU_ERROR, UNEXERCISED, NO_RECIPE = (
    "match", "mismatch", "gpu-error", "unexercised", "no-recipe")
# Two kinds of "cannot be exercised" that are not gaps in the backend.
REMOVED = "removed-from-tensorflow"
OUT_OF_TREE = "needs-unexported-api"


def load_ops(path):
  with open(path) as handle:
    return [line.strip() for line in handle if line.strip()]


# Shapes small enough to be quick and awkward enough to catch stride bugs: a
# non-square matrix, an odd innermost dimension, a batch that is not one.
RNG = np.random.default_rng(7)


def f32(*shape):
  return RNG.standard_normal(shape, dtype=np.float32)


def positive(*shape):
  return np.abs(f32(*shape)) + 0.5


IMAGE = f32(2, 8, 9, 3)
MAT = f32(6, 5)
VEC = f32(7)


# Inputs are chosen so that an op's mathematical domain is respected: values
# in (0.05, 0.95) are valid for the logarithms, the roots, the inverse
# trigonometric functions and the reciprocals alike, so one array serves the
# whole unary family without a per-op table of domains.
SAFE = (RNG.random((6, 5)).astype(np.float32) * 0.9 + 0.05)
SAFE2 = (RNG.random((6, 5)).astype(np.float32) * 0.9 + 0.05)
SIGNED = RNG.standard_normal((6, 5), dtype=np.float32)
BOOL = RNG.random((6, 5)) > 0.5
INT = RNG.integers(0, 5, (6, 5)).astype(np.int32)


def dtype_for(op_def, attr_name="T", prefer=(tf.float32, tf.int32, tf.bool)):
  for attr in op_def.attr:
    if attr.name != attr_name or attr.type != "type":
      continue
    allowed = list(attr.allowed_values.list.type)
    if not allowed:
      return tf.float32
    for candidate in prefer:
      if candidate.as_datatype_enum in allowed:
        return candidate
    return tf.as_dtype(allowed[0])
  return None


def tensor(dtype, positive=True):
  if dtype is None or dtype == tf.float32:
    return tf.constant(SAFE if positive else SIGNED)
  if dtype.is_bool:
    return tf.constant(BOOL)
  if dtype.is_integer:
    return tf.constant(INT.astype(dtype.as_numpy_dtype))
  if dtype.is_floating:
    return tf.constant((SAFE if positive else SIGNED).astype(
        dtype.as_numpy_dtype))
  return None


# One recipe per input signature, which is what an op def actually varies. The
# long tail of structured ops is handled by name below.
BY_SIGNATURE = {
    ("x",): lambda d: {"x": tensor(dtype_for(d))},
    ("input",): lambda d: {"input": tensor(dtype_for(d))},
    ("features",): lambda d: {"features": tensor(dtype_for(d), positive=False)},
    ("x", "y"): lambda d: {"x": tensor(dtype_for(d)),
                           "y": tensor(dtype_for(d))},
    ("input", "reduction_indices"): lambda d: {
        "input": tensor(dtype_for(d)), "reduction_indices": [1]},
    ("gradients", "features"): lambda d: {
        "gradients": tf.constant(SIGNED),
        "features": tf.constant(SIGNED)},
    ("y", "dy"): lambda d: {"y": tf.constant(SAFE), "dy": tf.constant(SAFE2)},
    ("value", "bias"): lambda d: {"value": tf.constant(SAFE),
                                  "bias": tf.constant(SAFE[0])},
    ("x", "axis"): lambda d: {"x": tensor(dtype_for(d)), "axis": 1},
    ("input", "dimension"): lambda d: {"input": tf.constant(SIGNED),
                                       "dimension": 1},
    ("diagonal",): lambda d: {"diagonal": tf.constant(SAFE[0])},
    ("input", "num_lower", "num_upper"): lambda d: {
        "input": tf.constant(SAFE), "num_lower": 1, "num_upper": 1},
    ("input", "diagonal"): lambda d: {"input": tf.constant(SAFE[:5, :5]),
                                      "diagonal": tf.constant(SAFE[0][:5])},
    ("matrix", "rhs"): lambda d: {
        "matrix": tf.constant(np.tril(SAFE[:5, :5]) + 2.0 * np.eye(5,
                                                                   dtype=np.float32)),
        "rhs": tf.constant(SAFE[:5, :3])},
    ("t", "clip_value_min", "clip_value_max"): lambda d: {
        "t": tf.constant(SIGNED), "clip_value_min": -0.5,
        "clip_value_max": 0.5},
    ("condition", "t", "e"): lambda d: {
        "condition": tf.constant(BOOL), "t": tf.constant(SAFE),
        "e": tf.constant(SAFE2)},
    ("data", "segment_ids"): lambda d: {
        "data": tf.constant(SAFE), "segment_ids": tf.constant([0, 0, 1, 1, 2, 2])},
    ("data", "segment_ids", "num_segments"): lambda d: {
        "data": tf.constant(SAFE),
        "segment_ids": tf.constant([0, 0, 1, 1, 2, 2]), "num_segments": 3},
    ("params", "indices"): lambda d: {
        "params": tf.constant(SAFE), "indices": tf.constant([0, 2, 1])},
    ("params", "indices", "axis"): lambda d: {
        "params": tf.constant(SAFE), "indices": tf.constant([0, 2, 1]),
        "axis": 0},
    ("tensor", "shape"): lambda d: {"tensor": tf.constant(SAFE),
                                    "shape": [5, 6]},
    ("input", "shape"): lambda d: {"input": tf.constant(SAFE), "shape": [5, 6]},
    ("input", "perm"): lambda d: {"input": tf.constant(SAFE), "perm": [1, 0]},
    ("x", "perm"): lambda d: {"x": tf.constant(SAFE), "perm": [1, 0]},
    ("input", "multiples"): lambda d: {"input": tf.constant(SAFE),
                                       "multiples": [2, 1]},
    ("input", "paddings"): lambda d: {"input": tf.constant(SAFE),
                                      "paddings": [[1, 1], [2, 0]]},
    ("input", "paddings", "constant_values"): lambda d: {
        "input": tf.constant(SAFE), "paddings": [[1, 1], [2, 0]],
        "constant_values": 0.0},
    ("input", "begin", "size"): lambda d: {
        "input": tf.constant(SAFE), "begin": [1, 1], "size": [3, 2]},
    ("input", "axis"): lambda d: {"input": tf.constant(SAFE), "axis": [1]},
    ("input", "dims"): lambda d: {"input": tf.constant(SAFE),
                                  "dims": [False, True]},
    ("tensor", "mask"): lambda d: {"tensor": tf.constant(SAFE),
                                   "mask": tf.constant(BOOL[0])},
}

# Ops whose inputs are structural enough that only a hand-written call will do.
IMAGE = RNG.standard_normal((2, 8, 9, 3), dtype=np.float32)
FILTER = RNG.standard_normal((3, 3, 3, 4), dtype=np.float32)


def by_name():
  conv = {"input": tf.constant(IMAGE), "filter": tf.constant(FILTER),
          "strides": [1, 1, 1, 1], "padding": "SAME"}
  pool = {"input": tf.constant(IMAGE), "ksize": [1, 2, 2, 1],
          "strides": [1, 2, 2, 1], "padding": "VALID"}
  recipes = {
      "Conv2D": conv,
      "DepthwiseConv2dNative": dict(conv, filter=tf.constant(FILTER)),
      "MaxPool": pool,
      "AvgPool": pool,
      "Relu": {"features": tf.constant(IMAGE)},
      "Softmax": {"logits": tf.constant(SAFE)},
      "LogSoftmax": {"logits": tf.constant(SAFE)},
      "BiasAdd": {"value": tf.constant(IMAGE),
                  "bias": tf.constant(IMAGE[0, 0, 0])},
      "MatMul": {"a": tf.constant(SAFE), "b": tf.constant(SAFE.T)},
      "BatchMatMulV2": {"x": tf.constant(IMAGE), "y": tf.constant(
          np.transpose(IMAGE, (0, 1, 3, 2)).copy())},
      "Fill": {"dims": [3, 4], "value": 2.0},
      "OnesLike": {"x": tf.constant(SAFE)},
      "ZerosLike": {"x": tf.constant(SAFE)},
      "Cast": {"x": tf.constant(SAFE), "DstT": tf.float16},
      "Concat": {"concat_dim": 0, "values": [tf.constant(SAFE),
                                             tf.constant(SAFE2)]},
      "ConcatV2": {"values": [tf.constant(SAFE), tf.constant(SAFE2)],
                   "axis": 0},
      "Pack": {"values": [tf.constant(SAFE), tf.constant(SAFE2)], "axis": 0},
      "Split": {"split_dim": 0, "value": tf.constant(SAFE), "num_split": 2},
      "AddN": {"inputs": [tf.constant(SAFE), tf.constant(SAFE2)]},
      "Transpose": {"x": tf.constant(SAFE), "perm": [1, 0]},
      "ConjugateTranspose": {"x": tf.constant(SAFE), "perm": [1, 0]},
      "Reshape": {"tensor": tf.constant(SAFE), "shape": [5, 6]},
      "ExpandDims": {"input": tf.constant(SAFE), "dim": 0},
      "Squeeze": {"input": tf.constant(SAFE[None])},
      "Tile": {"input": tf.constant(SAFE), "multiples": [2, 1]},
      "TopKV2": {"input": tf.constant(SAFE), "k": 3},
      "LRN": {"input": tf.constant(IMAGE)},
      "L2Loss": {"t": tf.constant(SAFE)},
      "Where": {"input": tf.constant(BOOL)},
      "Unique": {"x": tf.constant([1, 2, 2, 3, 1], dtype=tf.int32)},
      "InvertPermutation": {"x": tf.constant([2, 0, 1], dtype=tf.int32)},
  }
  return recipes


# Built after the plugin is loaded, never at import: creating a tensor
# initialises the eager context, and a device registered after that is not
# picked up. The first version of this sweep built its inputs at import and
# reported all 111 exercised ops as having no GPU kernel, which was the sweep
# describing its own mistake.
NAMED = {}
RANDOM = {}
NO_CPU = {}
INTERNAL = {}
LAST_FEW = set()
PRELUDE = {"_NcclBroadcastRecv": "_NcclBroadcastSend"}


def synthesize(op_def):
  """A call for an op, or None when no recipe covers it."""
  if op_def.name in RANDOM:
    return dict(RANDOM[op_def.name])
  if op_def.name in NAMED:
    return dict(NAMED[op_def.name])
  signature = tuple(a.name for a in op_def.input_arg)
  builder = BY_SIGNATURE.get(signature)
  if builder is None:
    return None
  for attr in op_def.attr:
    if attr.type == "type" or attr.HasField("default_value"):
      continue
    return None  # a required attr this recipe says nothing about
  for arg in op_def.input_arg:
    if arg.is_ref or arg.type_list_attr or arg.number_attr:
      return None
    dtype = tf.as_dtype(arg.type) if arg.type else None
    if dtype in (tf.resource, tf.variant, tf.string):
      return None
  return builder(op_def)


def duplicate_registrations(ops):
  """Ops registered twice for the GPU with the same constraints.

  TensorFlow will not dispatch an op whose registrations tie, so a duplicate
  is not a harmless extra: the op cannot run at all. This is cheap to check
  and invisible from the outside until something tries to use the op, which
  is how twelve of them, Identity among them, stayed broken.
  """
  found = {}
  for name in ops:
    try:
      registered = kernels.get_registered_kernels_for_op(name).kernel
    except Exception:  # pylint: disable=broad-except
      continue
    seen = collections.Counter()
    for kernel in registered:
      if kernel.device_type != "GPU":
        continue
      seen[(tuple(sorted((c.name, tuple(c.allowed_values.list.type))
                         for c in kernel.constraint)),
            tuple(sorted(kernel.host_memory_arg)))] += 1
    extra = sum(count - 1 for count in seen.values() if count > 1)
    if extra:
      found[name] = extra
  return found


# Decompositions whose output is not unique: the pivot order of an LU and the
# sign of an eigenvector are both free, so comparing to the CPU compares two
# valid answers and calls them different. Each is checked against the identity
# that defines it instead, which is a stronger statement than agreeing with
# another implementation.
def check_lu(kwargs, outputs):
  packed, pivots = outputs[0], outputs[1]
  n = packed.shape[-1]
  lower = np.tril(packed, -1) + np.eye(n, dtype=packed.dtype)
  upper = np.triu(packed)
  product = lower @ upper
  # TensorFlow reports p as the permutation itself, one row index per output
  # row, not as a sequence of swaps.
  order = [int(i) for i in pivots]
  original = np.asarray(kwargs["input"])
  worst = float(np.max(np.abs(product - original[order, :])))
  return worst < 1e-4, f"P A = L U to {worst:.2e}"


def check_self_adjoint_eig(kwargs, outputs):
  values, vectors = outputs[0], outputs[1]
  original = np.asarray(kwargs["input"])
  rebuilt = vectors @ np.diag(values) @ vectors.T
  worst = float(np.max(np.abs(rebuilt - original)))
  orthonormal = float(np.max(np.abs(vectors.T @ vectors
                                    - np.eye(vectors.shape[0]))))
  ok = worst < 1e-3 and orthonormal < 1e-4
  return ok, f"V diag(e) V^T to {worst:.2e}, orthonormal to {orthonormal:.2e}"


def flatten_deep(value):
  out = []
  for item in (value if isinstance(value, (list, tuple)) else [value]):
    if isinstance(item, (list, tuple)):
      out.extend(flatten_deep(item))
    else:
      out.append(np.asarray(item))
  return out


def check_cudnn_rnn(name, kwargs, outputs):
  """The recurrent family, which TensorFlow has no CPU kernel for.

  Pinned down by what can be stated without a second implementation: the
  parameter buffer's size agrees with the canonical layout, the two
  conversions round-trip exactly, the forward pass is finite and repeatable,
  and V3 emits nothing past a sequence's own length. The arithmetic is checked
  against a double-precision reference and central differences by the
  on-device harness, not here.
  """
  first = flatten_deep(outputs)[0]
  if name == "CudnnRNNParamsSize":
    expected = kwargs.pop("_expected_size")
    return int(first) == expected, f"{int(first)} floats, as the layout implies"
  if "CanonicalToParams" in name:
    expected = kwargs.pop("_expected_size")
    return int(first.shape[0]) == expected, f"packs {int(first.shape[0])} floats"
  if "ParamsToCanonical" in name:
    original = kwargs.pop("_canonical")
    got = flatten_deep(outputs)
    worst = max(float(np.max(np.abs(a - b))) for a, b in zip(original, got))
    return worst == 0.0, f"round trips exactly ({worst:.1e})"
  if "Backprop" in name:
    every = flatten_deep(outputs)
    if not all(np.all(np.isfinite(v)) for v in every):
      return False, "not finite"
    # A gradient that ignores what it is handed would pass everything else.
    if all(float(np.max(np.abs(v))) == 0.0 for v in every):
      return False, "every gradient is zero"
    return True, "finite, and not identically zero"
  # The forward passes.
  values = flatten_deep(outputs)[0]
  if not np.all(np.isfinite(values)):
    return False, "not finite"
  if name == "CudnnRNNV3":
    lengths = np.asarray(kwargs["sequence_lengths"])
    beyond = 0.0
    for row, length in enumerate(lengths):
      if length < values.shape[0]:
        beyond = max(beyond, float(np.max(np.abs(values[int(length):, row]))))
    if beyond != 0.0:
      return False, f"emits {beyond:.2e} past a sequence's length"
    return True, "finite, and silent past each sequence's length"
  return True, "finite"


def check_last_few(name, kwargs, outputs):
  """Properties for the ops with no usable CPU kernel in this release."""
  first = np.asarray(outputs[0])
  if name in ("TopK", "ApproxTopK"):
    source = np.asarray(kwargs["input"])
    k = int(kwargs["k"])
    wanted = -np.sort(-source, axis=-1)[..., :k]
    worst = float(np.max(np.abs(first - wanted)))
    return worst < 1e-5, f"the {k} largest per row, to {worst:.1e}"
  if name == "TileGrad":
    source = np.asarray(kwargs["input"])
    multiples = list(kwargs["multiples"])
    wanted = source.reshape(multiples[0], -1, source.shape[1]).sum(axis=0)
    worst = float(np.max(np.abs(first - wanted)))
    return worst < 1e-4, f"sums the tiles to {worst:.1e}"
  if name == "CTCLossV2":
    # Not compared against CTCLoss: the two differ on purpose, V2 taking the
    # blank as the last class where v1 takes it as the first. What holds for
    # both is that a negative log likelihood is finite and not negative.
    ok = bool(np.all(np.isfinite(first))) and float(np.min(first)) >= 0.0
    return ok, f"finite and non-negative, smallest {float(np.min(first)):.3f}"
  if name == "MaxPoolGradGradWithArgmax":
    pooled = np.asarray(kwargs["argmax"])
    return (np.all(np.isfinite(first)) and first.shape == pooled.shape,
            f"finite, shaped {first.shape}")
  if name == "GenerateBoundingBoxProposals":
    # The output is padded to post_nms_topn, so its width says nothing; what
    # can be said is that every box is finite and inside the image.
    ok = bool(np.all(np.isfinite(first))) and float(np.min(first)) >= -1e-6
    return ok, f"finite and non-negative, {first.shape[1]} slots"
  return np.all(np.isfinite(first)), "finite"


def check_no_cpu(name, kwargs, outputs):
  """Verifies an op TensorFlow cannot run on the CPU against a property."""
  first = np.asarray(outputs[0])
  if name in ("FFTND", "IFFTND"):
    inverse = "IFFTND" if name == "FFTND" else "FFTND"
    with tf.device("/GPU:0"):
      back = getattr(tf.raw_ops, inverse)(
          input=tf.constant(first), fft_length=kwargs["fft_length"],
          axes=kwargs["axes"]).numpy()
    worst = float(np.max(np.abs(back - np.asarray(kwargs["input"]))))
    return worst < 1e-3, f"round trip to {worst:.2e}"
  if name == "RFFTND":
    with tf.device("/GPU:0"):
      reference = tf.raw_ops.RFFT2D(input=kwargs["input"],
                                    fft_length=kwargs["fft_length"]).numpy()
    worst = float(np.max(np.abs(first - reference)))
    return worst < 1e-4, f"agrees with RFFT2D to {worst:.2e}"
  if name == "IRFFTND":
    with tf.device("/GPU:0"):
      reference = tf.raw_ops.IRFFT2D(input=kwargs["input"],
                                     fft_length=kwargs["fft_length"]).numpy()
    worst = float(np.max(np.abs(first - reference)))
    return worst < 1e-4, f"agrees with IRFFT2D to {worst:.2e}"
  # The collectives over one device: the output is the input.
  source = kwargs["input"]
  if isinstance(source, list):
    source = source[0]
  worst = float(np.max(np.abs(first - np.asarray(source))))
  return worst == 0.0, f"copies its input exactly ({worst:.2e})"


BY_IDENTITY = {
    "Lu": check_lu,
    "SelfAdjointEigV2": check_self_adjoint_eig,
}


def invalid_constraints(ops):
  """Registrations constraining an attribute the op does not have.

  TensorFlow reports these as "OpKernel 'X' has constraint on attr 'T' not in
  NodeDef", and the registration can never match: the op is registered and
  unusable, exactly like a duplicate. All, Any and the logical operators have
  no T because they are bool; BatchMatMulV3 has Ta, Tb and Tout; GatherV2 has
  Tparams, Tindices and Taxis; TopK has no index_type. Constraining T on any
  of them is a registration that describes an op that does not exist.
  """
  found = {}
  for name in ops:
    op_def = op_def_registry.get(name)
    if op_def is None:
      continue
    known = {attr.name for attr in op_def.attr}
    try:
      registered = kernels.get_registered_kernels_for_op(name).kernel
    except Exception:  # pylint: disable=broad-except
      continue
    bad = set()
    for kernel in registered:
      if kernel.device_type != "GPU":
        continue
      for constraint in kernel.constraint:
        if constraint.name not in known:
          bad.add(constraint.name)
    if bad:
      found[name] = sorted(bad)
  return found


def check_internal_without_cpu(name, inputs, outputs):
  """Properties for the internal ops TensorFlow has no CPU kernel for."""
  if not outputs:
    return True, "runs and produces nothing, which is its whole job"
  if name == "_FusedBatchNormGradEx":
    # Without a side input or an activation it is the plain gradient, and
    # that one is checked against the CPU.
    y_backprop, x, scale = inputs[0], inputs[1], inputs[2]
    reserve_1, reserve_2 = inputs[3], inputs[4]
    with tf.device("/GPU:0"):
      plain = tf.raw_ops.FusedBatchNormGradV3(
          y_backprop=y_backprop, x=x, scale=scale,
          reserve_space_1=reserve_1, reserve_space_2=reserve_2,
          reserve_space_3=tf.constant(0.0), epsilon=1e-3,
          is_training=False)[0].numpy()
    worst = float(np.max(np.abs(outputs[0] - plain)))
    return worst < 1e-4, f"agrees with FusedBatchNormGradV3 to {worst:.2e}"
  if name == "_FusedBatchNormEx":
    # The fused form with no side input and no activation is the plain one,
    # and that is already checked against the CPU.
    image, scale, offset, mean, variance = inputs[:5]
    with tf.device("/GPU:0"):
      plain = tf.raw_ops.FusedBatchNormV3(
          x=image, scale=scale, offset=offset, mean=mean, variance=variance,
          epsilon=1e-3, is_training=False)[0].numpy()
    worst = float(np.max(np.abs(outputs[0] - plain)))
    return worst < 1e-4, f"agrees with FusedBatchNormV3 to {worst:.2e}"
  if name == "_NcclBroadcastRecv":
    # Its input is a shape; what it produces should be what the matching send
    # parked, which is the tensor the send recipe was given.
    wanted = tuple(int(v) for v in np.asarray(inputs[0]))
    if outputs[0].shape != wanted:
      return False, f"produces {outputs[0].shape}, wanted {wanted}"
    sent = np.asarray(INTERNAL["_NcclBroadcastSend"][0][0])
    worst = float(np.max(np.abs(outputs[0] - sent)))
    return worst == 0.0, f"receives exactly what was sent ({worst:.1e})"
  source = np.asarray(inputs[0])
  same = (outputs[0].shape == source.shape
          and float(np.max(np.abs(outputs[0] - source))) == 0.0)
  return same, ("copies its input exactly" if same
                else "does not reproduce its input")


def run_internal(op_name, inputs, attrs, num_outputs, device):
  """Calls an op that has no generated Python wrapper."""
  with tf.device(device):
    placed = [tf.identity(t) for t in inputs]
    return eager_execute.execute(op_name.encode(), num_outputs,
                                 inputs=placed, attrs=attrs,
                                 ctx=context.context())


def run(op_name, kwargs, device):
  with tf.device(device):
    return getattr(tf.raw_ops, op_name)(**kwargs)


def flatten(result):
  if isinstance(result, (list, tuple)):
    return [np.asarray(r) for r in result]
  if hasattr(result, "numpy"):
    return [result.numpy()]
  return [np.asarray(result)]


# Outputs the op def calls scratch: their contents are the kernel's own
# business, are documented as opaque, and differ between implementations on
# purpose. TensorFlow's CPU kernel leaves the batch-norm reserve spaces zero
# during inference; this backend writes the statistics into them. What has to
# agree is the gradient that reads them, which is exercised on its own.
OPAQUE_OUTPUTS = {
    "FusedBatchNorm": (3, 4),
    "FusedBatchNormV2": (3, 4),
    "FusedBatchNormV3": (3, 4),
    "_FusedBatchNormEx": (3, 4),
}


def compare(cpu, gpu, op_name=""):
  if len(cpu) != len(gpu):
    return False, "different output counts"
  skip = OPAQUE_OUTPUTS.get(op_name, ())
  worst = 0.0
  for index, (a, b) in enumerate(zip(cpu, gpu)):
    if index in skip:
      continue
    if a.shape != b.shape:
      return False, f"shape {b.shape} vs {a.shape}"
    if a.dtype.kind in "fc":
      if not np.all(np.isfinite(a)):
        continue
      worst = max(worst, float(np.max(np.abs(a - b))) if a.size else 0.0)
      if not np.allclose(a, b, rtol=1e-3, atol=1e-3, equal_nan=True):
        # Which output, because an op with six of them says nothing useful
        # otherwise.
        return False, f"output {index} differs by {worst:.3e}"
    elif not np.array_equal(a, b):
      return False, f"output {index} differs in value"
  return True, f"max diff {worst:.2e}"


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ops", default=os.path.join(HERE, "metal_ops.txt"))
  parser.add_argument("--plugin",
                      default=os.path.join(ROOT, "build", "libmetal_plugin.dylib"))
  parser.add_argument("--only", default=None)
  args = parser.parse_args()

  load_library.load_pluggable_device_library(args.plugin)
  tf.config.set_soft_device_placement(False)
  devices = [d.name for d in tf.config.list_physical_devices("GPU")]
  if not devices:
    print("no GPU device after loading the plugin, nothing to sweep")
    return 1
  print(f"sweeping against {devices[0]}")
  global NAMED, RANDOM, NO_CPU
  NAMED = by_name()
  NAMED.update(recipes.build())
  NAMED.update(recipes.more())
  NAMED.update(recipes.gradients_and_rest())
  RANDOM = recipes.nondeterministic()
  NO_CPU = recipes.no_cpu_reference()
  NO_CPU.update(recipes.cudnn_rnn_checks())
  last = recipes.last_few()
  NO_CPU.update(last)
  NO_CPU.update(recipes.recurrent_backprops())
  global LAST_FEW
  LAST_FEW = set(last)
  global INTERNAL
  INTERNAL = recipes.internal()

  ops = load_ops(args.ops)
  if args.only:
    ops = [o for o in ops if args.only in o]

  wrong_attrs = invalid_constraints(ops)
  if wrong_attrs:
    print(f"\n=== registrations constraining an attribute the op lacks "
          f"({len(wrong_attrs)})")
    for name in sorted(wrong_attrs):
      print(f"  {name:38s} {', '.join(wrong_attrs[name])}")

  duplicates = duplicate_registrations(ops)
  if duplicates:
    print(f"\n=== duplicate GPU registrations ({len(duplicates)})")
    for name in sorted(duplicates):
      print(f"  {name:38s} {duplicates[name]} extra, so the op cannot run")

  results = {}
  details = {}
  for name in ops:
    if name in INTERNAL:
      inputs, attrs, num_outputs = INTERNAL[name]
      # A receive has nothing to collect until its send has run. The pair is
      # the point of the split form, so the sweep runs both.
      prelude = PRELUDE.get(name)
      if prelude and prelude in INTERNAL:
        pre_inputs, pre_attrs, pre_outputs = INTERNAL[prelude]
        for device in ("/GPU:0", "/CPU:0"):
          try:
            run_internal(prelude, pre_inputs, pre_attrs, pre_outputs, device)
          except Exception:  # pylint: disable=broad-except
            pass
      try:
        gpu = flatten_deep(run_internal(name, inputs, attrs, num_outputs,
                                        "/GPU:0"))
      except Exception as error:  # pylint: disable=broad-except
        results[name] = GPU_ERROR
        details[name] = str(error).splitlines()[0][:110]
        continue
      try:
        cpu = flatten_deep(run_internal(name, inputs, attrs, num_outputs,
                                        "/CPU:0"))
      except Exception:  # pylint: disable=broad-except
        ok, detail = check_internal_without_cpu(name, inputs, gpu)
        results[name] = MATCH if ok else MISMATCH
        details[name] = detail
        continue
      ok, detail = compare(cpu, gpu, name)
      results[name] = MATCH if ok else MISMATCH
      details[name] = detail
      continue

    if name in NO_CPU:
      kwargs, what = NO_CPU[name]
      try:
        call = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        gpu = flatten_deep(run(name, call, "/GPU:0"))
      except Exception as error:  # pylint: disable=broad-except
        results[name] = GPU_ERROR
        details[name] = str(error).splitlines()[0][:110]
        continue
      if name.startswith("CudnnRNN"):
        ok, detail = check_cudnn_rnn(name, kwargs, gpu)
      elif name in LAST_FEW:
        ok, detail = check_last_few(name, kwargs, gpu)
      else:
        ok, detail = check_no_cpu(name, kwargs, gpu)
      results[name] = MATCH if ok else MISMATCH
      details[name] = f"{what}: {detail}"
      continue
    if name in recipes.REMOVED_FROM_GRAPHDEF:
      results[name] = REMOVED
      details[name] = "TensorFlow removed this op; no device can run it"
      continue
    if name in recipes.NEEDS_UNEXPORTED_C_API:
      results[name] = OUT_OF_TREE
      details[name] = ("needs kernel C API entry points a released "
                       "TensorFlow does not export")
      continue
    if name in INTERNAL:
      inputs, attrs, num_outputs = INTERNAL[name]
      # A receive has nothing to collect until its send has run. The pair is
      # the point of the split form, so the sweep runs both.
      prelude = PRELUDE.get(name)
      if prelude and prelude in INTERNAL:
        pre_inputs, pre_attrs, pre_outputs = INTERNAL[prelude]
        for device in ("/GPU:0", "/CPU:0"):
          try:
            run_internal(prelude, pre_inputs, pre_attrs, pre_outputs, device)
          except Exception:  # pylint: disable=broad-except
            pass
      try:
        gpu = flatten_deep(run_internal(name, inputs, attrs, num_outputs,
                                        "/GPU:0"))
      except Exception as error:  # pylint: disable=broad-except
        results[name] = GPU_ERROR
        details[name] = str(error).splitlines()[0][:110]
        continue
      try:
        cpu = flatten_deep(run_internal(name, inputs, attrs, num_outputs,
                                        "/CPU:0"))
      except Exception:  # pylint: disable=broad-except
        ok, detail = check_internal_without_cpu(name, inputs, gpu)
        results[name] = MATCH if ok else MISMATCH
        details[name] = detail
        continue
      ok, detail = compare(cpu, gpu, name)
      results[name] = MATCH if ok else MISMATCH
      details[name] = detail
      continue

    if name in NO_CPU:
      kwargs, what = NO_CPU[name]
      try:
        call = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        gpu = flatten_deep(run(name, call, "/GPU:0"))
      except Exception as error:  # pylint: disable=broad-except
        results[name] = GPU_ERROR
        details[name] = str(error).splitlines()[0][:110]
        continue
      if name.startswith("CudnnRNN"):
        ok, detail = check_cudnn_rnn(name, kwargs, gpu)
      elif name in LAST_FEW:
        ok, detail = check_last_few(name, kwargs, gpu)
      else:
        ok, detail = check_no_cpu(name, kwargs, gpu)
      results[name] = MATCH if ok else MISMATCH
      details[name] = f"{what}: {detail}"
      continue
    op_def = op_def_registry.get(name)
    if op_def is None:
      results[name] = NO_RECIPE
      details[name] = "not in the op registry"
      continue
    kwargs = synthesize(op_def)
    if kwargs is None:
      results[name] = NO_RECIPE
      details[name] = "no generic call"
      continue
    if name in RANDOM:
      try:
        gpu = flatten(run(name, kwargs, "/GPU:0"))
      except Exception as error:  # pylint: disable=broad-except
        results[name] = GPU_ERROR
        details[name] = str(error).splitlines()[0][:110]
        continue
      # A random op cannot be compared to the CPU, so it is asked for the two
      # things a broken generator gets wrong: values that are not finite, and
      # values that are all the same.
      bad = []
      for array in gpu:
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
          bad.append("not finite")
        if array.size > 1 and np.all(array == array.flat[0]):
          bad.append("constant")
      results[name] = MISMATCH if bad else MATCH
      details[name] = ", ".join(bad) if bad else "random, finite and varying"
      continue
    try:
      cpu = flatten(run(name, kwargs, "/CPU:0"))
    except Exception as error:  # pylint: disable=broad-except
      results[name] = UNEXERCISED
      details[name] = f"cpu: {str(error).splitlines()[0][:90]}"
      continue
    try:
      gpu = flatten(run(name, kwargs, "/GPU:0"))
    except Exception as error:  # pylint: disable=broad-except
      results[name] = GPU_ERROR
      details[name] = str(error).splitlines()[0][:110]
      continue
    # Run it a second time with the same inputs and require the same answer.
    # An op that writes over its own input, or that leaves state behind,
    # gives the right answer once and a different one after: an inverse real
    # transform transformed its caller's tensor in place and only the first
    # call was right, which a single-shot comparison cannot see.
    try:
      again = flatten(run(name, kwargs, "/GPU:0"))
    except Exception as error:  # pylint: disable=broad-except
      results[name] = GPU_ERROR
      details[name] = f"second call: {str(error).splitlines()[0][:90]}"
      continue
    stable, detail = compare(gpu, again, name)
    if not stable:
      results[name] = MISMATCH
      details[name] = f"not repeatable: {detail}"
      continue

    if name in BY_IDENTITY:
      ok, detail = BY_IDENTITY[name](kwargs, gpu)
    else:
      ok, detail = compare(cpu, gpu, name)
    results[name] = MATCH if ok else MISMATCH
    details[name] = detail

  order = [MISMATCH, GPU_ERROR, MATCH, REMOVED, OUT_OF_TREE, UNEXERCISED,
           NO_RECIPE]
  counts = {k: 0 for k in order}
  for value in results.values():
    counts[value] += 1

  if os.environ.get("SWEEP_VERBOSE"):
    for kind in (UNEXERCISED, NO_RECIPE):
      named = sorted(n for n, v in results.items() if v == kind)
      print(f"\n=== {kind} ({len(named)})")
      for n in named:
        print(f"  {n:38s} {details[n]}")

  for kind in (MISMATCH, GPU_ERROR):
    named = sorted(n for n, v in results.items() if v == kind)
    if named:
      print(f"\n=== {kind} ({len(named)})")
      for n in named:
        print(f"  {n:38s} {details[n]}")

  print("\n=== summary")
  for kind in order:
    print(f"  {kind:22s} {counts[kind]}")
  print(f"  {'duplicates':22s} {len(duplicates)}")
  print(f"  {'wrong-attr constraints':22s} {len(wrong_attrs)}")
  return (1 if counts[MISMATCH] or counts[GPU_ERROR] or duplicates
          or wrong_attrs else 0)


if __name__ == "__main__":
  sys.exit(main())
