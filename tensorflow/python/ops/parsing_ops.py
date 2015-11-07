"""Parsing Ops."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.gen_parsing_ops import *


ops.NoGradient("DecodeRaw")
ops.NoGradient("StringToNumber")


# pylint: disable=protected-access
def parse_example(serialized,
                  names=None,
                  sparse_keys=None,
                  sparse_types=None,
                  dense_keys=None,
                  dense_types=None,
                  dense_defaults=None,
                  dense_shapes=None,
                  name="ParseExample"):
  """Parse Example protos.

  Args:
    serialized: string vector, a batch of binary serialized Example protos.
    names: A string vector, the names of the serialized protos.
      "names" may contain, e.g., table key (descriptive) names for the
      corresponding serialized protos.  These are purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      "names" may be an empty vector, if no names are available.
      If non-empty, this vector must be the same length as "serialized".
    sparse_keys: A string list of keys in the Examples' features.
      These keys are associated with sparse values.
    sparse_types: A list of DTypes.
      This list's length must match that of sparse_keys.  Currently
      parse_example supports tf.float32 (FloatList), tf.int64 (Int64List),
      and tf.string (BytesList).
    dense_keys: A string list of keys in the Examples' features.
      These keys are associated with dense values.
    dense_types: A list of DTypes.
      This list's length must match that of dense_keys.  Currently
      parse_example supports tf.float32 (FloatList), tf.int64 (Int64List),
      and tf.string (BytesList).
    dense_defaults: A dict of {key:Tensor} (some may be missing).
      The keys of the dict must match the dense_keys of the feature.
      If a key is not present in this dictionary, the corresponding dense
      Feature is required in all elements of serialized.
    dense_shapes: A list of tuples.
      Entries provide the shape of data in each dense Feature in features.
      The length of dense_shapes must be the same as the length of dense_keys.
      The number of elements in the Feature corresponding to dense_key[j]
      must always have np.prod(dense_shapes[j]) entries.
      If dense_shapes[j] == (D0, D1, ..., DN) then the the shape of output
      Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
      The dense outputs are just the inputs row-stacked by batch.
    name: (Optional) Name of Op in the graph.

  Returns:
    A dictionary mapping keys to Tensors and SparseTensors.

    The key dense_keys[j] is mapped to a tensor of type dense_types[j] and
    of shape (serialized.size(),) + dense_shapes[j] (i.e., the dense outputs are
    inputs, reshaped in row-major format and then row-stacked by batch).

    The key sparse_keys[j] is mapped to a SparseTensor of type sparse_types[j].
    The SparseTensor represents a ragged matrix.  Its indices are [batch, index]
    where "batch" is is the batch entry the value is from, and "index" is the
    value's index in the list of values associated with that feature
    and example.  For example, if one expects a tf.float32 sparse feature "ft"
    and three serialized examples are provided:

    serialized = [
      features:
        { feature: [ key: { "ft" value: float_list: { value: [1.0, 2.0] } } ] },
      features:
        { feature: [] },
      features:
        { feature: [ key: { "ft" value: float_list: { value: [3.0] } } ] }
    ]

    then the output will look like:

      {"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                          values=[1.0, 2.0, 3.0],
                          shape=(3, 2)) }

  Raises:
    ValueError: If sparse and dense keys intersect, or input lengths do not
      match up for sparse_* (similarly for dense_*).
    TypeError: If an input is malformed.

  Example input, format, and output: Just Sparse Inputs
  ================================================

  Given two brain.Example input protos:

  serialized:  // serialized versions of the protos below
    [features: {
      feature: { key: "kw" value: { bytes_list: { value: [ "knit", "big" ] } } }
      feature: { key: "gps" value: { float_list: { value: [] } } }
     },
     features: {
      feature: { key: "kw" value: { bytes_list: { value: [ "emmy" ] } } }
      feature: { key: "dank" value: { int64_list: { value: [ 42 ] } } }
      feature: { key: "gps" value: { } }
    }]
  names: ["input0", "input1"],
  sparse_keys: ["kw", "dank", "gps"]
  sparse_types: [DT_STRING, DT_INT64, DT_FLOAT]

  Then the expected output is a dictionary:
  {
    "kw": SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=["knit", "big", "emmy"]
        shape=[2, 2]),
    "dank": SparseTensor(
        indices=[[1, 0]],
        values=[42],
        shape=[2, 1]),
    "gps": SparseTensor(
        indices=[],
        values=[],
        shape=[2, 0]),
  }


  Example input, format, and output: Dense Inputs (without defaults)
  ==================================================================

  Given two brain.Example input protos:

  serialized:  // serialized versions of the protos below
    [features: {
      feature: { key: "age" value: { int64_list: { value: [ 0 ] } } }
      feature: { key: "gender" value: { bytes_list: { value: [ "f" ] } } }
     },
     features: {
      feature: { key: "age" value: { int64_list: { value: [] } } }
      feature: { key: "gender" value: { bytes_list: { value: [ "f" ] } } }
    }]
  names: ["input0", "input1"],
  dense_keys: np.array(["age", "gender"])
  dense_types: [tf.int64, tf.string]
  dense_defaults: {
    "age": -1  # defaults to -1 if missing
               # "gender" has no specified default so it's required
  }
  dense_shapes: [(1,), (1,)]  # age, gender, label, weight

  Then the expected output is a dictionary:
  {
    "age": [[0], [-1]],
    "gender": [["f"], ["f"]],
  }


  Example input, format, and output: Dense Inputs (with defaults)
  ===============================================================

  Given two brain.Example input protos:

  serialized:  // serialized versions of the protos below
    [features: {
      feature: { key: "weight" value: { float_list: { value: [ 1.0 ] } } }
     },
     features: {
      feature: { key: "label" value: { float_list: { value: [ -1.0, 0.0 ] } } }
    }]
  names: ["input0", "input1"],
  dense_keys: np.array(["label", "weight"])
  dense_defaults: {
    "label": [1.0, 2.0],  # float (default: vector)
    "weight": 5.0         # float (default: scalar, 5.0)
  }
  dense_shapes: [(2,), (1,)]  # age, gender, label, weight

  Then the expected output is a dictionary:
  {
    "label": [[1.0, 2.0], [-1.0, 0.0]],
    "weight": [[1.0], [5.0]],
  }
  """
  names = [] if names is None else names
  dense_defaults = {} if dense_defaults is None else dense_defaults
  sparse_keys = [] if sparse_keys is None else sparse_keys
  sparse_types = [] if sparse_types is None else sparse_types
  dense_keys = [] if dense_keys is None else dense_keys
  dense_types = [] if dense_types is None else dense_types
  dense_shapes = [
      []] * len(dense_keys) if dense_shapes is None else dense_shapes

  num_dense = len(dense_keys)
  num_sparse = len(sparse_keys)

  if len(dense_shapes) != num_dense:
    raise ValueError("len(dense_shapes) != len(dense_keys): %d vs. %d"
                     % (len(dense_shapes), num_dense))
  if len(dense_types) != num_dense:
    raise ValueError("len(dense_types) != len(num_dense): %d vs. %d"
                     % (len(dense_types), num_dense))
  if len(sparse_types) != num_sparse:
    raise ValueError("len(sparse_types) != len(sparse_keys): %d vs. %d"
                     % (len(sparse_types), num_sparse))
  if num_dense + num_sparse == 0:
    raise ValueError("Must provide at least one sparse key or dense key")
  if not set(dense_keys).isdisjoint(set(sparse_keys)):
    raise ValueError(
        "Dense and sparse keys must not intersect; intersection: %s" %
        set(dense_keys).intersection(set(sparse_keys)))

  dense_defaults_vec = []
  for i, key in enumerate(dense_keys):
    default_value = dense_defaults.get(key)
    if default_value is None:
      default_value = constant_op.constant([], dtype=dense_types[i])
    elif not isinstance(default_value, ops.Tensor):
      default_value = ops.convert_to_tensor(
          default_value, dtype=dense_types[i], name=key)
      default_value = array_ops.reshape(default_value, dense_shapes[i])

    dense_defaults_vec.append(default_value)

  dense_shapes = [tensor_util.MakeTensorShapeProto(shape)
                  if isinstance(shape, (list, tuple)) else shape
                  for shape in dense_shapes]

  outputs = gen_parsing_ops._parse_example(
      serialized=serialized,
      names=names,
      dense_defaults=dense_defaults_vec,
      sparse_keys=sparse_keys,
      sparse_types=sparse_types,
      dense_keys=dense_keys,
      dense_shapes=dense_shapes,
      name=name)

  (sparse_indices, sparse_values, sparse_shapes, dense_values) = outputs

  sparse_tensors = [ops.SparseTensor(ix, val, shape) for (ix, val, shape)
                    in zip(sparse_indices, sparse_values, sparse_shapes)]

  return dict(
      zip(sparse_keys + dense_keys, sparse_tensors + dense_values))


def parse_single_example(serialized,  # pylint: disable=invalid-name
                         names=None,
                         sparse_keys=None,
                         sparse_types=None,
                         dense_keys=None,
                         dense_types=None,
                         dense_defaults=None,
                         dense_shapes=None,
                         name="ParseSingleExample"):
  """Identical to parse_example but for scalar serialized and names.

  Args:
    serialized: A scalar string, a single serialized Example.
      See parse_example documentation for more details.
    names: (Optional) A scalar string, the associated name.
      See parse_example documentation for more details.
    sparse_keys: See parse_example documentation for more details.
    sparse_types: See parse_example documentation for more details.
    dense_keys: See parse_example documentation for more details.
    dense_types: See parse_example documentation for more details.
    dense_defaults: See parse_example documentation for more details.
    dense_shapes: See parse_example documentation for more details.
    name: Optional op name.

  Returns:
    A dictionary mapping keys to Tensors and SparseTensors.

    For dense tensors, the Tensor is identical to the output of parse_example,
    except it is one less dimension (the first, batch, dimension is removed).

    For SparseTensors:
      The first (batch) column of the indices matrix is removed
        (it is now a column vector).
      The values vector is unchanged.
      The first (batch_size) entry of the shape vector is removed
        (it is now a single element vector).

  Raises:
    ValueError: if "scalar" or "names" have known shapes, and are not scalars.
  """
  with ops.op_scope([serialized], name, "parse_single_example"):
    serialized = ops.convert_to_tensor(serialized)
    serialized_shape = serialized.get_shape()
    if serialized_shape.ndims is not None:
      if serialized_shape.ndims != 0:
        raise ValueError("Input serialized must be a scalar")
    else:
      serialized = control_flow_ops.with_dependencies(
          [logging_ops.Assert(
              math_ops.equal(array_ops.rank(serialized), 0),
              ["Input serialized must be a scalar"],
              name="SerializedIsScalar")],
          serialized,
          name="SerializedDependencies")
    serialized = array_ops.expand_dims(serialized, 0)
    if names is not None:
      names = ops.convert_to_tensor(names)
      names_shape = names.get_shape()
      if names_shape.ndims is not None:
        if names_shape.ndims != 0:
          raise ValueError("Input names must be a scalar")
      else:
        names = control_flow_ops.with_dependencies(
            [logging_ops.Assert(
                math_ops.equal(array_ops.rank(names), 0),
                ["Input names must be a scalar"],
                name="NamesIsScalar")],
            names,
            name="NamesDependencies")
      names = array_ops.expand_dims(names, 0)

    outputs = parse_example(serialized,
                            names=names,
                            sparse_keys=sparse_keys,
                            sparse_types=sparse_types,
                            dense_keys=dense_keys,
                            dense_types=dense_types,
                            dense_defaults=dense_defaults,
                            dense_shapes=dense_shapes,
                            name=name)
    if dense_keys is not None:
      for d in dense_keys:
        outputs[d] = array_ops.squeeze(outputs[d], [0], name="Squeeze_%s" % d)
    if sparse_keys is not None:
      for s in sparse_keys:
        outputs[s] = ops.SparseTensor(
            array_ops.slice(outputs[s].indices,
                            [0, 1], [-1, -1], name="Slice_Indices_%s" % s),
            outputs[s].values,
            array_ops.slice(outputs[s].shape,
                            [1], [-1], name="Squeeze_Shape_%s" % s))
    return outputs


@ops.RegisterShape("ParseExample")
def _ParseExampleShape(op):
  """Shape function for the ParseExample op."""
  input_shape = op.inputs[0].get_shape().with_rank(1)
  num_sparse = op.get_attr("Nsparse")
  num_dense = op.get_attr("Ndense")
  dense_shapes = op.get_attr("dense_shapes")
  sparse_index_shapes = [
      tensor_shape.matrix(None, 2) for _ in range(num_sparse)]
  sparse_value_shapes = [tensor_shape.vector(None) for _ in range(num_sparse)]
  sparse_shape_shapes = [tensor_shape.vector(2) for _ in range(num_sparse)]
  assert num_dense == len(dense_shapes)
  dense_shapes = [
      input_shape.concatenate((d.size for d in dense_shape.dim))
      for dense_shape in dense_shapes]
  return (sparse_index_shapes + sparse_value_shapes + sparse_shape_shapes +
          dense_shapes)


ops.RegisterShape("StringToNumber")(
    common_shapes.unchanged_shape)


@ops.RegisterShape("DecodeRaw")
def _DecodeRawShape(op):
  """Shape function for the DecodeRaw op."""
  # NOTE(mrry): Last dimension is data-dependent.
  return [op.inputs[0].get_shape().concatenate([None])]


@ops.RegisterShape("DecodeCSV")
def _DecodeCSVShape(op):
  """Shape function for the DecodeCSV op."""
  input_shape = op.inputs[0].get_shape()
  # Optionally check that all of other inputs are scalar or empty.
  for default_input in op.inputs[1:]:
    default_input_shape = default_input.get_shape().with_rank(1)
    if default_input_shape[0] > 1:
      raise ValueError(
          "Shape of a default must be a length-0 or length-1 vector.")
  return [input_shape] * len(op.outputs)
