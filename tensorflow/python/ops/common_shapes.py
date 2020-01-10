"""A library of common shape functions."""
import math

from tensorflow.python.framework import tensor_shape


def scalar_shape(unused_op):
  """Shape function for ops that output a scalar value."""
  return [tensor_shape.scalar()]


def unchanged_shape(op):
  """Shape function for ops that output an tensor like their first input."""
  return [op.inputs[0].get_shape()]


def unchanged_shape_with_rank(rank):
  """Returns a shape function for ops that constrain the rank of their input.

  Args:
    rank: The exact rank of the input and output.

  Returns:
    A shape function for ops that output a tensor of the same size as their
    input, with a particular rank.
  """
  def _ShapeFunction(op):
    return [op.inputs[0].get_shape().with_rank(rank)]
  return _ShapeFunction


def unchanged_shape_with_rank_at_least(rank):
  """Returns a shape function for ops that constrain the rank of their input.

  Args:
    rank: A lower bound on the rank of the input and output.

  Returns:
    A shape function for ops that output a tensor of the same size as their
    input, with a particular rank.
  """
  def _ShapeFunction(op):
    return [op.inputs[0].get_shape().with_rank_at_least(rank)]
  return _ShapeFunction


def unchanged_shape_with_rank_at_most(rank):
  """Returns a shape function for ops that constrain the rank of their input.

  Args:
    rank: An upper bound on the rank of the input and output.

  Returns:
    A shape function for ops that output a tensor of the same size as their
    input, with a particular rank.
  """
  def _ShapeFunction(op):
    return [op.inputs[0].get_shape().with_rank_at_most(rank)]
  return _ShapeFunction


def matmul_shape(op):
  """Shape function for a MatMul op."""
  a_shape = op.inputs[0].get_shape().with_rank(2)
  transpose_a = op.get_attr("transpose_a")
  b_shape = op.inputs[1].get_shape().with_rank(2)
  transpose_b = op.get_attr("transpose_b")
  output_rows = a_shape[1] if transpose_a else a_shape[0]
  output_cols = b_shape[0] if transpose_b else b_shape[1]
  inner_a = a_shape[0] if transpose_a else a_shape[1]
  inner_b = b_shape[1] if transpose_b else b_shape[0]
  inner_a.assert_is_compatible_with(inner_b)
  return [tensor_shape.TensorShape([output_rows, output_cols])]


def bias_add_shape(op):
  """Shape function for a BiasAdd op."""
  input_shape = op.inputs[0].get_shape().with_rank_at_least(2)
  bias_shape = op.inputs[1].get_shape().with_rank(1)
  if input_shape.ndims is not None:
    # Output has the same shape as input, and matches the length of
    # bias in its last dimension.
    output_shape = input_shape[0:-1].concatenate(
        input_shape[-1].merge_with(bias_shape[0]))
  else:
    output_shape = tensor_shape.unknown_shape()
  return [output_shape]


def _Get2DOutputSize(input_height, input_width, filter_height, filter_width,
                     row_stride, col_stride, padding_type):
  """Returns the number of rows and columns in a convolution/pooling output."""
  input_height = tensor_shape.as_dimension(input_height)
  input_width = tensor_shape.as_dimension(input_width)
  filter_height = tensor_shape.as_dimension(filter_height)
  filter_width = tensor_shape.as_dimension(filter_width)
  row_stride = int(row_stride)
  col_stride = int(col_stride)

  if filter_height.value == 1 and filter_width.value == 1 and (
      row_stride == 1 and col_stride == 1):
    return input_height, input_width
  else:
    if filter_height > input_height or filter_width > input_width:
      raise ValueError("filter must not be larger than the input: ",
                       "Filter: [", filter_height, "x", filter_width, "] ",
                       "Input: [", input_height, "x", input_width, "] ")
    if row_stride > filter_height or col_stride > filter_width:
      raise ValueError("stride must be less than or equal to filter size",
                       "stride: [", row_stride, "x", col_stride, "] ",
                       "filter: [", filter_height, "x", filter_width, "] ")

    # Compute number of rows in the output, based on the padding.
    if input_height.value is None or filter_height.value is None:
      out_rows = None
    elif padding_type == "VALID":
      out_rows = int(
          math.ceil((input_height.value - filter_height.value + 1.0)
                    / row_stride))
    elif padding_type == "SAME":
      out_rows = int(math.ceil(input_height.value * 1.0
                               / row_stride))
    else:
      raise ValueError("Invalid value for padding: %r" % padding_type)

    # Compute number of columns in the output, based on the padding.
    if input_width.value is None or filter_width.value is None:
      out_cols = None
    elif padding_type == "VALID":
      out_cols = int(
          math.ceil((input_width.value - filter_width.value + 1.0)
                    / col_stride))
    elif padding_type == "SAME":
      out_cols = int(math.ceil(input_width.value * 1.0 / col_stride))

    return out_rows, out_cols


def conv2d_shape(op):
  """Shape function for a Conv2D op.

  This op has two inputs:

  * input, a 4D tensor with shape = [batch_size, rows, cols, depth_in]
  * filter, a 4D tensor with shape =  [filter_rows, filter_cols,
    depth_in, depth_out]

  The output is a 4D tensor with shape = [batch_size, out_rows,
  out_cols, depth_out], where out_rows and out_cols depend on the
  value of the op's "padding" and "strides" attrs.

  Args:
    op: A Conv2D Operation.

  Returns:
    A list containing the Shape of the Conv2D output.

  Raises:
    ValueError: If the shapes of the input or filter are incompatible.
  """
  input_shape = op.inputs[0].get_shape().with_rank(4)
  filter_shape = op.inputs[1].get_shape().with_rank(4)

  batch_size = input_shape[0]
  in_rows = input_shape[1]
  in_cols = input_shape[2]

  filter_rows = filter_shape[0]
  filter_cols = filter_shape[1]
  depth_out = filter_shape[3]
  # Check that the input depths are compatible.
  input_shape[3].assert_is_compatible_with(filter_shape[2])

  stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")
  if stride_b != 1 or stride_d != 1:
    raise ValueError("Current implementation does not yet support "
                     "strides in the batch and depth dimensions.")
  if stride_r != stride_c:
    # TODO(shlens): Add support for this.
    raise ValueError("Current implementation only supports equal length "
                     "strides in the row and column dimensions.")

  # TODO(mrry,shlens): Raise an error if the stride would cause
  # information in the input to be ignored. This will require a change
  # in the kernel implementation.
  stride = stride_r
  padding = op.get_attr("padding")
  out_rows, out_cols = _Get2DOutputSize(
      in_rows, in_cols, filter_rows, filter_cols, stride, stride, padding)

  return [tensor_shape.TensorShape([batch_size, out_rows, out_cols, depth_out])]


def separable_conv2d_shape(op):
  """Shape function for a SeparableConv2D op.

  This op has three inputs:

  * input, a 4D tensor with shape = [batch_size, rows, cols, depth_in]

  * depthwise_filter, a 4D tensor with shape = [filter_rows,
    filter_cols, depth_in, depth_multiplier]

  * pointwise_filter, a 4D tensor with shape = [1, 1, depth_in *
    depth_multiplier, depth_out]

  The output is a 4D tensor with shape = [batch_size, out_rows,
  out_cols, depth_out], where out_rows and out_cols depend on the
  value of the op's "padding" and "strides" attrs.

  Args:
    op: A SeparableConv2D Operation.

  Returns:
    A list containing the Shape of the SeparableConv2D output.

  Raises:
    ValueError: If the shapes of the input or filter are incompatible.
  """
  input_shape = op.inputs[0].get_shape().with_rank(4)
  depthwise_filter_shape = op.inputs[1].get_shape().merge_with(
      tensor_shape.TensorShape([None, None, input_shape[3], None]))
  pointwise_depth_in = depthwise_filter_shape[2] * depthwise_filter_shape[3]

  pointwise_filter_shape = op.inputs[2].get_shape().merge_with(
      tensor_shape.TensorShape([1, 1, pointwise_depth_in, None]))

  batch_size = input_shape[0]
  in_rows = input_shape[1]
  in_cols = input_shape[2]

  filter_rows = depthwise_filter_shape[0]
  filter_cols = depthwise_filter_shape[1]
  depth_out = pointwise_filter_shape[3]

  stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")
  if stride_b != 1 or stride_d != 1:
    raise ValueError("Current implementation does not yet support "
                     "strides in the batch and depth dimensions.")
  if stride_r != stride_c:
    # TODO(shlens): Add support for this.
    raise ValueError("Current implementation only supports equal length "
                     "strides in the row and column dimensions.")

  # TODO(mrry,shlens): Raise an error if the stride would cause
  # information in the input to be ignored. This will require a change
  # in the kernel implementation.
  stride = stride_r
  padding = op.get_attr("padding")
  out_rows, out_cols = _Get2DOutputSize(
      in_rows, in_cols, filter_rows, filter_cols, stride, stride, padding)

  return [tensor_shape.TensorShape([batch_size, out_rows, out_cols, depth_out])]


def avg_pool_shape(op):
  """Shape function for an AvgPool op.

  This op has one input:

  * input, a 4D tensor with shape = [batch_size, rows, cols, depth]

  The output is a 4D tensor with shape = [batch_size, out_rows,
  out_cols, depth_out], where out_rows and out_cols depend on the
  value of the op's "ksize", "strides", and "padding" attrs.

  Args:
    op: An AvgPool Operation.

  Returns:
    A single-element list containing the Shape of the AvgPool output.

  Raises:
    ValueError: If the shape of the input is invalid or incompatible with
      the values of the attrs.
  """
  input_shape = op.inputs[0].get_shape().with_rank(4)
  ksize_b, ksize_r, ksize_c, ksize_d = op.get_attr("ksize")
  stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")

  batch_size = input_shape[0]
  in_rows = input_shape[1]
  in_cols = input_shape[2]
  depth = input_shape[3]

  if ksize_b != 1 or ksize_d != 1:
    raise ValueError("Current implementation does not support pooling "
                     "in the batch and depth dimensions.")
  if stride_b != 1 or stride_d != 1:
    raise ValueError("Current implementation does not support strides "
                     "in the batch and depth dimensions.")

  # TODO(mrry,shlens): Raise an error if the stride would cause
  # information in the input to be ignored. This will require a change
  # in the kernel implementation.
  padding = op.get_attr("padding")

  out_rows, out_cols = _Get2DOutputSize(
      in_rows, in_cols, ksize_r, ksize_c, stride_r, stride_c, padding)

  return [tensor_shape.TensorShape([batch_size, out_rows, out_cols, depth])]


def max_pool_shape(op):
  """Shape function for a MaxPool op.

  This op has one input:

  * input, a 4D tensor with shape = [batch_size, rows, cols, depth_in]

  The output is a 4D tensor with shape = [batch_size, out_rows,
  out_cols, depth_out], where out_rows, out_cols, and depth_out depend
  on the value of the op's "ksize", "strides", and "padding" attrs.

  Args:
    op: A MaxPool Operation.

  Returns:
    A single-element list containing the Shape of the MaxPool output.

  Raises:
    ValueError: If the shape of the input is invalid or incompatible with
      the values of the attrs.
  """
  input_shape = op.inputs[0].get_shape().with_rank(4)
  ksize_b, ksize_r, ksize_c, ksize_d = op.get_attr("ksize")
  stride_b, stride_r, stride_c, stride_d = op.get_attr("strides")

  batch_size = input_shape[0]
  in_rows = input_shape[1]
  in_cols = input_shape[2]
  depth = input_shape[3]

  if ksize_b != 1:
    raise ValueError("Current implementation does not support pooling "
                     "in the batch dimension.")
  if stride_b != 1:
    raise ValueError("Current implementation does not support strides "
                     "in the batch dimension.")

  if not ((ksize_r == 1 and ksize_c == 1) or ksize_d == 1):
    raise ValueError("MaxPooling supports exactly one of pooling across depth "
                     "or pooling across width/height.")

  # TODO(mrry,shlens): Raise an error if the stride would cause
  # information in the input to be ignored. This will require a change
  # in the kernel implementation.
  if ksize_d == 1:
    padding = op.get_attr("padding")
    out_rows, out_cols = _Get2DOutputSize(
        in_rows, in_cols, ksize_r, ksize_c, stride_r, stride_c, padding)
    return [tensor_shape.TensorShape([batch_size, out_rows, out_cols, depth])]
  else:
    if depth % ksize_d > 0:
      raise ValueError("Depthwise max pooling requires the depth window "
                       "to evenly divide the input depth.")
    if stride_d != ksize_d:
      raise ValueError("Depthwise max pooling requires the depth window "
                       "to equal the depth stride.")
    return [tensor_shape.TensorShape(
        [batch_size, in_rows, in_cols, depth / ksize_d])]


def no_outputs(unused_op):
  """Shape function for use with ops that have no outputs."""
  return []


def unknown_shape(op):
  """Shape function for use with ops whose output shapes are unknown."""
  return [tensor_shape.unknown_shape() for _ in op.outputs]
