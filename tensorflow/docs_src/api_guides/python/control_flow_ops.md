# Control Flow

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

## Control Flow Operations

TensorFlow provides several operations and classes that you can use to control
the execution of operations and add conditional dependencies to your graph.

*   @{tf.identity}
*   @{tf.tuple}
*   @{tf.group}
*   @{tf.no_op}
*   @{tf.count_up_to}
*   @{tf.cond}
*   @{tf.case}
*   @{tf.while_loop}

## Logical Operators

TensorFlow provides several operations that you can use to add logical operators
to your graph.

*   @{tf.logical_and}
*   @{tf.logical_not}
*   @{tf.logical_or}
*   @{tf.logical_xor}

## Comparison Operators

TensorFlow provides several operations that you can use to add comparison
operators to your graph.

*   @{tf.equal}
*   @{tf.not_equal}
*   @{tf.less}
*   @{tf.less_equal}
*   @{tf.greater}
*   @{tf.greater_equal}
*   @{tf.where}

## Debugging Operations

TensorFlow provides several operations that you can use to validate values and
debug your graph.

*   @{tf.is_finite}
*   @{tf.is_inf}
*   @{tf.is_nan}
*   @{tf.verify_tensor_all_finite}
*   @{tf.check_numerics}
*   @{tf.add_check_numerics_ops}
*   @{tf.Assert}
*   @{tf.Print}
