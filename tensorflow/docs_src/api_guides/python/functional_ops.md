# Higher Order Functions

Note: Functions taking `Tensor` arguments can also take anything accepted by
@{tf.convert_to_tensor}.

[TOC]

Functional operations.

## Higher Order Operators

TensorFlow provides several higher order operators to simplify the common
map-reduce programming patterns.

*   @{tf.map_fn}
*   @{tf.foldl}
*   @{tf.foldr}
*   @{tf.scan}
