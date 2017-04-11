# TensorFlow contrib layers.

## initializers.py

Functions that produce variable initializer functions with signature:

`foo(shape, dtype) : Tensor`

These are typically consumed by functions in [layers.py](#layers.py).

## layers.py {#.py}

Functions that produce layer operations and associated weight & bias variables. Signatures will vary for different functions, but they will often take many of
these arguments.

`foo(x,
     num_outputs,
     …,
     weight_init=<DEFAULT>,
     bias_init=<DEFAULT>,
     weight_collections=(tf.GraphKeys.WEIGHTS,),
     bias_collections=(tf.GraphKeys.BIASES,),
     output_collections=(tf.GraphKeys.ACTIVATIONS,),
     weight_regularizer=None,
     bias_regularizer=None,
     name=None) : Tensor`

`x` is the input tensor.

Weights, biases, and activations (i.e., outputs) are, by default, added to the specified collections. Weights and biases are also added to
`tf.GraphKeys.GLOBAL_VARIABLES` and `tf.GraphKeys.TRAINABLE_VARIABLES`.

## optimizers.py

Functions that add optimization ops given `loss` and `global_step` tensors.

## regularizers.py

Functions that produce weight regularization functions with signature

`foo(weight_vars, name=None) : Operation`

These are typically consumed by functions in [layers.py](#layers.py).

## summaries.py

Functions that add summary ops to the standard `tf.GraphKeys.SUMMARIES`
collection. They also avoid name conflicts in the summary key.
