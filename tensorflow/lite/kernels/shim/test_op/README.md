This directory contains a fake operation in order to demonstrate and test the
interfaces.

First test op `SimpleOp` which is an op that test various attributes and input
and output types. The other one is `TmplOp` which tests a templatized kernel.

The contents:

## `simple_op.h|cc`, `tmpl_op.h|cc`

This is where the actual implementation of this op resides

## `simple_tf_op.cc`, `tmpl_tf_op.cc`

The TF op definition.

## `simple_tflite_op.h|cc`

The TFLite op definition.
