This directory contains a fake operation in order to demonstrate and test the
interfaces.

The contents:

## `simple_op.h|cc`

This is where the actual implementation of this op resides

## `simple_tf_op.cc`

The TF op definition which uses `simple_op`.

## `simple_tflite_op.h|cc`

The TFLite op definition which uses `simple_op`.
