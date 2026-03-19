# TfLite Variants

Variant tensors in TensorFlow wrap and store arbitrary C++ objects within their
data members. Common usage regards non-trivial and potentially referential
buffer semantics (TensorLists and DataSets being cannonical examples).

This directory contains implementations for these containers
and kernels for the tflite runtime.
Currently tflite supplies performant custom kernels for a subset of
`tf.list_ops`.
Also refer to [variant tensor in the tflite common api](https://github.com/tensorflow/tensorflow/blob/61c76427561a46d03605370fc685d810c1c3e717/tensorflow/lite/core/c/common.h#L1320C9-L1320C9)
, [tensor list legalization](https://github.com/tensorflow/tensorflow/blob/d36fb81bc1ef258d5024b791d61cdd5136ca09af/tensorflow/compiler/mlir/lite/transforms/legalize_tensorlist.cc)
, and [end to end python tests](https://github.com/tensorflow/tensorflow/blob/d36fb81bc1ef258d5024b791d61cdd5136ca09af/tensorflow/lite/kernels/variants/py/end_to_end_test.py)
for example usage.

**api**

* `./py/register_list_ops_py.py` : Bindings for registering ops in python.
* `./list_ops_lib` : Include for tensorlist kernel registrations.
* `./register_list_ops` : Register all kernels with op resolver in C++.

**implementations**

* `./list_kernels/` : Custom `TensorList*` kernels for tflite.
* `./tensor_array` : A variable length array of reference counted `TfLiteTensor`
.

**tests**

* `/list_ops_subgraph_test` : Multi-Op tests through C++ api.
* `/py/end_to_end_test.py` : Tests through python api and compare to `tf.list_ops`.
