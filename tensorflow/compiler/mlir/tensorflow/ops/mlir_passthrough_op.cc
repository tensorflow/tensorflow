/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("MlirPassthroughOp")
    .Attr("mlir_module: string")
    .Attr("Tinputs : list(type) >= 0")
    .Input("inputs: Tinputs")
    .Attr("Toutputs : list(type) >= 0")
    .Output("outputs: Toutputs")
    .Doc(R"doc(
Wraps an arbitrary MLIR computation expressed as a module with a main() function.

This operation does not have an associated kernel and is not intended to be
executed in a regular TensorFlow session. Instead it is intended to be used for
testing or for special case where a user intends to pass custom MLIR computation
through a TensorFlow graph with the intent of having custom tooling processing
it downstream (when targeting a different environment, like TensorFlow lite for
example).
The MLIR module is expected to have a main() function that will be used as an
entry point. The inputs to the operations will be passed as argument to the
main() function and the returned values of the main function mapped to the
outputs.
Example usage:

```
import tensorflow as tf
from tensorflow.compiler.mlir.tensorflow.gen_mlir_passthrough_op import mlir_passthrough_op

mlir_module = '''
func @main(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10x10xf32> {
   %add = "magic.op"(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
   return %ret : tensor<10x10xf32>
}
'''

@tf.function
def foo(x, y):
  return = mlir_passthrough_op([x, y], mlir_module, Toutputs=[tf.float32])

graph_def = foo.get_concrete_function(tf.TensorSpec([10], tf.float32), tf.TensorSpec([10], tf.float32)).graph.as_graph_def()
```
)doc");

}  // namespace tensorflow
