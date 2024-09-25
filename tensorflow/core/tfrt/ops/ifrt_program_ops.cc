/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
namespace tfrt_stub {

REGISTER_OP("IfrtCall")
    .Input("args: Tin")
    .Output("results: Tout")
    .Attr("Tin: list(type) >= 0")
    .Attr("Tout: list(type) >= 0")
    .Attr("program_id: int")
    .Attr("variable_arg_indices: list(int)")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .Doc(R"(
Calls an IFRT program identified by the given program id.

This op looks up a `ServingExecutable` from `ServingExecutableRegistry` using
the program id, calls the executable with the op's inputs as arguments, and
returns its results as the op's outputs.

Note that this op is not part of a stable interface. Users must not use this op
in their SavedModel and instead rely on Ifrt Serving's mechanism that
automatically inserts this op with graph rewrite.

program_id: int64 id that can be used to look up compiled programs from
ServingExecutableRegistry`.

variable_arg_indices: must be in sorted ascending order. The argument at position
variable_arg_indices[k] in tpu program is already loaded as an ifrt array and
the input `args[variable_arg_indices[k]]` is the key to look for this loaded array.
)");

REGISTER_OP("IfrtLoadVariable")
    .Input("variable: Tin")
    .Output("array_key: Tout")
    .Output("tensor: Tout")
    .Attr("Tin: type")
    .Attr("Tout: type")
    .Attr("used_by_host: bool")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::UnknownShape)
    .Doc(R"(
This op loads a restored variable tensor as a tensor future. It is areplacement of `tf.ReadVariableOp`.

This op returns a scalar string tensor containing the restored variable name, which 
is composed from `container_name` and `shared_name` from a `var_handle` and can be
used as a key within the runtime, as well as a future for the tensor.

Note that this op is not part of a stable interface. Users must not use this op
in their SavedModel and instead rely on Ifrt Serving's mechanism that
automatically inserts this op with graph rewrite.

variable: the variable handle of the variable tensor to be loaded.
array_key: the key to be used to look up the loaded array by the 'IfrtCall' op.
tensor: the future of the loaded tensor. The future contains a valid tensor if `use_by_host` is true.
'used_by_host': a boolean indicating whether the variable is used by the host OP
or excelusively by the TPU.


)");

}  // namespace tfrt_stub
}  // namespace tensorflow
