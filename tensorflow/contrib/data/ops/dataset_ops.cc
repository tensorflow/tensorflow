/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("IgnoreErrorsDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that contains the elements of `input_dataset` ignoring errors.
)doc");

REGISTER_OP("FunctionBufferingResource")
    .Input("string_arg: string")
    .Input("target_device: string")
    .Output("resource: resource")
    .Attr("shared_name: string")
    .Attr("container: string")
    .Attr("f: func")
    .Attr("buffer_size: int")
    .Attr("thread_pool_size: int")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Creates a resource that fills up a buffer by making function calls.

string_arg: String argument to the function call.
target_device: Target device to execute the function on.
resource: Handle to the resource created.
f: Function to be executed.
buffer_size: Size of the buffer.
thread_pool_size: Size of the threadpool doing the prefetching.
container: If non-empty, this resource is placed in the given container.
  Otherwise, a default container is used.
shared_name: If non-empty, this resource will be shared under the given name
  across multiple sessions.
)doc");

REGISTER_OP("FunctionBufferingResourceGetNext")
    .Input("function_buffer_resource: resource")
    .Attr("output_types: list(type)")
    .Output("output: output_types")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Gets the next element from a FunctionBufferingResource.

function_buffer_resource: The FunctionBufferingResource handle.
output: A list of return values.
output_types: The type list for the return values.
)doc");

}  // namespace tensorflow
