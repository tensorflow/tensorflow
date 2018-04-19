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

REGISTER_OP("DirectedInterleaveDataset")
    .Input("selector_input_dataset: variant")
    .Input("data_input_datasets: N * variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
A substitute for `InterleaveDataset` on a fixed list of `N` datasets.

selector_input_dataset: A dataset of scalar `DT_INT64` elements that determines
  which of the `N` data inputs should produce the next output element.
data_input_datasets: `N` datasets with the same type that will be interleaved
  according to the values of `selector_input_dataset`.
)doc");

REGISTER_OP("IgnoreErrorsDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that contains the elements of `input_dataset` ignoring errors.
)doc");

REGISTER_OP("UniqueDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that contains the unique elements of `input_dataset`.
)doc");

REGISTER_OP("IteratorGetDevice")
    .Input("resource: resource")
    .Output("device: string")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Returns the name of the device on which `resource` has been placed.
)doc");

REGISTER_OP("FunctionBufferingResource")
    .Input("string_arg: string")
    .Input("target_device: string")
    .Output("resource: resource")
    .Attr("shared_name: string")
    .Attr("container: string")
    .Attr("f: func")
    .Attr("buffer_size: int")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Creates a resource that fills up a buffer by making function calls.

string_arg: String argument to the function call.
target_device: Target device to execute the function on.
resource: Handle to the resource created.
f: Function to be executed.
buffer_size: Size of the buffer.
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

REGISTER_OP("FunctionBufferingResourceReset")
    .Input("function_buffer_resource: resource")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Resets the FunctionBufferingResource.

function_buffer_resource: The FunctionBufferingResource handle.
)doc");

REGISTER_OP("ThreadPoolDataset")
    .Input("input_dataset: variant")
    .Input("thread_pool: resource")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
Creates a dataset that uses a custom thread pool to compute `input_dataset`.

handle: A resource produced by the ThreadPoolHandle op.
)doc");

REGISTER_OP("ThreadPoolHandle")
    .Output("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Attr("num_threads: int")
    .Attr("display_name: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Doc(R"doc(
Creates a custom thread pool with the given number of threads.

handle: A resource that can be consumed by one or more ThreadPoolDataset ops.
num_threads: The number of threads in the thread pool.
display_name: A human-readable name for the threads that may be visible in
  some visualizations.
)doc");

}  // namespace tensorflow
