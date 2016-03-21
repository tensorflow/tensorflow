/* Copyright 2015 Google Inc. All Rights Reserved.

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

// --------------------------------------------------------------------------
REGISTER_OP("Switch")
    .Input("data: T")
    .Input("pred: bool")
    .Output("output_false: T")
    .Output("output_true: T")
    .Attr("T: type")
    .Doc(R"doc(
Forwards `data` to the output port determined by `pred`.

If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.

See also `RefSwitch` and `Merge`.

data: The tensor to be forwarded to the appropriate output.
pred: A scalar that specifies which output port will receive data.
output_false: If `pred` is false, data will be forwarded to this output.
output_true: If `pred` is true, data will be forwarded to this output.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("RefSwitch")
    .Input("data: Ref(T)")
    .Input("pred: bool")
    .Output("output_false: Ref(T)")
    .Output("output_true: Ref(T)")
    .Attr("T: type")
    .Doc(R"doc(
Forwards the ref tensor `data` to the output port determined by `pred`.

If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
the data goes to `output_false`.

See also `Switch` and `Merge`.

data: The ref tensor to be forwarded to the appropriate output.
pred: A scalar that specifies which output port will receive data.
output_false: If `pred` is false, data will be forwarded to this output.
output_true: If `pred` is true, data will be forwarded to this output.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("RefSelect")
    .Input("index: int32")
    .Input("inputs: Ref(N * T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Doc(R"doc(
Forwards the `index`th element of `inputs` to `output`.

index: A scalar that determines the input that gets selected.
inputs: A list of ref tensors, one of which will be forwarded to `output`.
output: The forwarded tensor.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Merge")
    .Input("inputs: N * T")
    .Output("output: T")
    .Output("value_index: int32")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Doc(R"doc(
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.

`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`.

It is an error if more than one tensor in `inputs` is available.

inputs: The input tensors, exactly one of which will become available.
output: Will be set to the available input tensor.
value_index: The index of the chosen input tensor in `inputs`.
)doc");

REGISTER_OP("RefMerge")
    .Input("inputs: Ref(N * T)")
    .Output("output: Ref(T)")
    .Output("value_index: int32")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Doc(R"doc(
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.

`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`.

It is an error if more than one tensor in `inputs` is available.

inputs: The input tensors, exactly one of which will become available.
output: Will be set to the available input tensor.
value_index: The index of the chosen input tensor in `inputs`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Enter")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("frame_name: string")
    .Attr("is_constant: bool = false")
    .Attr("parallel_iterations: int = 10")
    .Doc(R"doc(
Creates or finds a child frame, and makes `data` available to the child frame.

This op is used together with `Exit` to create loops in the graph.
The unique `frame_name` is used by the `Executor` to identify frames. If
`is_constant` is true, `output` is a constant in the child frame; otherwise
it may be changed in the child frame. At most `parallel_iterations` iterations
are run in parallel in the child frame.

data: The tensor to be made available to the child frame.
frame_name: The name of the child frame.
is_constant: If true, the output is constant within the child frame.
parallel_iterations: The number of iterations allowed to run in parallel.
output: The same tensor as `data`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("RefEnter")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .Attr("frame_name: string")
    .Attr("is_constant: bool = false")
    .Attr("parallel_iterations: int = 10")
    .Doc(R"doc(
Creates or finds a child frame, and makes `data` available to the child frame.

The unique `frame_name` is used by the `Executor` to identify frames. If
`is_constant` is true, `output` is a constant in the child frame; otherwise
it may be changed in the child frame. At most `parallel_iterations` iterations
are run in parallel in the child frame.

data: The tensor to be made available to the child frame.
frame_name: The name of the child frame.
is_constant: If true, the output is constant within the child frame.
parallel_iterations: The number of iterations allowed to run in parallel.
output: The same tensor as `data`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Exit")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"doc(
Exits the current frame to its parent frame.

Exit makes its input `data` available to the parent frame.

data: The tensor to be made available to the parent frame.
output: The same tensor as `data`.
)doc");

REGISTER_OP("RefExit")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .Doc(R"doc(
Exits the current frame to its parent frame.

Exit makes its input `data` available to the parent frame.

data: The tensor to be made available to the parent frame.
output: The same tensor as `data`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("NextIteration")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"doc(
Makes its input available to the next iteration.

data: The tensor to be made available to the next iteration.
output: The same tensor as `data`.
)doc");

REGISTER_OP("RefNextIteration")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .Doc(R"doc(
Makes its input available to the next iteration.

data: The tensor to be made available to the next iteration.
output: The same tensor as `data`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("LoopCond")
    .Input("input: bool")
    .Output("output: bool")
    .Doc(R"doc(
Forwards the input to the output.

This operator represents the loop termination condition used by the
"pivot" switches of a loop.

input: A boolean scalar, representing the branch predicate of the Switch op.
output: The same tensor as `input`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ControlTrigger")
    .Doc(R"doc(
Does nothing. Serves as a control trigger for scheduling. Only useful as a
placeholder for control edges.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("Abort")
    .Attr("error_msg: string = ''")
    .Doc(R"doc(
Raise a exception to abort the process when called.

Returns nothing but an exception.

error_msg: A string which is the message associated with the exception.
)doc");

}  // namespace tensorflow
