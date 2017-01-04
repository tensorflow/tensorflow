/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// --------------------------------------------------------------------------
namespace {
Status SwitchShape(InferenceContext* c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
  ShapeHandle out = c->input(0);
  c->set_output(0, out);
  c->set_output(1, out);

  // Handle resource shape / dtype.
  c->set_output_handle_shape(0, c->input_handle_shape(0));
  c->set_output_handle_shape(1, c->input_handle_shape(0));
  c->set_output_handle_dtype(0, c->input_handle_dtype(0));
  c->set_output_handle_dtype(1, c->input_handle_dtype(0));
  return Status::OK();
}
}  // namespace

REGISTER_OP("Switch")
    .Input("data: T")
    .Input("pred: bool")
    .Output("output_false: T")
    .Output("output_true: T")
    .Attr("T: type")
    .SetShapeFn(SwitchShape)
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

REGISTER_OP("RefSwitch")
    .Input("data: Ref(T)")
    .Input("pred: bool")
    .Output("output_false: Ref(T)")
    .Output("output_true: Ref(T)")
    .Attr("T: type")
    .SetAllowsUninitializedInput()
    .SetShapeFn(SwitchShape)
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      ShapeHandle first_input = c->input(1);
      if (!c->FullyDefined(first_input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }
      // If any inputs aren't fully defined or don't match, we return unknown.
      for (int i = 2; i < c->num_inputs(); ++i) {
        ShapeHandle input = c->input(i);
        if (!c->FullyDefined(input) ||
            !c->Merge(first_input, input, &unused).ok()) {
          c->set_output(0, c->UnknownShape());
          return Status::OK();
        }
      }
      c->set_output(0, first_input);
      return Status::OK();
    })
    .Doc(R"doc(
Forwards the `index`th element of `inputs` to `output`.

index: A scalar that determines the input that gets selected.
inputs: A list of ref tensors, one of which will be forwarded to `output`.
output: The forwarded tensor.
)doc");

// --------------------------------------------------------------------------
namespace {
Status MergeShape(InferenceContext* c) {
  ShapeHandle out = c->input(0);
  if (!c->RankKnown(out)) {
    out = c->UnknownShape();
  } else {
    int32 rank = c->Rank(out);
    for (int i = 1; i < c->num_inputs(); ++i) {
      ShapeHandle input = c->input(i);
      if (!c->RankKnown(input) || c->Rank(input) != rank) {
        out = c->UnknownShape();
        break;
      }

      for (int d = 0; d < rank; ++d) {
        if (c->Value(c->Dim(input, d)) != c->Value(c->Dim(out, d))) {
          TF_RETURN_IF_ERROR(c->ReplaceDim(out, d, c->UnknownDim(), &out));
        }
      }
    }
  }
  c->set_output(0, out);
  c->set_output(1, c->Scalar());
  return Status::OK();
}
}  // namespace

REGISTER_OP("Merge")
    .Input("inputs: N * T")
    .Output("output: T")
    .Output("value_index: int32")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .SetShapeFn(MergeShape)
    .Doc(R"doc(
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.

`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`.

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
    .SetShapeFn(MergeShape)
    .Doc(R"doc(
Forwards the value of an available tensor from `inputs` to `output`.

`Merge` waits for at least one of the tensors in `inputs` to become available.
It is usually combined with `Switch` to implement branching.

`Merge` forwards the first tensor for become available to `output`, and sets
`value_index` to its index in `inputs`.

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
    .SetShapeFn(shape_inference::UnknownShape)
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
    .SetShapeFn(shape_inference::UnchangedShape)
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
    .SetShapeFn(shape_inference::UnchangedShape)
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
    .SetShapeFn(shape_inference::UnchangedShape)
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
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Makes its input available to the next iteration.

data: The tensor to be made available to the next iteration.
output: The same tensor as `data`.
)doc");

REGISTER_OP("RefNextIteration")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Makes its input available to the next iteration.

data: The tensor to be made available to the next iteration.
output: The same tensor as `data`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("LoopCond")
    .Input("input: bool")
    .Output("output: bool")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 0);
    })
    .Doc(R"doc(
Forwards the input to the output.

This operator represents the loop termination condition used by the
"pivot" switches of a loop.

input: A boolean scalar, representing the branch predicate of the Switch op.
output: The same tensor as `input`.
)doc");

// --------------------------------------------------------------------------
REGISTER_OP("ControlTrigger")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"docstring(
Does nothing. Serves as a control trigger for scheduling.

Only useful as a placeholder for control edges.
)docstring");

// --------------------------------------------------------------------------
REGISTER_OP("Abort")
    .Attr("error_msg: string = ''")
    .Attr("exit_without_error: bool = false")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Raise a exception to abort the process when called. If exit_without_error is true, the process will exit normally, otherwise it will exit with a SIGABORT signal.

Returns nothing but an exception.

error_msg: A string which is the message associated with the exception.
)doc");

}  // namespace tensorflow
