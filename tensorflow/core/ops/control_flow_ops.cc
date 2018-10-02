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
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data != nullptr) {
    c->set_output_handle_shapes_and_types(0, *handle_data);
    c->set_output_handle_shapes_and_types(1, *handle_data);
  }
  return Status::OK();
}
}  // namespace

REGISTER_OP("Switch")
    .Input("data: T")
    .Input("pred: bool")
    .Output("output_false: T")
    .Output("output_true: T")
    .Attr("T: type")
    .SetShapeFn(SwitchShape);

REGISTER_OP("RefSwitch")
    .Input("data: Ref(T)")
    .Input("pred: bool")
    .Output("output_false: Ref(T)")
    .Output("output_true: Ref(T)")
    .Attr("T: type")
    .SetAllowsUninitializedInput()
    .SetShapeFn(SwitchShape);

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
    });

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
    .SetShapeFn(MergeShape);

REGISTER_OP("RefMerge")
    .Input("inputs: Ref(N * T)")
    .Output("output: Ref(T)")
    .Output("value_index: int32")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .SetShapeFn(MergeShape);

// --------------------------------------------------------------------------
REGISTER_OP("Enter")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("frame_name: string")
    .Attr("is_constant: bool = false")
    .Attr("parallel_iterations: int = 10")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());

      // Handle resource shape / dtype, if present.
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      // Propagate shape if output is a constant.
      bool is_constant;
      TF_RETURN_IF_ERROR(c->GetAttr("is_constant", &is_constant));
      if (is_constant) {
        c->set_output(0, c->input(0));
      }

      return Status::OK();
    });

// --------------------------------------------------------------------------
REGISTER_OP("RefEnter")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .Attr("frame_name: string")
    .Attr("is_constant: bool = false")
    .Attr("parallel_iterations: int = 10")
    .SetShapeFn(shape_inference::UnchangedShape);

// --------------------------------------------------------------------------
REGISTER_OP("Exit")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RefExit")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

// --------------------------------------------------------------------------
REGISTER_OP("NextIteration")
    .Input("data: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RefNextIteration")
    .Input("data: Ref(T)")
    .Output("output: Ref(T)")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnchangedShape);

// --------------------------------------------------------------------------
REGISTER_OP("LoopCond")
    .Input("input: bool")
    .Output("output: bool")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 0);
    });

// --------------------------------------------------------------------------
REGISTER_OP("ControlTrigger").SetShapeFn(shape_inference::NoOutputs);

// --------------------------------------------------------------------------
REGISTER_OP("Abort")
    .Attr("error_msg: string = ''")
    .Attr("exit_without_error: bool = false")
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow
