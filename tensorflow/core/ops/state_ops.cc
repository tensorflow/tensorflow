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

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("VariableV2")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("Variable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      PartialTensorShape shape;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &shape));

      // Variable has legacy behavior where we cannot tell the difference
      // between a scalar shape attribute and 'unknown shape'.  So if the shape
      // is a scalar, we return an unknown shape.
      if (shape.dims() <= 0) {
        return shape_inference::UnknownShape(c);
      }

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(shape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("IsVariableInitialized")
    .Input("ref: Ref(dtype)")
    .Output("is_initialized: bool")
    .Attr("dtype: type")
    .SetAllowsUninitializedInput()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TemporaryVariable")
    .Output("ref: Ref(dtype)")
    .Attr("shape: shape")
    .Attr("dtype: type")
    .Attr("var_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("DestroyTemporaryVariable")
    .Input("ref: Ref(T)")
    .Output("value: T")
    .Attr("T: type")
    .Attr("var_name: string")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Assign")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("validate_shape: bool = true")
    .Attr("use_locking: bool = true")
    .SetAllowsUninitializedInput()
    .SetShapeFn([](InferenceContext* c) {
      bool validate_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("validate_shape", &validate_shape));
      if (validate_shape) {
        return shape_inference::MergeBothInputsShapeFn(c);
      }

      c->set_output(0, c->input(1));
      return Status::OK();
    });

REGISTER_OP("AssignAdd")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

REGISTER_OP("AssignSub")
    .Input("ref: Ref(T)")
    .Input("value: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::MergeBothInputsShapeFn);

namespace {

Status ScatterUpdateShape(InferenceContext* c) {
  ShapeHandle var_shape = c->input(0);
  ShapeHandle indices_shape = c->input(1);

  ShapeHandle unused_updates_shape;
  ShapeHandle concat;
  ShapeHandle var_subshape;
  TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
  TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
  TF_RETURN_IF_ERROR(
      InferenceContext::Rank(c->input(2)) == 0
          ? Status::OK()
          : c->Merge(c->input(2), concat, &unused_updates_shape));

  c->set_output(0, var_shape);
  return Status::OK();
}

}  // namespace

REGISTER_OP("ScatterUpdate")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterAdd")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterSub")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterMul")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterDiv")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterMin")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: {half, bfloat16, float, double, int32, int64}")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterMax")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: {half, bfloat16, float, double, int32, int64}")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(ScatterUpdateShape);

REGISTER_OP("ScatterNdUpdate")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdUpdate")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdAdd")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdSub")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdMin")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("ResourceScatterNdMax")
    .Input("ref: resource")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = true")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("ScatterNdAdd")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("ScatterNdSub")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("ScatterNdMax")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("ScatterNdMin")
    .Input("ref: Ref(T)")
    .Input("indices: Tindices")
    .Input("updates: T")
    .Output("output_ref: Ref(T)")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32, int64}")
    .Attr("use_locking: bool = false")
    .SetShapeFn(shape_inference::ScatterNdUpdateShape);

REGISTER_OP("CountUpTo")
    .Input("ref: Ref(T)")
    .Output("output: T")
    .Attr("limit: int")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &output));
      c->set_output(0, output);
      return Status::OK();
    });

REGISTER_OP("ResourceCountUpTo")
    .Input("resource: resource")
    .Output("output: T")
    .Attr("limit: int")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data == nullptr || handle_data->empty()) {
        return errors::InvalidArgument("Handle has no shape/type information.");
      }
      shape_inference::ShapeAndType shape_and_type = (*handle_data)[0];
      DataType value_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &value_dtype));
      if (value_dtype != shape_and_type.dtype) {
        return errors::InvalidArgument(
            "Data types do not match: ", DataTypeString(value_dtype), " and ",
            DataTypeString(shape_and_type.dtype));
      }
      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->WithRank(shape_and_type.shape, 0, &output));
      c->set_output(0, output);
      return Status::OK();
    });

}  // namespace tensorflow
