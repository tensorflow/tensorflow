// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;

namespace tensorflow {

namespace {

Status ValidateVariableResourceHandle(InferenceContext* c,
                                      ShapeAndType* shape_and_type) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    shape_and_type->shape = c->UnknownShape();
    shape_and_type->dtype = DT_INVALID;
  } else {
    *shape_and_type = (*handle_data)[0];
    DataType value_dtype;
    TF_RETURN_IF_ERROR(c->GetAttr("dtype", &value_dtype));
    if (shape_and_type->dtype != value_dtype) {
      return errors::InvalidArgument(
          "Trying to read variable with wrong dtype. "
          "Expected ",
          DataTypeString(shape_and_type->dtype), " got ",
          DataTypeString(value_dtype));
    }
  }
  return Status::OK();
}

Status ReadVariableShapeFn(InferenceContext* c) {
  ShapeAndType shape_and_type;
  TF_RETURN_IF_ERROR(ValidateVariableResourceHandle(c, &shape_and_type));
  c->set_output(0, shape_and_type.shape);
  return Status::OK();
}

}  // namespace

REGISTER_OP("VarHandleOp")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("dtype: type")
    .Attr("shape: shape")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("dtype", &t));
      PartialTensorShape p;
      TF_RETURN_IF_ERROR(c->GetAttr("shape", &p));
      ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(p, &s));
      c->set_output_handle_shapes_and_types(0,
                                            std::vector<ShapeAndType>{{s, t}});

      return Status::OK();
    });

REGISTER_OP("ReadVariableOp")
    .Input("resource: resource")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(ReadVariableShapeFn);

REGISTER_OP("DestroyResourceOp")
    .Input("resource: resource")
    .Attr("ignore_lookup_error: bool = true")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

Status CreateAssignShapeFn(InferenceContext* c) {
  ShapeAndType handle_shape_and_type;
  TF_RETURN_IF_ERROR(ValidateVariableResourceHandle(c, &handle_shape_and_type));

  ShapeHandle value_shape = c->input(1);
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(handle_shape_and_type.shape, value_shape, &unused));
  return Status::OK();
}

REGISTER_OP("AssignVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn);

REGISTER_OP("AssignAddVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn);

REGISTER_OP("AssignSubVariableOp")
    .Input("resource: resource")
    .Input("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn(CreateAssignShapeFn);

REGISTER_OP("VarIsInitializedOp")
    .Input("resource: resource")
    .Output("is_initialized: bool")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

Status VariableShapeShapeFn(InferenceContext* c) {
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data == nullptr || handle_data->empty()) {
    return errors::InvalidArgument("Handle doesn't have shape information.");
  }
  ShapeHandle var_shape = (*handle_data)[0].shape;
  int64 rank = c->RankKnown(var_shape) ? c->Rank(var_shape)
                                       : InferenceContext::kUnknownDim;
  c->set_output(0, c->Vector(rank));
  return Status::OK();
}

REGISTER_OP("VariableShape")
    .Input("input: resource")
    .Output("output: out_type")
    .Attr("out_type: {int32, int64} = DT_INT32")
    .SetShapeFn(VariableShapeShapeFn);

REGISTER_OP("ResourceGather")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Attr("validate_indices: bool = true")
    .Output("output: dtype")
    .Attr("dtype: type")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));

      ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRankAtLeast(handle_shape_and_type.shape, 1, &unused));
      ShapeHandle params_subshape;
      TF_RETURN_IF_ERROR(
          c->Subshape(handle_shape_and_type.shape, 1, &params_subshape));
      ShapeHandle indices_shape = c->input(1);
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, params_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("ResourceScatterAdd")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));
      ShapeHandle var_shape = handle_shape_and_type.shape;
      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused_updates_shape;
      ShapeHandle concat;
      ShapeHandle var_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));
      return Status::OK();
    });

REGISTER_OP("ResourceScatterUpdate")
    .Input("resource: resource")
    .Input("indices: Tindices")
    .Input("updates: dtype")
    .Attr("dtype: numbertype")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType handle_shape_and_type;
      TF_RETURN_IF_ERROR(
          ValidateVariableResourceHandle(c, &handle_shape_and_type));
      ShapeHandle var_shape = handle_shape_and_type.shape;
      ShapeHandle indices_shape = c->input(1);

      ShapeHandle unused_updates_shape;
      ShapeHandle concat;
      ShapeHandle var_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(var_shape, 1, &var_subshape));
      TF_RETURN_IF_ERROR(c->Concatenate(indices_shape, var_subshape, &concat));
      TF_RETURN_IF_ERROR(c->Merge(c->input(2), concat, &unused_updates_shape));
      return Status::OK();
    });

REGISTER_OP("CriticalSectionOp")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Output("resource: resource")
    .SetIsStateful()
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("ExecuteInCriticalSection")
    .Input("critical_section: resource")
    .Input("arguments: Targuments")
    .Output("outputs: output_types")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 0")
    .Attr("output_shapes: list(shape) >= 0")
    .SetShapeFn([](InferenceContext* c) {
      std::vector<PartialTensorShape> output_shapes;
      TF_RETURN_IF_ERROR(c->GetAttr("output_shapes", &output_shapes));
      for (int i = 0; i < output_shapes.size(); ++i) {
        ShapeHandle s;
        TF_RETURN_IF_ERROR(
            c->MakeShapeFromPartialTensorShape(output_shapes[i], &s));
        c->set_output(i, s);
      }
      return Status::OK();
    });

}  // namespace tensorflow
