/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"

// Please use the appropriate namespace for your project
namespace tensorflow {
namespace custom_op_examples {

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeAndType;
using ::tensorflow::shape_inference::ShapeHandle;

Status ScalarOutput(InferenceContext* c) {
  c->set_output(0, c->Scalar());
  return OkStatus();
}

Status TwoScalarInputs(InferenceContext* c) {
  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
  return OkStatus();
}

Status TwoScalarInputsScalarOutput(InferenceContext* c) {
  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
  return ScalarOutput(c);
}

Status ThreeScalarInputs(InferenceContext* c) {
  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &handle));
  return OkStatus();
}

Status ThreeScalarInputsScalarOutput(InferenceContext* c) {
  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &handle));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &handle));
  return ScalarOutput(c);
}

Status ValidateTableType(InferenceContext* c,
                         const ShapeAndType& key_shape_and_type,
                         const string& key_dtype_attr,
                         const ShapeAndType& value_shape_and_type,
                         const string& value_dtype_attr) {
  DataType key_dtype;
  TF_RETURN_IF_ERROR(c->GetAttr(key_dtype_attr, &key_dtype));
  if (key_shape_and_type.dtype != key_dtype) {
    return errors::InvalidArgument(
        "Trying to read value with wrong dtype. "
        "Expected ",
        DataTypeString(key_shape_and_type.dtype), " got ",
        DataTypeString(key_dtype));
  }
  DataType value_dtype;
  TF_RETURN_IF_ERROR(c->GetAttr(value_dtype_attr, &value_dtype));
  if (value_shape_and_type.dtype != value_dtype) {
    return errors::InvalidArgument(
        "Trying to read value with wrong dtype. "
        "Expected ",
        DataTypeString(value_shape_and_type.dtype), " got ",
        DataTypeString(value_dtype));
  }
  return OkStatus();
}

Status ExportShapeFunction(InferenceContext* c) {
  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data != nullptr && handle_data->size() == 2) {
    const ShapeAndType& key_shape_and_type = (*handle_data)[0];
    const ShapeAndType& value_shape_and_type = (*handle_data)[1];
    TF_RETURN_IF_ERROR(ValidateTableType(c, key_shape_and_type,
                                         /*key_dtype_attr*/ "key_dtype",
                                         value_shape_and_type,
                                         /*value_dtype_attr*/ "value_dtype"));
  }
  // Different lookup tables have different output shapes.
  c->set_output(0, c->UnknownShape());
  c->set_output(1, c->UnknownShape());
  return OkStatus();
}

Status ImportShapeFunction(InferenceContext* c) {
  ShapeHandle handle;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &handle));

  ShapeHandle keys;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &keys));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(
      c->Merge(c->Dim(keys, 0), c->Dim(c->input(2), 0), &unused));
  return OkStatus();
}

// Note that if an op has any Input or Output of type "resource", it
// is automatically marked as stateful so there is no need to explicitly
// use "SetIsStateful()".
// (See FinalizeInputOrOutput in core/framework/op_def_builder.cc.)

REGISTER_OP("Examples>SimpleHashTableCreate")
    .Output("output: resource")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ScalarOutput);

REGISTER_OP("Examples>SimpleHashTableFind")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Input("default_value: value_dtype")
    .Output("value: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ThreeScalarInputsScalarOutput);

REGISTER_OP("Examples>SimpleHashTableInsert")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Input("value: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ThreeScalarInputs);

REGISTER_OP("Examples>SimpleHashTableRemove")
    .Input("resource_handle: resource")
    .Input("key: key_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(TwoScalarInputs);

REGISTER_OP("Examples>SimpleHashTableExport")
    .Input("table_handle: resource")
    .Output("keys: key_dtype")
    .Output("values: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ExportShapeFunction);

REGISTER_OP("Examples>SimpleHashTableImport")
    .Input("table_handle: resource")
    .Input("keys: key_dtype")
    .Input("values: value_dtype")
    .Attr("key_dtype: type")
    .Attr("value_dtype: type")
    .SetShapeFn(ImportShapeFunction);

}  // namespace custom_op_examples
}  // namespace tensorflow
