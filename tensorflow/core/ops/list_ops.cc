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
namespace {

// Verifies that `shapes_and_types` is a valid list handle and has the right
// dtype.
Status VerifyHandleData(
    shape_inference::InferenceContext* c,
    const std::vector<shape_inference::ShapeAndType>& shapes_and_types,
    DataType element_dtype) {
  if (shapes_and_types.size() != 1) {
    return errors::InvalidArgument(
        "Invalid handle_data for input list. Expected length of "
        "shape_and_types: ",
        1, " Saw: ", shapes_and_types.size());
  }
  const shape_inference::ShapeAndType& list_shape_type = shapes_and_types[0];
  if (list_shape_type.dtype != element_dtype) {
    return errors::InvalidArgument("Expected list with element dtype ",
                                   DataTypeString(element_dtype),
                                   " but got list with element dtype ",
                                   DataTypeString(list_shape_type.dtype));
  }
  return Status::OK();
}

bool IsValidTensorListHandleData(
    const std::vector<shape_inference::ShapeAndType>* handle_data) {
  return handle_data != nullptr && handle_data->size() == 1;
}

// Assumes that the handle_data is valid.
shape_inference::ShapeHandle GetElementShapeFromHandleData(
    const std::vector<shape_inference::ShapeAndType>& shapes_and_types) {
  return shapes_and_types[0].shape;
}

REGISTER_OP("EmptyTensorList")
    .Input("element_shape: shape_type")
    .Input("max_num_elements: int32")
    .Output("handle: variant")
    .Attr("element_dtype: type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          0, &element_shape));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{
                 {element_shape, element_dtype}});
      return Status::OK();
    });

REGISTER_OP("TensorListPushBack")
    .Input("input_handle: variant")
    .Input("tensor: element_dtype")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle element_shape = c->UnknownShape();

      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && handle_data->size() > 1) {
        return errors::InvalidArgument(
            "Trying to push to list with wrong variant data.");
      }
      if (IsValidTensorListHandleData(handle_data)) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        if (list_shape_type.dtype != element_dtype) {
          return errors::InvalidArgument(
              "Trying to push to list with wrong element dtype. List has type ",
              DataTypeString(list_shape_type.dtype),
              " but trying to push element with type ",
              DataTypeString(element_dtype));
        }
        shape_inference::ShapeHandle ignored;
        TF_RETURN_IF_ERROR(
            c->Merge(element_shape, list_shape_type.shape, &ignored));
        element_shape = list_shape_type.shape;
      }
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{
                 {element_shape, element_dtype}});
      return Status::OK();
    });

REGISTER_OP("TensorListPushBackBatch")
    .Input("input_handles: variant")
    .Input("tensor: element_dtype")
    .Output("output_handles: variant")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_handles;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_handles));

      shape_inference::ShapeHandle tensor;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &tensor));

      TF_RETURN_IF_ERROR(
          c->MergePrefix(tensor, input_handles, &tensor, &input_handles));

      c->set_output(0, input_handles);

      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle element_shape = c->UnknownShape();

      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && handle_data->size() > 1) {
        return errors::InvalidArgument(
            "Trying to push to list with wrong variant data.");
      }
      if (IsValidTensorListHandleData(handle_data)) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        if (list_shape_type.dtype != element_dtype) {
          return errors::InvalidArgument(
              "Trying to push to list with wrong element dtype. List has type ",
              DataTypeString(list_shape_type.dtype),
              " but trying to push element with type ",
              DataTypeString(element_dtype));
        }
        shape_inference::ShapeHandle ignored;
        TF_RETURN_IF_ERROR(
            c->Merge(element_shape, list_shape_type.shape, &ignored));
        element_shape = list_shape_type.shape;
      }
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{
                 {element_shape, element_dtype}});
      return Status::OK();
    });

REGISTER_OP("TensorListLength")
    .Input("input_handle: variant")
    .Output("length: int32")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TensorListPopBack")
    .Input("input_handle: variant")
    .Input("element_shape: int32")
    .Output("output_handle: variant")
    .Output("tensor: element_dtype")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle tensor_shape = c->UnknownShape();
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && handle_data->size() > 1) {
        return errors::InvalidArgument(
            "Trying to read from list with invalid variant data.");
      }
      if (IsValidTensorListHandleData(handle_data)) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        if (list_shape_type.dtype != element_dtype) {
          return errors::InvalidArgument(
              "Trying to read from list with wrong element dtype. List has "
              "type ",
              DataTypeString(list_shape_type.dtype),
              " but trying to push element with type ",
              DataTypeString(element_dtype));
        }
        shape_inference::ShapeHandle ignored;
        TF_RETURN_IF_ERROR(
            c->Merge(tensor_shape, list_shape_type.shape, &ignored));
        c->set_output_handle_shapes_and_types(0, *handle_data);
        tensor_shape = list_shape_type.shape;
      }
      c->set_output(1, tensor_shape);
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("TensorListStack")
    .Input("input_handle: variant")
    .Input("element_shape: int32")
    .Output("tensor: element_dtype")
    .Attr("element_dtype: type")
    .Attr("num_elements: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle element_shape = c->UnknownShape();
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && handle_data->size() > 1) {
        return errors::InvalidArgument(
            "Trying to read from list with wrong variant data.");
      }
      if (IsValidTensorListHandleData(handle_data)) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        if (list_shape_type.dtype != element_dtype) {
          return errors::InvalidArgument(
              "Trying to read from list with wrong element dtype. List has "
              "type ",
              DataTypeString(list_shape_type.dtype), " but expected type ",
              DataTypeString(element_dtype));
        }
        shape_inference::ShapeHandle ignored;
        TF_RETURN_IF_ERROR(
            c->Merge(element_shape, list_shape_type.shape, &ignored));
        element_shape = list_shape_type.shape;
      }
      shape_inference::ShapeHandle element_shape_input = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          1, &element_shape_input));
      TF_RETURN_IF_ERROR(
          c->Merge(element_shape, element_shape_input, &element_shape));
      int expected_num_elements = -1;
      TF_RETURN_IF_ERROR(c->GetAttr("num_elements", &expected_num_elements));
      shape_inference::ShapeHandle num_elements;
      if (expected_num_elements == -1) {
        num_elements = c->MakeShape({c->UnknownDim()});
      } else {
        num_elements = c->MakeShape({expected_num_elements});
      }
      shape_inference::ShapeHandle result;
      TF_RETURN_IF_ERROR(c->Concatenate(num_elements, element_shape, &result));
      c->set_output(0, result);
      return Status::OK();
    });

Status TensorListConcatShapeInference(
    shape_inference::InferenceContext* c,
    shape_inference::ShapeHandle element_shape) {
  DataType element_dtype;
  TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
  auto* handle_data = c->input_handle_shapes_and_types(0);
  if (handle_data != nullptr && handle_data->size() > 1) {
    return errors::InvalidArgument(
        "Trying to read from list with wrong variant data.");
  }
  if (IsValidTensorListHandleData(handle_data)) {
    const shape_inference::ShapeAndType& list_shape_type = (*handle_data)[0];
    if (list_shape_type.dtype != element_dtype) {
      return errors::InvalidArgument(
          "Trying to read from list with wrong element dtype. List has "
          "type ",
          DataTypeString(list_shape_type.dtype), " but expected type ",
          DataTypeString(element_dtype));
    }
    shape_inference::ShapeHandle merged;
    TF_RETURN_IF_ERROR(c->Merge(element_shape, list_shape_type.shape, &merged));
    element_shape = merged;
  }
  if (c->RankKnown(element_shape)) {
    shape_inference::ShapeHandle result;
    TF_RETURN_IF_ERROR(c->Subshape(element_shape, 1, &result));
    TF_RETURN_IF_ERROR(
        c->Concatenate(c->MakeShape({c->UnknownDim()}), result, &result));
    c->set_output(0, result);
  } else {
    c->set_output(0, c->UnknownShape());
  }
  c->set_output(1, c->MakeShape({c->UnknownDim()}));
  return Status::OK();
}

REGISTER_OP("TensorListConcat")
    .Input("input_handle: variant")
    .Output("tensor: element_dtype")
    .Output("lengths: int64")
    .Attr("element_dtype: type")
    .Attr("element_shape: shape = { unknown_rank: true }")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      PartialTensorShape raw_element_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("element_shape", &raw_element_shape));
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromPartialTensorShape(raw_element_shape,
                                                            &element_shape));
      return TensorListConcatShapeInference(c, element_shape);
    });

REGISTER_OP("TensorListConcatV2")
    .Input("input_handle: variant")
    .Input("element_shape: shape_type")
    .Input("leading_dims: int64")
    .Output("tensor: element_dtype")
    .Output("lengths: int64")
    .Attr("element_dtype: type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          1, &element_shape));
      return TensorListConcatShapeInference(c, element_shape);
    });

REGISTER_OP("TensorListSplit")
    .Input("tensor: element_dtype")
    .Input("element_shape: shape_type")
    .Input("lengths: int64")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle tensor_shape = c->input(0);
      shape_inference::ShapeHandle ignored;
      // Check that tensor is at least a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(tensor_shape, 1, &ignored));
      // Check that lengths is a vector.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &ignored));
      shape_inference::ShapeHandle element_shape_from_tensor_shape;
      TF_RETURN_IF_ERROR(
          c->Subshape(tensor_shape, 1, &element_shape_from_tensor_shape));
      TF_RETURN_IF_ERROR(c->Concatenate(c->MakeShape({c->UnknownDim()}),
                                        element_shape_from_tensor_shape,
                                        &element_shape_from_tensor_shape));
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          1, &element_shape));
      TF_RETURN_IF_ERROR(c->Merge(element_shape_from_tensor_shape,
                                  element_shape,
                                  &element_shape_from_tensor_shape));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{
                 {element_shape, element_dtype}});
      return Status::OK();
    });

REGISTER_OP("TensorListFromTensor")
    .Input("tensor: element_dtype")
    .Input("element_shape: shape_type")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle tensor_shape = c->input(0);
      shape_inference::ShapeHandle tensor_shape_except_first_dim;
      TF_RETURN_IF_ERROR(
          c->Subshape(tensor_shape, 1, &tensor_shape_except_first_dim));
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          1, &element_shape));
      TF_RETURN_IF_ERROR(c->Merge(tensor_shape_except_first_dim, element_shape,
                                  &tensor_shape_except_first_dim));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{
                 {element_shape, element_dtype}});
      return Status::OK();
    });

REGISTER_OP("TensorListElementShape")
    .Input("input_handle: variant")
    .Output("element_shape: shape_type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (!IsValidTensorListHandleData(handle_data)) {
        c->set_output(0, c->Vector(c->UnknownDim()));
        return Status::OK();
      }
      c->set_output(0, c->Vector(c->Rank((*handle_data)[0].shape)));
      return Status::OK();
    });

REGISTER_OP("TensorListReserve")
    .Input("element_shape: shape_type")
    .Input("num_elements: int32")
    .Output("handle: variant")
    .Attr("element_dtype: type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          0, &element_shape));
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{
                 {element_shape, element_dtype}});
      return Status::OK();
    });

REGISTER_OP("TensorListGetItem")
    .Input("input_handle: variant")
    .Input("index: int32")
    .Input("element_shape: int32")
    .Output("item: element_dtype")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      shape_inference::ShapeHandle element_shape = c->UnknownShape();
      if (IsValidTensorListHandleData(handle_data)) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        element_shape = list_shape_type.shape;
        if (list_shape_type.dtype != element_dtype) {
          return errors::InvalidArgument("Expected list with element dtype ",
                                         DataTypeString(element_dtype),
                                         " but got list with element dtype ",
                                         DataTypeString(list_shape_type.dtype));
        }
      }
      shape_inference::ShapeHandle element_shape_input = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          2, &element_shape_input));
      TF_RETURN_IF_ERROR(
          c->Merge(element_shape, element_shape_input, &element_shape));
      c->set_output(0, element_shape);
      return Status::OK();
    });

REGISTER_OP("TensorListResize")
    .Input("input_handle: variant")
    .Input("size: int32")
    .Output("output_handle: variant")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Check that `size` has scalar shape.
      shape_inference::ShapeHandle size_shape = c->input(1);
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(size_shape, 0, &unused));
      c->set_output(0, c->Scalar());
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (IsValidTensorListHandleData(handle_data)) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      return Status::OK();
    });

REGISTER_OP("TensorListSetItem")
    .Input("input_handle: variant")
    .Input("index: int32")
    .Input("item: element_dtype")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      c->set_output(0, c->Scalar());
      if (IsValidTensorListHandleData(handle_data)) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        shape_inference::ShapeHandle item_shape = c->input(2);
        TF_RETURN_IF_ERROR(
            c->Merge(item_shape, list_shape_type.shape, &item_shape));
        c->set_output_handle_shapes_and_types(0, *handle_data);
      } else {
        c->set_output_handle_shapes_and_types(
            0, {{c->UnknownShape(), element_dtype}});
      }
      return Status::OK();
    });

REGISTER_OP("TensorListGather")
    .Input("input_handle: variant")
    .Input("indices: int32")
    .Input("element_shape: int32")
    .Output("values: element_dtype")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      shape_inference::ShapeHandle element_shape = c->UnknownShape();
      if (IsValidTensorListHandleData(handle_data)) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        element_shape = list_shape_type.shape;
        if (list_shape_type.dtype != element_dtype) {
          return errors::InvalidArgument("Expected list with element dtype ",
                                         DataTypeString(element_dtype),
                                         " but got list with element dtype ",
                                         DataTypeString(list_shape_type.dtype));
        }
      }
      shape_inference::ShapeHandle element_shape_input = c->UnknownShape();
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          2, &element_shape_input));
      TF_RETURN_IF_ERROR(
          c->Merge(element_shape, element_shape_input, &element_shape));
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(c->input(1), element_shape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("TensorListScatter")
    .Input("tensor: element_dtype")
    .Input("indices: int32")
    .Input("element_shape: shape_type")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          2, &element_shape));
      c->set_output_handle_shapes_and_types(0,
                                            {{element_shape, element_dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("TensorListScatterV2")
    .Input("tensor: element_dtype")
    .Input("indices: int32")
    .Input("element_shape: shape_type")
    .Input("num_elements: int32")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensorTreatScalarAsUnknownShape(
          2, &element_shape));
      c->set_output_handle_shapes_and_types(0,
                                            {{element_shape, element_dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("TensorListScatterIntoExistingList")
    .Input("input_handle: variant")
    .Input("tensor: element_dtype")
    .Input("indices: int32")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle ignored;
      // Check that tensor is at least a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &ignored));
      // Check that indices is a vector.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &ignored));

      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle element_shape = c->UnknownShape();

      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (IsValidTensorListHandleData(handle_data)) {
        TF_RETURN_IF_ERROR(VerifyHandleData(c, *handle_data, element_dtype));
        element_shape = GetElementShapeFromHandleData(*handle_data);
      }
      c->set_output_handle_shapes_and_types(0,
                                            {{element_shape, element_dtype}});
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("TensorListConcatLists")
    .Input("input_a: variant")
    .Input("input_b: variant")
    .Attr("element_dtype: type")
    .Output("output: variant")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto input_a = c->input(0);
      auto input_b = c->input(1);
      TF_RETURN_IF_ERROR(c->Merge(input_a, input_b, &input_a));
      c->set_output(0, input_a);

      DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));

      auto* handle_data_a = c->input_handle_shapes_and_types(0);
      auto* handle_data_b = c->input_handle_shapes_and_types(1);
      bool handle_data_a_nonempty = handle_data_a && !handle_data_a->empty();
      bool handle_data_b_nonempty = handle_data_b && !handle_data_b->empty();
      if (!(handle_data_a_nonempty || handle_data_b_nonempty)) {
        c->set_output_handle_shapes_and_types(
            0, {{c->UnknownShape(), element_dtype}});
        return Status::OK();
      }
      shape_inference::ShapeAndType list_shape_type_a =
          handle_data_a_nonempty ? handle_data_a->at(0) : handle_data_b->at(0);
      const shape_inference::ShapeAndType& list_shape_type_b =
          handle_data_b_nonempty ? handle_data_b->at(0) : handle_data_a->at(0);
      if (list_shape_type_a.dtype != element_dtype) {
        return errors::InvalidArgument("input_a.type != element_dtype: ",
                                       DataTypeString(list_shape_type_a.dtype),
                                       " vs. ", DataTypeString(element_dtype));
      }
      if (list_shape_type_b.dtype != element_dtype) {
        return errors::InvalidArgument("input_b.type != element_dtype: ",
                                       DataTypeString(list_shape_type_b.dtype),
                                       " vs. ", DataTypeString(element_dtype));
      }
      TF_RETURN_IF_ERROR(c->Merge(list_shape_type_a.shape,
                                  list_shape_type_b.shape,
                                  &list_shape_type_a.shape));
      c->set_output_handle_shapes_and_types(0, {list_shape_type_a});
      return Status::OK();
    });

}  // namespace
}  // namespace tensorflow
