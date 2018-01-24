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

REGISTER_OP("EmptyTensorList")
    .Input("element_shape: shape_type")
    .Output("handle: variant")
    .Attr("element_dtype: type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &t));
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{{s, t}});
      return Status::OK();
    });

REGISTER_OP("TensorListPushBack")
    .Input("input_handle: variant")
    .Input("tensor: element_dtype")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &t));
      shape_inference::ShapeHandle s = c->UnknownShape();

      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && handle_data->size() != 1) {
        return errors::InvalidArgument(
            "Trying to push to list with wrong variant data.");
      }
      if (handle_data != nullptr) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        if (list_shape_type.dtype != t) {
          return errors::InvalidArgument(
              "Trying to push to list with wrong element dtype. List has type ",
              DataTypeString(list_shape_type.dtype),
              " but trying to push element with type ", DataTypeString(t));
        }
        shape_inference::ShapeHandle ignored;
        TF_RETURN_IF_ERROR(c->Merge(s, list_shape_type.shape, &ignored));
        s = list_shape_type.shape;
      }
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{{s, t}});
      return Status::OK();
    });

REGISTER_OP("TensorListLength")
    .Input("input_handle: variant")
    .Output("length: int32")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TensorListPopBack")
    .Input("input_handle: variant")
    .Output("output_handle: variant")
    .Output("tensor: element_dtype")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &t));
      shape_inference::ShapeHandle s = c->UnknownShape();
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && handle_data->size() != 1) {
        return errors::InvalidArgument(
            "Trying to read from list with invalid variant data.");
      }
      if (handle_data != nullptr) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        if (list_shape_type.dtype != t) {
          return errors::InvalidArgument(
              "Trying to read from list with wrong element dtype. List has "
              "type ",
              DataTypeString(list_shape_type.dtype),
              " but trying to push element with type ", DataTypeString(t));
        }
        shape_inference::ShapeHandle ignored;
        TF_RETURN_IF_ERROR(c->Merge(s, list_shape_type.shape, &ignored));
        c->set_output_handle_shapes_and_types(0, *handle_data);
        s = list_shape_type.shape;
      }
      c->set_output(1, s);
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("TensorListStack")
    .Input("input_handle: variant")
    .Output("tensor: element_dtype")
    .Attr("element_dtype: type")
    .Attr("num_elements: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &t));
      shape_inference::ShapeHandle s = c->UnknownShape();
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && handle_data->size() != 1) {
        return errors::InvalidArgument(
            "Trying to read from list with wrong variant data.");
      }
      if (handle_data != nullptr) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        if (list_shape_type.dtype != t) {
          return errors::InvalidArgument(
              "Trying to read from list with wrong element dtype. List has "
              "type ",
              DataTypeString(list_shape_type.dtype), " but expectec type ",
              DataTypeString(t));
        }
        shape_inference::ShapeHandle ignored;
        TF_RETURN_IF_ERROR(c->Merge(s, list_shape_type.shape, &ignored));
        if (!c->FullyDefined(s) || !c->FullyDefined(list_shape_type.shape)) {
          return errors::InvalidArgument(
              "Can only gather from a list with fully defined shapes.");
        }
        s = list_shape_type.shape;
      }
      int expected_num_elements = -1;
      TF_RETURN_IF_ERROR(c->GetAttr("num_elements", &expected_num_elements));
      shape_inference::ShapeHandle num_elements;
      if (expected_num_elements == -1) {
        num_elements = c->MakeShape({c->UnknownDim()});
      } else {
        num_elements = c->MakeShape({expected_num_elements});
      }
      shape_inference::ShapeHandle result;
      TF_RETURN_IF_ERROR(c->Concatenate(num_elements, s, &result));
      c->set_output(0, result);
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
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &t));
      shape_inference::ShapeHandle s = c->input(0);
      shape_inference::ShapeHandle o;
      TF_RETURN_IF_ERROR(c->Subshape(s, 1, &o));
      shape_inference::ShapeHandle element_shape;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &element_shape));
      TF_RETURN_IF_ERROR(c->Merge(o, element_shape, &o));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{{element_shape, t}});
      return Status::OK();
    });

REGISTER_OP("TensorListElementShape")
    .Input("input_handle: variant")
    .Output("element_shape: shape_type")
    .Attr("shape_type: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data == nullptr) {
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
      shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &t));
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{{s, t}});
      return Status::OK();
    });

REGISTER_OP("TensorListGetItem")
    .Input("input_handle: variant")
    .Input("index: int32")
    .Output("item: element_dtype")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &t));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      shape_inference::ShapeHandle element_shape = c->UnknownShape();
      if (handle_data != nullptr) {
        const shape_inference::ShapeAndType& list_shape_type =
            (*handle_data)[0];
        element_shape = list_shape_type.shape;
        if (list_shape_type.dtype != t) {
          return errors::InvalidArgument("Expected list with element dtype ",
                                         DataTypeString(t),
                                         " but got list with element dtype ",
                                         DataTypeString(list_shape_type.dtype));
        }
      }
      c->set_output(0, element_shape);
      return Status::OK();
    });

REGISTER_OP("TensorListSetItem")
    .Input("input_handle: variant")
    .Input("index: int32")
    .Input("item: element_dtype")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      DataType t;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &t));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data == nullptr) {
        c->set_output_handle_shapes_and_types(0, {{c->UnknownShape(), t}});
        return Status::OK();
      }
      const shape_inference::ShapeAndType& list_shape_type = (*handle_data)[0];
      shape_inference::ShapeHandle s = c->input(2);
      TF_RETURN_IF_ERROR(c->Merge(s, list_shape_type.shape, &s));
      c->set_output_handle_shapes_and_types(0, *handle_data);
      return Status::OK();
    });

}  // namespace
}  // namespace tensorflow
