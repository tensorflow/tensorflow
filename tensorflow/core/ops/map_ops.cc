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

bool IsValidTensorMapHandleData(
    const std::vector<shape_inference::ShapeAndType>* handle_data) {
        std::cout << "is valid tensor map handle data " << handle_data->size() << std::endl;
        return true;
  //return handle_data != nullptr && handle_data->size() == 1;
}

REGISTER_OP("EmptyTensorMap")
    .Output("handle: variant")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      return Status::OK();
    });

REGISTER_OP("TensorMapSize")
    .Input("input_handle: variant")
    .Output("size: int32")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TensorMapInsert")
    .Input("input_handle: variant")
    .Input("key: element_dtype")
    .Input("value: element_dtype")
    .Output("output_handle: variant")
    .Attr("element_dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      /*DataType element_dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("element_dtype", &element_dtype));
      shape_inference::ShapeHandle element_shape = c->UnknownShape();*/

      /*auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr && handle_data->size() > 1) {
        return errors::InvalidArgument(
            "Trying to push to list with wrong variant data.");
      }
      if (IsValidTensorMapHandleData(handle_data)) {
        const shape_inference::ShapeAndType& map_shape_type = (*handle_data)[0];
        if (list_shape_type.dtype != element_dtype) {
          return errors::InvalidArgument(
              "Trying to push to list with wrong element dtype. List has type ",
              DataTypeString(list_shape_type.dtype),
              " but trying to push element with type ",
              DataTypeString(element_dtype));
        }
        shape_inference::ShapeHandle ignored;
        TF_RETURN_IF_ERROR(
            c->Merge(element_shape, map_shape_type.shape, &ignored));
        element_shape = map_shape_type.shape;
      }
      c->set_output_handle_shapes_and_types(
          0, std::vector<shape_inference::ShapeAndType>{
                 {element_shape, element_dtype}});*/
      return Status::OK();
    });

/*REGISTER_OP("TensorMapErase")
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
    });*/


REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      //c->set_output(0, c->Scalar());
      c->set_output(0, c->input(0));
      return Status::OK();
    });

}  // namespace
}  // namespace tensorflow
