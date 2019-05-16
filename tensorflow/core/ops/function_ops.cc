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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

REGISTER_SYSTEM_OP("_Arg")
    .Output("output: T")
    .Attr("T: type")
    .Attr("index: int >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      const AttrValue* dtype_attr = context->attrs().Find("T");
      if (!dtype_attr) {
        return errors::InvalidArgument(
            "_Arg node does not have attribute \"T\"");
      }

      if (dtype_attr->type() == DT_RESOURCE) {
        const AttrValue* dtype_attr = context->attrs().Find("_handle_dtypes");
        const AttrValue* shape_attr = context->attrs().Find("_handle_shapes");
        if (dtype_attr && shape_attr) {
          if (dtype_attr->list().type().empty()) {
            return errors::InvalidArgument(
                "Invalid \"_handle_dtypes\" attribute value for _Arg node: ",
                shape_attr->DebugString());
          }
          if (shape_attr->list().shape().empty()) {
            return errors::InvalidArgument(
                "Invalid \"_handle_shapes\" attribute value for _Arg node: ",
                shape_attr->DebugString());
          }
          DataType dtype = dtype_attr->list().type(0);
          const TensorShapeProto& shape_proto = shape_attr->list().shape(0);
          shape_inference::ShapeHandle shape_handle;
          TF_RETURN_IF_ERROR(
              context->MakeShapeFromShapeProto(shape_proto, &shape_handle));
          context->set_output(0, shape_handle);
          context->set_output_handle_shapes_and_types(
              0, std::vector<shape_inference::ShapeAndType>{
                     {shape_handle, dtype}});
        } else {
          context->set_output(0, context->UnknownShape());
        }
      } else {
        const AttrValue* shape_attr = context->attrs().Find("_output_shapes");
        if (shape_attr && shape_attr->has_list()) {
          if (shape_attr->list().shape().empty()) {
            return errors::InvalidArgument(
                "Invalid \"_output_shapes\" attribute value for _Arg node: ",
                shape_attr->DebugString());
          }
          const TensorShapeProto& shape_proto = shape_attr->list().shape(0);
          shape_inference::ShapeHandle shape_handle;
          TF_RETURN_IF_ERROR(
              context->MakeShapeFromShapeProto(shape_proto, &shape_handle));
          context->set_output(0, shape_handle);
        } else {
          context->set_output(0, context->UnknownShape());
        }
      }
      return Status::OK();
    })
    .Doc(R"doc(
A graph node which represents an argument to a function.

output: The argument.
index: This argument is the index-th argument of the function.

Attributes for shape inference:
1. _output_shapes: this attribute can be set on an _Arg node producing
   non-resource output(s). If set, its value should contain a list of
   TensorShapeProto describing the shape(s) of the tensor(s) this _Arg node will
   produce. If set, _Arg node's shape inference function will use it as the
   node's output shapes.
2. _handle_dtypes and _handle_shapes: these attributes can be set on an _Arg
   node producing resource output(s). If set, value of _handle_dtypes should
   contain the dtype(s) of the resource(s) and value of _handle_shapes should
   contain the shape(s) of the resource(s). If both attributes are set, _Arg
   node's shape inference function will use their values as the node's output
   type(s) and shape(s).
)doc");

REGISTER_SYSTEM_OP("_DeviceArg")
    .Output("output: T")
    .Attr("T: type")
    .Attr("index: int >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      context->set_output(0, context->UnknownShape());
      return Status::OK();
    })
    .Doc(R"doc(
A graph node which represents an argument to a function.

output: The argument.
index: This argument is the index-th argument of the function.
)doc");

REGISTER_SYSTEM_OP("_Retval")
    .Input("input: T")
    .Attr("T: type")
    .Attr("index: int >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      return Status::OK();
    })
    .Doc(R"doc(
A graph node which represents a return value of a function.

input: The return value.
index: This return value is the index-th return value of the function.
)doc");

REGISTER_SYSTEM_OP("_DeviceRetval")
    .Input("input: T")
    .Attr("T: type")
    .Attr("index: int >= 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* context) {
      return Status::OK();
    })
    .Doc(R"doc(
A graph node which represents a return value of a function.

input: The return value.
index: This return value is the index-th return value of the function.
)doc");

REGISTER_OP("_ListToArray")
    .Input("input: Tin")
    .Output("output: N * T")
    .Attr("Tin: list(type)")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Converts a list of tensors to an array of tensors.
)doc");

REGISTER_OP("_ArrayToList")
    .Input("input: N * T")
    .Output("output: out_types")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Attr("out_types: list(type)")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Converts an array of tensors to a list of tensors.
)doc");

}  // namespace tensorflow
