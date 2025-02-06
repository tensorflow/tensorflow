/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"

#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

// TensorList is represented by a tuple.
// - The first part of the tuple is a buffer containing all the tensors,
// - The following parts are push indices for all nested levels of
//   TensorLists. The last part is push index for the outermost TensorList.
//
// TensorList, as it name suggests, is conceptually a list of tensors. In actual
// representation of a non-nested TensorList, the buffer shape is
// [element_shape, tensor_list_size]. We will call tensor_list_size "leading
// dimension" below. Notice that the leading dimension must be a compile time
// constant, since it's part of the buffer shape.
//
// Example: consider a 3-level nested TensorList whose element type is scalar.
// Assume inner TensorList has leading dimension 4, middle TensorList has 3,
// and outer TensorList has 3.
// Assume that lower cased letter means there is data in that position, and "."
// means there is no data in that position.
// First element of outer TensorList:
// [ a . . . ]
// [ b c . . ]
// [ d e f . ]
// Second element of outer TensorList:
// [ g h i . ]
// [ j k . . ]
// [ . . . . ]
// Third element: not pushed yet.
//
// The first part of the tuple is an array of shape [3, 3, 4] containing data.
// The second part is an array of shape [3, 3], each element is push index
// for the inner TensorList. In this case, its values are:
// [ 1 2 3 ]
// [ 3 2 . ]
// [ . . . ]
// The third part is an array of shape [3], each element is push index for
// the middle TensorList. In this case, its values are:
// [ 3 ]
// [ 2 ]
// [ . ]
// The forth (and last) part is a scalar. It's the push index for the outer
// TensorList. In this case, its values is 2.
//
// Now imagine we need to push the following element to the outer TensorList:
// [ l . . . ]
// [ m n . . ]
// [ . . . . ]
// This element is represented by a tuple of 3 parts:
// First part is all data.
// Second part is push indices for the inner TensorList, which is [ 1 2 . ].
// Third part is push index for the middle TensorList, which is 2.
// Now let's do the push.
// First, we append its data to outer TensorList's data.
// Then we start to deal with push indices. Similar to data, we append push
// indices for each level of TensorList.
// For the inner TensorList: append push indices for the pushed element.
// [ 1 2 3 ]               [ 1 2 3 ]
// [ 3 2 . ] +           = [ 3 2 . ]
// [ . . . ]   [ 1 2 . ]   [ 1 2 . ]
// For the middle TensorList: append push indices for the pushed element.
// [ 3 ]           [ 3 ]
// [ 2 ] +       = [ 2 ]
// [ . ]   [ 2 ]   [ 2 ]
// For the outer TensorList: just add 1.
// 2 + 1 = 3
//
// Popping an element from the outer TensorList also follows a similar process.
// First part is data. We get data by slicing data with push index for outer
// TensorList (which is 3).
// Second part is push indices for inner TensorList. We get it by slicing
// push indices for inner TensorList with push index for outer TensorList (which
// is 3).
// [ 1 2 3 ]
// [ 3 2 . ]
// [ 1 2 . ] ===> This is what we want
// Third part is push index for middle TensorList. We get it by slicing
// push indices for middle TensorList with push index for outer TensorList
// (which is 3).
// [ 3 ]
// [ 2 ]
// [ 2 ] ===> This is what we want

namespace tensorflow {

bool IsTensorListInput(XlaOpKernelContext* ctx, int index) {
  return ctx->InputExpression(index).kind() == XlaExpression::Kind::kTensorList;
}

absl::Status IsTensorListInitialized(xla::XlaOp list, bool* is_initialized) {
  TF_ASSIGN_OR_RETURN(xla::Shape list_shape, list.builder()->GetShape(list));
  *is_initialized = list_shape.IsTuple();
  return absl::OkStatus();
}

absl::Status IsNestedTensorList(xla::XlaOp list, bool* is_nested_list) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (!is_initialized) {
    return errors::InvalidArgument("TensorList is not initialized");
  }
  TF_ASSIGN_OR_RETURN(xla::Shape list_shape, list.builder()->GetShape(list));
  *is_nested_list = (xla::ShapeUtil::TupleElementCount(list_shape) > 2);
  return absl::OkStatus();
}

absl::Status BuildNonNestedTensorList(xla::XlaOp buffer, xla::XlaOp push_index,
                                      xla::XlaOp* output_list) {
  TF_RET_CHECK(buffer.builder());
  *output_list = xla::Tuple(buffer.builder(), {buffer, push_index});
  return absl::OkStatus();
}

absl::Status GetTensorListBufferShape(xla::XlaOp list,
                                      xla::Shape* buffer_shape) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (!is_initialized) {
    return errors::InvalidArgument("TensorList is not initialized");
  }
  TF_ASSIGN_OR_RETURN(xla::Shape list_shape, list.builder()->GetShape(list));
  *buffer_shape = xla::ShapeUtil::GetTupleElementShape(list_shape, 0);
  return absl::OkStatus();
}

absl::Status GetTensorListBuffer(xla::XlaOp list, xla::XlaOp* buffer) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (!is_initialized) {
    return errors::InvalidArgument("TensorList is not initialized");
  }
  *buffer = xla::GetTupleElement(list, 0);
  return absl::OkStatus();
}

absl::Status GetTensorListPushIndex(xla::XlaOp list, xla::XlaOp* push_index) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (!is_initialized) {
    return errors::InvalidArgument("TensorList is not initialized");
  }
  TF_ASSIGN_OR_RETURN(xla::Shape list_shape, list.builder()->GetShape(list));
  int tuple_size = xla::ShapeUtil::TupleElementCount(list_shape);
  *push_index = xla::GetTupleElement(list, tuple_size - 1);
  return absl::OkStatus();
}

absl::Status SetTensorListPushIndex(xla::XlaOp list, xla::XlaOp push_index,
                                    xla::XlaOp* result) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (!is_initialized) {
    return errors::InvalidArgument("TensorList is not initialized");
  }
  TF_ASSIGN_OR_RETURN(xla::Shape list_shape, list.builder()->GetShape(list));
  int tuple_size = xla::ShapeUtil::TupleElementCount(list_shape);
  std::vector<xla::XlaOp> result_parts;
  result_parts.reserve(tuple_size);
  for (int i = 0; i < tuple_size - 1; i++) {
    result_parts.push_back(xla::GetTupleElement(list, i));
  }
  result_parts.push_back(push_index);
  *result = xla::Tuple(list.builder(), result_parts);
  return absl::OkStatus();
}

xla::XlaOp BuildUninitializedTensorList(xla::XlaBuilder* b,
                                        int64_t leading_dimension,
                                        bool leading_size_is_dynamic,
                                        xla::XlaOp leading_dim_size) {
  auto zero =
      xla::ConstantLiteral(b, xla::LiteralUtil::Zero(xla::PrimitiveType::S32));
  auto broadcast =
      xla::Broadcast(zero, std::vector<int64_t>{leading_dimension});
  if (leading_size_is_dynamic) {
    return xla::SetDimensionSize(broadcast, leading_dim_size, 0);
  } else {
    return broadcast;
  }
}

absl::Status GetLeadingDimForTensorList(xla::XlaOp list, int64_t* leading_dim,
                                        bool* leading_dim_is_dynamic,
                                        xla::XlaOp* leading_dim_dynamic_size) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  TF_ASSIGN_OR_RETURN(xla::Shape list_shape, list.builder()->GetShape(list));
  if (is_initialized) {
    auto buffer_shape = xla::ShapeUtil::GetTupleElementShape(list_shape, 0);
    *leading_dim_is_dynamic = buffer_shape.is_dynamic_dimension(0);
    auto buffer = xla::GetTupleElement(list, 0);
    *leading_dim = buffer_shape.dimensions(0);
    *leading_dim_dynamic_size = xla::GetDimensionSize(buffer, 0);
  } else {
    *leading_dim_is_dynamic = list_shape.is_dynamic_dimension(0);
    *leading_dim = list_shape.dimensions(0);
    *leading_dim_dynamic_size = xla::GetDimensionSize(list, 0);
  }
  return absl::OkStatus();
}

absl::Status GetTensorListShapeFromElementTensorListShape(
    const xla::Shape& element_tensor_list_shape, int64_t leading_dim,
    bool leading_dim_is_dynamic, xla::Shape* tensor_list_shape) {
  std::vector<xla::Shape> shapes;
  int tuple_size = xla::ShapeUtil::TupleElementCount(element_tensor_list_shape);
  for (int i = 0; i < tuple_size; i++) {
    const xla::Shape& shape =
        xla::ShapeUtil::GetTupleElementShape(element_tensor_list_shape, i);
    std::vector<int64_t> dimensions = xla::SpanToVector(shape.dimensions());
    dimensions.insert(dimensions.begin(), leading_dim);
    shapes.push_back(
        xla::ShapeUtil::MakeShape(shape.element_type(), dimensions));
    if (leading_dim_is_dynamic) {
      shapes.back().set_dynamic_dimension(0, true);
    }
  }
  shapes.push_back(xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32,
                                             std::vector<int64_t>{}));
  *tensor_list_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  return absl::OkStatus();
}

absl::Status GetTensorListShapeFromElementShape(const xla::Shape& element_shape,
                                                int64_t leading_dim,
                                                bool leading_dim_is_dynamic,
                                                xla::Shape* tensor_list_shape) {
  if (!element_shape.IsArray()) {
    return errors::InvalidArgument(
        "GetTensorListShapeFromElementShape() only supports normal tensor "
        "shape. But element shape is ",
        element_shape.DebugString());
  }
  std::vector<xla::Shape> shapes;
  std::vector<int64_t> dimensions =
      xla::SpanToVector(element_shape.dimensions());
  dimensions.insert(dimensions.begin(), leading_dim);
  shapes.push_back(
      xla::ShapeUtil::MakeShape(element_shape.element_type(), dimensions));
  shapes.back().set_dynamic_dimension(0, leading_dim_is_dynamic);
  shapes.push_back(xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32,
                                             std::vector<int64_t>{}));
  *tensor_list_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  return absl::OkStatus();
}

absl::Status CreateZerosTensorListWithShape(
    xla::XlaBuilder* b, const xla::Shape& list_shape,
    const std::vector<std::vector<xla::XlaOp>>& dynamic_dims,
    xla::XlaOp* list) {
  int tuple_size = xla::ShapeUtil::TupleElementCount(list_shape);
  std::vector<xla::XlaOp> elements;
  TF_RET_CHECK(dynamic_dims.size() == tuple_size - 1);
  for (int i = 0; i < tuple_size - 1; i++) {
    const xla::Shape& shape =
        xla::ShapeUtil::GetTupleElementShape(list_shape, i);
    xla::XlaOp zero =
        xla::ConstantLiteral(b, xla::LiteralUtil::Zero(shape.element_type()));
    xla::XlaOp zeros = xla::Broadcast(zero, shape.dimensions());
    TF_RET_CHECK(dynamic_dims[i].size() == shape.dimensions_size());
    for (int64_t dim = 0; dim < shape.dimensions_size(); ++dim) {
      if (shape.is_dynamic_dimension(dim)) {
        zeros = xla::SetDimensionSize(zeros, dynamic_dims[i][dim], dim);
      }
    }
    elements.push_back(zeros);
  }
  // List size (last item) has to be S32.
  TF_RET_CHECK(xla::ShapeUtil::GetTupleElementShape(list_shape, tuple_size - 1)
                   .element_type() == xla::S32);
  elements.push_back(xla::ConstantLiteral(b, xla::LiteralUtil::Zero(xla::S32)));
  *list = xla::Tuple(b, elements);
  return absl::OkStatus();
}

absl::Status GetInitializedTensorListForElement(xla::XlaOp list,
                                                xla::XlaOp element,
                                                bool element_is_tensor_list,
                                                xla::XlaOp* initialized_list) {
  int64_t leading_dim;
  xla::XlaOp leading_dim_dynamic_size;
  bool leading_dim_is_dynamic;
  TF_RETURN_IF_ERROR(GetLeadingDimForTensorList(
      list, &leading_dim, &leading_dim_is_dynamic, &leading_dim_dynamic_size));

  xla::XlaBuilder* b = list.builder();
  xla::Shape list_shape;
  TF_ASSIGN_OR_RETURN(xla::Shape element_shape, b->GetShape(element));

  if (element_is_tensor_list) {
    TF_RETURN_IF_ERROR(GetTensorListShapeFromElementTensorListShape(
        element_shape, leading_dim, leading_dim_is_dynamic, &list_shape));
  } else {
    TF_RETURN_IF_ERROR(GetTensorListShapeFromElementShape(
        element_shape, leading_dim, leading_dim_is_dynamic, &list_shape));
  }
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (is_initialized) {
    // Check shape of initialized list is correct.
    TF_ASSIGN_OR_RETURN(xla::Shape original_list_shape, b->GetShape(list));
    if (!xla::ShapeUtil::Compatible(original_list_shape, list_shape)) {
      return errors::Internal(
          "Invalid TensorList shape: ", original_list_shape.DebugString(),
          ", expected: ", list_shape.DebugString());
    }
    *initialized_list = list;
    return absl::OkStatus();
  } else {
    // Prepare dynamic dimension dimensions for zero tensor list. The dynamic
    // sizes are created by reading the dynamic dimension size of sub-elements.
    std::vector<std::vector<xla::XlaOp>> list_dynamic_dims;
    for (int i = 0; i < list_shape.tuple_shapes_size() - 1; ++i) {
      std::vector<xla::XlaOp> dynamic_dims;
      const xla::Shape& shape = list_shape.tuple_shapes(i);
      dynamic_dims.push_back(leading_dim_dynamic_size);
      xla::XlaOp sub_element;
      if (element_is_tensor_list) {
        sub_element = xla::GetTupleElement(element, i);
      } else {
        sub_element = element;
      }
      for (int64_t dim = 0; dim < shape.dimensions_size() - 1; ++dim) {
        dynamic_dims.push_back(xla::GetDimensionSize(sub_element, dim));
      }
      list_dynamic_dims.push_back(dynamic_dims);
    }
    return CreateZerosTensorListWithShape(b, list_shape, list_dynamic_dims,
                                          initialized_list);
  }
}

absl::Status ExecuteTensorListPushBack(xla::XlaOp list, xla::XlaOp element,
                                       bool element_is_tensor_list,
                                       xla::XlaOp* result) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (!is_initialized) {
    return errors::InvalidArgument("TensorList is not initialized");
  }

  xla::XlaBuilder* b = list.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape list_shape, b->GetShape(list));
  int list_tuple_size = xla::ShapeUtil::TupleElementCount(list_shape);
  xla::XlaOp push_index = xla::GetTupleElement(list, list_tuple_size - 1);

  std::vector<xla::XlaOp> result_parts;

  if (element_is_tensor_list) {
    TF_ASSIGN_OR_RETURN(xla::Shape element_shape, b->GetShape(element));
    int element_tuple_size = xla::ShapeUtil::TupleElementCount(element_shape);
    for (int i = 0; i < element_tuple_size; i++) {
      const xla::Shape& element_part_shape =
          xla::ShapeUtil::GetTupleElementShape(element_shape, i);
      xla::XlaOp element_part = xla::GetTupleElement(element, i);
      std::vector<int64_t> element_part_dims =
          xla::SpanToVector(element_part_shape.dimensions());
      element_part_dims.insert(element_part_dims.begin(), 1);
      element_part = xla::Reshape(element_part, element_part_dims);

      std::vector<xla::XlaOp> start_indices(
          element_part_shape.dimensions_size() + 1,
          xla::ConstantR0<int32>(b, 0));
      start_indices[0] = push_index;

      xla::XlaOp list_part = xla::GetTupleElement(list, i);
      xla::XlaOp updated_list_part =
          xla::DynamicUpdateSlice(list_part, element_part, start_indices);
      result_parts.push_back(updated_list_part);
    }
  } else {
    TF_ASSIGN_OR_RETURN(xla::Shape element_shape, b->GetShape(element));
    std::vector<int64_t> element_dims =
        xla::SpanToVector(element_shape.dimensions());
    element_dims.insert(element_dims.begin(), 1);
    xla::XlaOp update = xla::Reshape(element, element_dims);

    std::vector<xla::XlaOp> start_indices(element_shape.dimensions_size() + 1,
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = push_index;

    xla::XlaOp list_part = xla::GetTupleElement(list, 0);
    xla::XlaOp updated_list_part =
        xla::DynamicUpdateSlice(list_part, update, start_indices);
    result_parts.push_back(updated_list_part);
  }

  xla::XlaOp updated_push_index = push_index + xla::ConstantR0<int32>(b, 1);
  result_parts.push_back(updated_push_index);

  *result = xla::Tuple(b, result_parts);
  return absl::OkStatus();
}

absl::Status ExecuteTensorListPopBack(xla::XlaOp list, xla::XlaOp* list_result,
                                      xla::XlaOp* element_result,
                                      bool* element_is_tensor_list) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (!is_initialized) {
    return errors::InvalidArgument("TensorList is not initialized");
  }

  // If the TensorList is a nested TensorList, element will be TensorList.
  TF_RETURN_IF_ERROR(IsNestedTensorList(list, element_is_tensor_list));

  xla::XlaBuilder* b = list.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape list_shape, b->GetShape(list));
  int list_tuple_size = xla::ShapeUtil::TupleElementCount(list_shape);
  xla::XlaOp push_index = xla::GetTupleElement(list, list_tuple_size - 1);
  push_index = push_index - xla::ConstantR0<int32>(b, 1);

  std::vector<xla::XlaOp> list_result_parts, element_result_parts;
  for (int i = 0; i < list_tuple_size - 1; i++) {
    const xla::Shape& list_part_shape =
        xla::ShapeUtil::GetTupleElementShape(list_shape, i);
    std::vector<xla::XlaOp> start_indices(list_part_shape.dimensions_size(),
                                          xla::ConstantR0<int32>(b, 0));
    start_indices[0] = push_index;

    std::vector<int64_t> slice_shape =
        xla::SpanToVector(list_part_shape.dimensions());
    slice_shape[0] = 1LL;

    xla::XlaOp list_part = xla::GetTupleElement(list, i);
    xla::XlaOp read = xla::DynamicSlice(list_part, start_indices, slice_shape);

    slice_shape.erase(slice_shape.begin());
    element_result_parts.push_back(xla::Reshape(read, slice_shape));
    list_result_parts.push_back(list_part);
  }
  list_result_parts.push_back(push_index);

  *list_result = xla::Tuple(b, list_result_parts);
  if (*element_is_tensor_list) {
    *element_result = xla::Tuple(b, element_result_parts);
  } else {
    *element_result = element_result_parts[0];
  }

  return absl::OkStatus();
}

absl::Status ExecuteTensorListSetItem(xla::XlaOp list, xla::XlaOp index,
                                      xla::XlaOp element, xla::XlaOp* result) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (!is_initialized) {
    return errors::InvalidArgument("TensorList is not initialized");
  }
  bool is_nested;
  TF_RETURN_IF_ERROR(IsNestedTensorList(list, &is_nested));
  if (is_nested) {
    return errors::Unimplemented(
        "ExecuteTensorListSetItem() only supports non-nested TensorList");
  }

  xla::XlaBuilder* b = list.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape element_shape, b->GetShape(element));
  std::vector<int64_t> element_dims =
      xla::SpanToVector(element_shape.dimensions());
  element_dims.insert(element_dims.begin(), 1);
  xla::XlaOp update = xla::Reshape(element, element_dims);

  std::vector<xla::XlaOp> start_indices(element_shape.dimensions_size() + 1,
                                        xla::ConstantR0<int32>(b, 0));
  start_indices[0] = index;

  xla::XlaOp list_part = xla::GetTupleElement(list, 0);
  {
    TF_ASSIGN_OR_RETURN(const xla::Shape* list_part_shape,
                        b->GetShapePtr(list_part));
    TF_ASSIGN_OR_RETURN(const xla::Shape* update_shape, b->GetShapePtr(update));
    for (int i = 0; i < list_part_shape->dimensions_size(); ++i) {
      auto list_part_dim_size = list_part_shape->dimensions(i);
      auto update_dim_size = update_shape->dimensions(i);
      // If the update is larger than the list part, the DynamicUpdateSlice will
      // fail so just ignore this operation and return list as is.
      if (update_dim_size > list_part_dim_size) {
        LOG_FIRST_N(WARNING, 1)
            << "Warning: TensorListSetItem: ignoring set item because the "
               "update dim ["
            << update_dim_size << "] is larger than the list dim ["
            << list_part_dim_size << "] at dimension " << i << ".";

        *result = list;
        return absl::OkStatus();
      }
    }
  }
  xla::XlaOp updated_list_part =
      xla::DynamicUpdateSlice(list_part, update, start_indices);

  std::vector<xla::XlaOp> result_parts;
  result_parts.push_back(updated_list_part);
  result_parts.push_back(xla::GetTupleElement(list, 1));
  *result = xla::Tuple(b, result_parts);
  return absl::OkStatus();
}

absl::Status ExecuteTensorListGetItem(xla::XlaOp list, xla::XlaOp index,
                                      xla::XlaOp* result) {
  bool is_initialized;
  TF_RETURN_IF_ERROR(IsTensorListInitialized(list, &is_initialized));
  if (!is_initialized) {
    return errors::InvalidArgument("TensorList is not initialized");
  }
  bool is_nested;
  TF_RETURN_IF_ERROR(IsNestedTensorList(list, &is_nested));
  if (is_nested) {
    return errors::Unimplemented(
        "ExecuteTensorListGetItem() only supports non-nested TensorList");
  }

  xla::XlaBuilder* b = list.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape list_shape, b->GetShape(list));
  const xla::Shape& buffer_shape =
      xla::ShapeUtil::GetTupleElementShape(list_shape, 0);
  std::vector<xla::XlaOp> start_indices(buffer_shape.dimensions_size(),
                                        xla::ConstantR0<int32>(b, 0));
  start_indices[0] = index;

  std::vector<int64_t> slice_shape =
      xla::SpanToVector(buffer_shape.dimensions());
  slice_shape[0] = 1LL;

  xla::XlaOp list_part = xla::GetTupleElement(list, 0);
  xla::XlaOp read = xla::DynamicSlice(list_part, start_indices, slice_shape);
  // Propagate dynamic dimensions from buffer to the sliced buffer, except for
  // leading dimension (which is always static 1).
  for (int64_t i = 1; i < buffer_shape.dimensions_size(); ++i) {
    if (buffer_shape.is_dynamic_dimension(i)) {
      auto buffer = xla::GetTupleElement(list, 0);
      auto gds = xla::GetDimensionSize(buffer, i);
      read = xla::SetDimensionSize(read, gds, i);
    }
  }
  slice_shape.erase(slice_shape.begin());
  *result = xla::Reshape(read, slice_shape);
  return absl::OkStatus();
}

absl::Status ExecuteTensorListFromTensor(int push_index, xla::XlaOp tensor,
                                         xla::XlaOp* result) {
  xla::XlaBuilder* b = tensor.builder();
  TF_ASSIGN_OR_RETURN(xla::Shape shape, b->GetShape(tensor));
  if (!shape.IsArray()) {
    return errors::InvalidArgument(
        "ExecuteTensorListFromTensor() only supports normal tensor. But input "
        "shape is ",
        shape.DebugString());
  }

  std::vector<xla::XlaOp> result_parts{tensor,
                                       xla::ConstantR0<int32>(b, push_index)};
  *result = xla::Tuple(b, result_parts);
  return absl::OkStatus();
}

}  // namespace tensorflow
