/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/computation_builder.h"

#include <stddef.h>
#include <array>
#include <numeric>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

ComputationDataHandle ComputationBuilder::ParseOpResponse(
    const Status& status, OpResponse* response) {
  VLOG(2) << "done with op request";

  if (!status.ok()) {
    NoteError(status);
    return ComputationDataHandle();
  }

  if (response->output().handle() == 0) {
    NoteError(InternalError("No output handle"));
    return ComputationDataHandle();
  }
  return response->output();
}

ComputationBuilder::ComputationBuilder(Client* client,
                                       const string& computation_name)
    : name_(computation_name), first_error_(Status::OK()), client_(client) {}

ComputationBuilder::~ComputationBuilder() {}

void ComputationBuilder::NoteError(const Status& error) {
  if (die_immediately_on_error_) {
    LOG(FATAL) << "error building computation: " << error;
  }

  if (first_error_.ok()) {
    first_error_ = error;
    first_error_backtrace_.CreateCurrent(/*skip_count=*/1);
  }
}

std::unique_ptr<ComputationBuilder> ComputationBuilder::CreateSubBuilder(
    const string& computation_name) {
  auto sub_builder = MakeUnique<ComputationBuilder>(client_, computation_name);
  sub_builder->parent_builder_ = this;
  sub_builder->die_immediately_on_error_ = die_immediately_on_error_;
  return sub_builder;
}

Status ComputationBuilder::PrepareComputation() {
  if (!first_error_.ok()) {
    return first_error_;
  }
  if (!computation_.IsNull()) {
    return Status::OK();
  }

  ComputationRequest request;
  request.set_name(name_);
  ComputationResponse response;

  VLOG(2) << "making computation request";
  Status s = client_->stub()->Computation(&request, &response);
  VLOG(2) << "done with computation request";

  if (!s.ok()) {
    NoteError(s);
    return first_error_;
  }

  computation_ = Computation(client_->stub(), response.computation());
  return Status::OK();
}

bool ComputationBuilder::MakeWindow(
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    tensorflow::gtl::ArraySlice<int64> lhs_dilation,
    tensorflow::gtl::ArraySlice<int64> rhs_dilation, Window* window) {
  const auto verify_size = [&](const size_t x, const char* x_name) {
    if (x == 0 || x == window_dimensions.size()) {
      return true;
    } else {
      NoteError(InvalidArgument(
          "%s", tensorflow::strings::StrCat(
                    "Window has different number of window dimensions than of ",
                    x_name, "\nNumber of window dimensions: ",
                    window_dimensions.size(), "\nNumber of ", x_name, ": ", x,
                    "\n")
                    .c_str()));  //
      return false;
    }
  };
  if (!verify_size(window_strides.size(), "window strides") ||
      !verify_size(padding.size(), "padding entries") ||
      !verify_size(lhs_dilation.size(), "lhs dilation factors") ||
      !verify_size(rhs_dilation.size(), "rhs dilation factors")) {
    return false;
  }

  window->Clear();
  for (size_t i = 0; i < window_dimensions.size(); i++) {
    auto dim = window->add_dimensions();
    dim->set_size(window_dimensions[i]);
    if (!window_strides.empty()) {
      dim->set_stride(window_strides[i]);
    } else {
      dim->set_stride(1);
    }
    if (!padding.empty()) {
      dim->set_padding_low(padding[i].first);
      dim->set_padding_high(padding[i].second);
    } else {
      dim->set_padding_low(0);
      dim->set_padding_high(0);
    }
    if (!lhs_dilation.empty()) {
      dim->set_base_dilation(lhs_dilation[i]);
    } else {
      dim->set_base_dilation(1);
    }
    if (!rhs_dilation.empty()) {
      dim->set_window_dilation(rhs_dilation[i]);
    } else {
      dim->set_window_dilation(1);
    }
    dim->set_window_reversal(false);
  }
  return true;
}

ComputationDataHandle ComputationBuilder::ConstantOp(
    const PopulateLiteral& populate) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  ConstantRequest request;
  Literal literal;
  populate(&literal);
  *request.mutable_literal() = literal.ToProto();
  VLOG(3) << "created constant: " << request.literal().ShortDebugString();
  OpRequest op_request;
  *op_request.mutable_constant_request() = request;
  *op_request.mutable_computation() = computation_.handle();
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making constant request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::ConstantLiteral(
    const Literal& literal) {
  return ConstantOp(
      [literal](Literal* mutable_literal) { *mutable_literal = literal; });
}

ComputationDataHandle ComputationBuilder::Parameter(int64 parameter_number,
                                                    const Shape& shape,
                                                    const string& name) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  ParameterRequest request;
  *request.mutable_shape() = shape;
  request.set_parameter(parameter_number);
  request.set_name(name);
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_parameter_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making parameter request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

StatusOr<std::unique_ptr<Shape>> ComputationBuilder::GetShape(
    const ComputationDataHandle& operand) {
  if (!first_error_.ok()) {
    return first_error_;
  }

  GetLocalShapeRequest request;
  *request.mutable_computation() = computation_.handle();
  *request.mutable_operand() = operand;
  GetLocalShapeResponse response;

  VLOG(2) << "making get-shape request";
  Status s = client_->stub()->GetLocalShape(&request, &response);
  VLOG(2) << "done with request";

  if (!s.ok()) {
    NoteError(s);
    return first_error_;
  }
  TF_RET_CHECK(response.has_shape());
  std::unique_ptr<Shape> shape = WrapUnique(response.release_shape());
  TF_RET_CHECK(shape != nullptr);
  return std::move(shape);
}

ComputationDataHandle ComputationBuilder::CheckShape(
    const ComputationDataHandle& operand, const Shape& expected_shape) {
  std::unique_ptr<Shape> actual_shape = GetShape(operand).ConsumeValueOrDie();
  CHECK(ShapeUtil::Equal(expected_shape, *actual_shape))
      << "want " << ShapeUtil::HumanString(expected_shape) << " got "
      << ShapeUtil::HumanString(*actual_shape);
  return operand;
}

void ComputationBuilder::CheckSameShape(const ComputationDataHandle& lhs,
                                        const ComputationDataHandle& rhs) {
  std::unique_ptr<Shape> lhs_shape = GetShape(lhs).ConsumeValueOrDie();
  std::unique_ptr<Shape> rhs_shape = GetShape(rhs).ConsumeValueOrDie();
  VLOG(2) << "checking " << ShapeUtil::HumanString(*lhs_shape) << " equals "
          << ShapeUtil::HumanString(*rhs_shape);
  CHECK(ShapeUtil::Equal(*lhs_shape, *rhs_shape))
      << "lhs " << ShapeUtil::HumanString(*lhs_shape) << " rhs "
      << ShapeUtil::HumanString(*rhs_shape);
}

ComputationDataHandle ComputationBuilder::Slice(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices,
    tensorflow::gtl::ArraySlice<int64> strides) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  SliceRequest request;
  *request.mutable_operand() = operand;
  for (int64 index : start_indices) {
    request.add_start_indices(index);
  }
  for (int64 index : limit_indices) {
    request.add_limit_indices(index);
  }
  for (int64 index : strides) {
    request.add_strides(index);
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_slice_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making slice request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::SliceInDim(
    const ComputationDataHandle& operand, int64 start_index, int64 limit_index,
    int64 stride, int64 dimno) {
  StatusOr<std::unique_ptr<Shape>> shape_status = GetShape(operand);
  if (!shape_status.ok()) {
    NoteError(shape_status.status());
    return ComputationDataHandle{};
  }
  const Shape& shape = *shape_status.ValueOrDie();
  std::vector<int64> starts(ShapeUtil::Rank(shape), 0);
  std::vector<int64> limits(shape.dimensions().begin(),
                            shape.dimensions().end());
  std::vector<int64> strides(ShapeUtil::Rank(shape), 1);
  starts[dimno] = start_index;
  limits[dimno] = limit_index;
  strides[dimno] = stride;
  return Slice(operand, starts, limits, strides);
}

ComputationDataHandle ComputationBuilder::DynamicSlice(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& start_indices,
    tensorflow::gtl::ArraySlice<int64> slice_sizes) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  DynamicSliceRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_start_indices() = start_indices;
  for (int64 index : slice_sizes) {
    request.add_slice_sizes(index);
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_dynamic_slice_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making dynamic slice request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::DynamicUpdateSlice(
    const ComputationDataHandle& operand, const ComputationDataHandle& update,
    const ComputationDataHandle& start_indices) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  DynamicUpdateSliceRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_update() = update;
  *request.mutable_start_indices() = start_indices;
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_dynamic_update_slice_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making dynamic update slice request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::ConcatInDim(
    tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
    int64 dimension) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  ConcatenateRequest request;
  for (const ComputationDataHandle& operand : operands) {
    *request.add_operands() = operand;
  }
  request.set_dimension(dimension);
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_concatenate_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making concatenate request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Broadcast(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> broadcast_sizes) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  BroadcastRequest request;
  *request.mutable_operand() = operand;
  for (int64 size : broadcast_sizes) {
    request.add_broadcast_sizes(size);
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_broadcast_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making broadcast request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Pad(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& padding_value,
    const PaddingConfig& padding_config) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  PadRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_padding_value() = padding_value;
  *request.mutable_padding_config() = padding_config;
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_pad_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making pad request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Reshape(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> dimensions,
    tensorflow::gtl::ArraySlice<int64> new_sizes) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  ReshapeRequest request;
  *request.mutable_operand() = operand;
  for (int64 dimension : dimensions) {
    request.add_dimensions(dimension);
  }
  for (int64 new_size : new_sizes) {
    request.add_new_sizes(new_size);
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_reshape_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making reshape request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Reshape(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> new_sizes) {
  if (!first_error_.ok()) {
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> shape = GetShape(operand);
  if (!shape.ok()) {
    first_error_ = shape.status();
    return ComputationDataHandle();
  }
  std::vector<int64> dimensions(shape.ValueOrDie()->dimensions().size());
  std::iota(dimensions.begin(), dimensions.end(), 0);
  return Reshape(operand, dimensions, new_sizes);
}

ComputationDataHandle ComputationBuilder::Collapse(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> dims_to_collapse) {
  if (!first_error_.ok()) {
    return ComputationDataHandle();
  }

  // Don't support out-of-order collapse here.
  // Checks that the collapsed dimensions are in order and consecutive.
  for (tensorflow::gtl::ArraySlice<int64>::size_type i = 1;
       i < dims_to_collapse.size(); ++i) {
    if (dims_to_collapse[i] - 1 != dims_to_collapse[i - 1]) {
      NoteError(InvalidArgument(
          "Collapsed dimensions are not in order and consecutive."));
      return ComputationDataHandle();
    }
  }

  // Create a new sizes vector from the old shape, replacing the collapsed
  // dimensions by the product of their sizes.
  StatusOr<std::unique_ptr<Shape>> shape_or_status = GetShape(operand);
  if (!shape_or_status.ok()) {
    first_error_ = shape_or_status.status();
    return ComputationDataHandle();
  }
  std::unique_ptr<Shape> original_shape = shape_or_status.ConsumeValueOrDie();

  VLOG(3) << "original shape: " << ShapeUtil::HumanString(*original_shape);
  VLOG(3) << "dims to collapse: "
          << tensorflow::str_util::Join(dims_to_collapse, ",");

  if (dims_to_collapse.size() <= 1) {
    // Not collapsing anything, trivially we can return the operand versus
    // enqueueing a trivial reshape.
    return operand;
  }

  std::vector<int64> new_sizes;
  for (int i = 0; i < ShapeUtil::Rank(*original_shape); ++i) {
    if (i <= dims_to_collapse.front() || i > dims_to_collapse.back()) {
      new_sizes.push_back(original_shape->dimensions(i));
    } else {
      new_sizes.back() *= original_shape->dimensions(i);
    }
  }

  VLOG(3) << "new sizes: [" << tensorflow::str_util::Join(new_sizes, ",")
          << "]";

  return Reshape(operand, new_sizes);
}

void ComputationBuilder::Trace(const string& tag,
                               const ComputationDataHandle& operand) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return;
  }

  TraceRequest request;
  request.set_tag(tag);
  *request.mutable_operand() = operand;
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_trace_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making trace request";
  Status s = client_->stub()->Op(&op_request, &response);
  VLOG(2) << "done with request";

  if (!s.ok()) {
    NoteError(s);
  }
}

ComputationDataHandle ComputationBuilder::Select(
    const ComputationDataHandle& pred, const ComputationDataHandle& on_true,
    const ComputationDataHandle& on_false) {
  return TernaryOp(TRIOP_SELECT, pred, on_true, on_false);
}

ComputationDataHandle ComputationBuilder::Tuple(
    tensorflow::gtl::ArraySlice<ComputationDataHandle> elements) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  VariadicOpRequest request;
  request.set_varop(VAROP_TUPLE);
  for (const ComputationDataHandle& operand : elements) {
    *request.add_operands() = operand;
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_variadic_op_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making variadic op request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::GetTupleElement(
    const ComputationDataHandle& tuple_data, int64 index) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  GetTupleElementRequest request;
  *request.mutable_operand() = tuple_data;
  request.set_index(index);
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_get_tuple_element_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making get tuple element op request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Eq(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_EQ, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Ne(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_NE, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Ge(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_GE, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Gt(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_GT, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Le(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_LE, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Lt(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_LT, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Dot(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs) {
  StatusOr<std::unique_ptr<Shape>> lhs_shape_or_status = GetShape(lhs);
  if (!lhs_shape_or_status.ok()) {
    NoteError(lhs_shape_or_status.status());
    return ComputationDataHandle();
  }
  std::unique_ptr<Shape> lhs_shape = lhs_shape_or_status.ConsumeValueOrDie();

  DotDimensionNumbers dimension_numbers;
  dimension_numbers.add_lhs_contracting_dimensions(
      lhs_shape->dimensions_size() == 1 ? 0 : 1);
  dimension_numbers.add_rhs_contracting_dimensions(0);
  return DotGeneral(lhs, rhs, dimension_numbers);
}

ComputationDataHandle ComputationBuilder::DotGeneral(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    const DotDimensionNumbers& dimension_numbers) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  DotRequest request;
  *request.mutable_lhs() = lhs;
  *request.mutable_rhs() = rhs;
  *request.mutable_dimension_numbers() = dimension_numbers;

  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_dot_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making Dot request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Conv(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  return ConvWithGeneralDimensions(
      lhs, rhs, window_strides, padding,
      CreateDefaultConvDimensionNumbers(window_strides.size()));
}

ComputationDataHandle ComputationBuilder::ConvWithGeneralPadding(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  return ConvGeneral(lhs, rhs, window_strides, padding,
                     CreateDefaultConvDimensionNumbers(window_strides.size()));
}

bool ComputationBuilder::VerifyConvolution(
    const Shape& lhs_shape, const Shape& rhs_shape,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  if (ShapeUtil::Rank(lhs_shape) != ShapeUtil::Rank(rhs_shape)) {
    NoteError(
        InvalidArgument("Convolution arguments must have same number of "
                        "dimensions. Got: %s and %s",
                        ShapeUtil::HumanString(lhs_shape).c_str(),
                        ShapeUtil::HumanString(rhs_shape).c_str()));
    return false;
  }
  int num_dims = ShapeUtil::Rank(lhs_shape);
  if (num_dims < 2) {
    NoteError(InvalidArgument(
        "Convolution expects argument arrays with >= 3 dimensions. "
        "Got: %s and %s",
        ShapeUtil::HumanString(lhs_shape).c_str(),
        ShapeUtil::HumanString(rhs_shape).c_str()));
    return false;
  }
  int num_spatial_dims = num_dims - 2;

  const auto check_spatial_dimensions =
      [&](const char* const field_name,
          const tensorflow::protobuf::RepeatedField<tensorflow::protobuf_int64>&
              numbers) {
        if (numbers.size() != num_spatial_dims) {
          NoteError(InvalidArgument("Expected %d elements for %s, but got %d.",
                                    num_spatial_dims, field_name,
                                    numbers.size()));
          return false;
        }
        for (int i = 0; i < numbers.size(); ++i) {
          if (numbers.Get(i) < 0 || numbers.Get(i) >= num_dims) {
            NoteError(
                InvalidArgument("Convolution %s[%d] is out of bounds: %lld",
                                field_name, i, numbers.Get(i)));
            return false;
          }
        }
        return true;
      };
  return check_spatial_dimensions(
             "input_spatial_dimensions",
             dimension_numbers.input_spatial_dimensions()) &&
         check_spatial_dimensions(
             "kernel_spatial_dimensions",
             dimension_numbers.kernel_spatial_dimensions()) &&
         check_spatial_dimensions(
             "output_spatial_dimensions",
             dimension_numbers.output_spatial_dimensions());
}

ComputationDataHandle ComputationBuilder::ConvWithGeneralDimensions(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> lhs_shape_or_status = GetShape(lhs);
  if (!lhs_shape_or_status.ok()) {
    first_error_ = lhs_shape_or_status.status();
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> rhs_shape_or_status = GetShape(rhs);
  if (!rhs_shape_or_status.ok()) {
    first_error_ = rhs_shape_or_status.status();
    return ComputationDataHandle();
  }

  std::unique_ptr<Shape> lhs_shape = lhs_shape_or_status.ConsumeValueOrDie();
  std::unique_ptr<Shape> rhs_shape = rhs_shape_or_status.ConsumeValueOrDie();

  if (!VerifyConvolution(*lhs_shape, *rhs_shape, dimension_numbers)) {
    NoteError(InternalError("failed to verify convolution"));
    return ComputationDataHandle();
  }

  std::vector<int64> base_area_dimensions(
      dimension_numbers.input_spatial_dimensions_size());
  for (std::vector<int64>::size_type i = 0; i < base_area_dimensions.size();
       ++i) {
    base_area_dimensions[i] =
        lhs_shape->dimensions(dimension_numbers.input_spatial_dimensions(i));
  }

  std::vector<int64> window_dimensions(
      dimension_numbers.kernel_spatial_dimensions_size());
  for (std::vector<int64>::size_type i = 0; i < window_dimensions.size(); ++i) {
    window_dimensions[i] =
        rhs_shape->dimensions(dimension_numbers.kernel_spatial_dimensions(i));
  }

  return ConvGeneral(lhs, rhs, window_strides,
                     MakePadding(base_area_dimensions, window_dimensions,
                                 window_strides, padding),
                     dimension_numbers);
}

ComputationDataHandle ComputationBuilder::ConvGeneral(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  return ConvGeneralDilated(lhs, rhs, window_strides, padding, {}, {},
                            dimension_numbers);
}

ComputationDataHandle ComputationBuilder::ConvGeneralDilated(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    tensorflow::gtl::ArraySlice<int64> lhs_dilation,
    tensorflow::gtl::ArraySlice<int64> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> lhs_shape_or_status = GetShape(lhs);
  if (!lhs_shape_or_status.ok()) {
    first_error_ = lhs_shape_or_status.status();
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> rhs_shape_or_status = GetShape(rhs);
  if (!rhs_shape_or_status.ok()) {
    first_error_ = rhs_shape_or_status.status();
    return ComputationDataHandle();
  }

  std::unique_ptr<Shape> lhs_shape = lhs_shape_or_status.ConsumeValueOrDie();
  std::unique_ptr<Shape> rhs_shape = rhs_shape_or_status.ConsumeValueOrDie();
  if (!VerifyConvolution(*lhs_shape, *rhs_shape, dimension_numbers)) {
    // Error is recorded in VerifyConvolution.
    return ComputationDataHandle();
  }

  std::vector<int64> window_dimensions(
      dimension_numbers.kernel_spatial_dimensions_size());
  for (std::vector<int64>::size_type i = 0; i < window_dimensions.size(); ++i) {
    window_dimensions[i] =
        rhs_shape->dimensions(dimension_numbers.kernel_spatial_dimensions(i));
  }

  ConvolveRequest request;
  *request.mutable_lhs() = lhs;
  *request.mutable_rhs() = rhs;
  *request.mutable_dimension_numbers() = dimension_numbers;

  if (!MakeWindow(window_dimensions, window_strides, padding, lhs_dilation,
                  rhs_dilation, request.mutable_window())) {
    // Error is recorded in MakeWindow.
    return ComputationDataHandle();
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_convolve_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making convolve request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Infeed(const Shape& shape,
                                                 const string& config) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  InfeedRequest request;
  *request.mutable_shape() = shape;
  *request.mutable_config() = config;
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_infeed_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making infeed op request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

void ComputationBuilder::Outfeed(const ComputationDataHandle& operand,
                                 const Shape& shape,
                                 const string& outfeed_config) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return;
  }

  OutfeedRequest request;
  request.set_outfeed_config(outfeed_config);
  *request.mutable_operand() = operand;
  *request.mutable_shape() = shape;
  OpRequest op_request;
  *op_request.mutable_outfeed_request() = request;
  *op_request.mutable_computation() = computation_.handle();
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making outfeed op request";
  tensorflow::Status s = client_->stub()->Op(&op_request, &response);

  if (!s.ok()) {
    NoteError(s);
    return;
  }
}

ComputationDataHandle ComputationBuilder::Call(
    const Computation& computation,
    tensorflow::gtl::ArraySlice<ComputationDataHandle> operands) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  CallRequest request;
  *request.mutable_to_apply() = computation.handle();
  for (const ComputationDataHandle& operand : operands) {
    *request.add_operands() = operand;
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_call_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making call op request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::CustomCall(
    const string& call_target_name,
    tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
    const Shape& shape) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  CustomCallRequest request;
  request.set_call_target_name(call_target_name);
  for (const ComputationDataHandle& operand : operands) {
    *request.add_operands() = operand;
  }
  *request.mutable_shape() = shape;
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_custom_call_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making custom call op request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Complex(
    const ComputationDataHandle& real, const ComputationDataHandle& imag,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_COMPLEX, real, imag, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Conj(
    const ComputationDataHandle& operand) {
  return Complex(Real(operand), Neg(Imag(operand)));
}

ComputationDataHandle ComputationBuilder::Add(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_ADD, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Sub(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_SUB, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Mul(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_MUL, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Div(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_DIV, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Rem(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_REM, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Max(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_MAX, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Min(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_MIN, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::And(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_AND, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Or(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_OR, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Not(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_NOT, operand);
}

ComputationDataHandle ComputationBuilder::ShiftLeft(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_SHIFT_LEFT, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::ShiftRightArithmetic(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_SHIFT_RIGHT_ARITHMETIC, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::ShiftRightLogical(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_SHIFT_RIGHT_LOGICAL, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Abs(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_ABS, operand);
}

ComputationDataHandle ComputationBuilder::Atan2(
    const ComputationDataHandle& y, const ComputationDataHandle& x,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_ATAN2, y, x, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::Exp(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_EXP, operand);
}

ComputationDataHandle ComputationBuilder::Floor(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_FLOOR, operand);
}

ComputationDataHandle ComputationBuilder::Ceil(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_CEIL, operand);
}

ComputationDataHandle ComputationBuilder::Round(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_ROUND_NEAREST_AFZ, operand);
}

ComputationDataHandle ComputationBuilder::Log(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_LOG, operand);
}

ComputationDataHandle ComputationBuilder::Sign(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_SIGN, operand);
}

ComputationDataHandle ComputationBuilder::Cos(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_COS, operand);
}

ComputationDataHandle ComputationBuilder::Sin(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_SIN, operand);
}

ComputationDataHandle ComputationBuilder::Tanh(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_TANH, operand);
}

ComputationDataHandle ComputationBuilder::Real(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_REAL, operand);
}

ComputationDataHandle ComputationBuilder::Imag(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_IMAG, operand);
}

ComputationDataHandle ComputationBuilder::IsFinite(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_IS_FINITE, operand);
}

ComputationDataHandle ComputationBuilder::Transpose(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> permutation) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  TransposeRequest* request = op_request.mutable_transpose_request();
  *request->mutable_operand() = operand;
  for (int64 dimension : permutation) {
    request->add_dimensions(dimension);
  }
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making transpose request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Rev(
    const ComputationDataHandle& operand,
    tensorflow::gtl::ArraySlice<int64> dimensions) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  ReverseRequest request;
  *request.mutable_operand() = operand;
  for (int64 dimension : dimensions) {
    request.add_dimensions(dimension);
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_reverse_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making reverse op request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Sort(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_SORT, operand);
}

ComputationDataHandle ComputationBuilder::SqrtF32(
    const ComputationDataHandle& operand) {
  return BinaryOp(BINOP_POW, operand, ConstantR0<float>(0.5),
                  /*broadcast_dimensions=*/{});
}

ComputationDataHandle ComputationBuilder::Pow(
    const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(BINOP_POW, lhs, rhs, broadcast_dimensions);
}

ComputationDataHandle ComputationBuilder::ConvertElementType(
    const ComputationDataHandle& operand, PrimitiveType new_element_type) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> shape_status = GetShape(operand);
  if (!shape_status.ok()) {
    first_error_ = shape_status.status();
    return ComputationDataHandle();
  }
  std::unique_ptr<Shape> original = shape_status.ConsumeValueOrDie();

  ConvertRequest request;
  *request.mutable_operand() = operand;
  request.set_new_element_type(new_element_type);
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_convert_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making convert request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::BitcastConvertType(
    const ComputationDataHandle& operand, PrimitiveType new_element_type) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> shape_status = GetShape(operand);
  if (!shape_status.ok()) {
    first_error_ = shape_status.status();
    return ComputationDataHandle();
  }
  std::unique_ptr<Shape> original = shape_status.ConsumeValueOrDie();

  ConvertRequest request;
  *request.mutable_operand() = operand;
  request.set_new_element_type(new_element_type);
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_bitcast_convert_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making bitcast convert request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::SquareF32(
    const ComputationDataHandle& operand) {
  return BinaryOp(BINOP_POW, operand, ConstantR0<float>(2.0),
                  /*broadcast_dimensions=*/{});
}

ComputationDataHandle ComputationBuilder::ReciprocalF32(
    const ComputationDataHandle& operand) {
  return BinaryOp(BINOP_POW, operand, ConstantR0<float>(-1.0),
                  /*broadcast_dimensions=*/{});
}

ComputationDataHandle ComputationBuilder::Neg(
    const ComputationDataHandle& operand) {
  return UnaryOp(UNOP_NEGATE, operand);
}

ComputationDataHandle ComputationBuilder::Clamp(
    const ComputationDataHandle& min, const ComputationDataHandle& operand,
    const ComputationDataHandle& max) {
  return TernaryOp(TRIOP_CLAMP, min, operand, max);
}

ComputationDataHandle ComputationBuilder::UnaryOp(
    UnaryOperation unop, const ComputationDataHandle& operand) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  UnaryOpRequest request;
  request.set_unop(unop);
  *request.mutable_operand() = operand;
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_unary_op_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making unop request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::BinaryOp(
    BinaryOperation binop, const ComputationDataHandle& lhs,
    const ComputationDataHandle& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  BinaryOpRequest request;
  request.set_binop(binop);
  *request.mutable_lhs() = lhs;
  *request.mutable_rhs() = rhs;
  for (int64 dimension : broadcast_dimensions) {
    request.add_broadcast_dimensions(dimension);
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_binary_op_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making binop request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::RngOp(
    RandomDistribution distribution,
    tensorflow::gtl::ArraySlice<ComputationDataHandle> parameters,
    const Shape& shape) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  RngRequest request;
  request.set_distribution(distribution);
  for (const ComputationDataHandle& param : parameters) {
    *request.add_parameter() = param;
  }
  *request.mutable_shape() = shape;
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_rng_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making rngop request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::TernaryOp(
    TernaryOperation triop, const ComputationDataHandle& lhs,
    const ComputationDataHandle& rhs, const ComputationDataHandle& ehs) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  TernaryOpRequest request;
  request.set_triop(triop);
  *request.mutable_lhs() = lhs;
  *request.mutable_rhs() = rhs;
  *request.mutable_ehs() = ehs;
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_ternary_op_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making triop request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

Status ComputationBuilder::SetReturnValue(
    const ComputationDataHandle& operand) {
  if (!first_error_.ok()) {
    return first_error_;
  }

  SetReturnValueRequest request;
  *request.mutable_computation() = computation_.handle();
  *request.mutable_operand() = operand;

  SetReturnValueResponse response;

  VLOG(2) << "making set-handle-to-execute request";
  Status s = client_->stub()->SetReturnValue(&request, &response);
  VLOG(2) << "done with request";

  if (!s.ok()) {
    NoteError(s);
    return first_error_;
  }

  return Status::OK();
}

StatusOr<bool> ComputationBuilder::IsConstant(
    const ComputationDataHandle& operand, int64 num_parameters) {
  if (!first_error_.ok()) {
    return first_error_;
  }

  IsConstantRequest request;
  *request.mutable_computation() = computation_.handle();
  *request.mutable_operand() = operand;
  request.set_num_parameters(num_parameters);
  IsConstantResponse response;

  VLOG(2) << "making IsConstant request";
  Status s = client_->stub()->IsConstant(&request, &response);
  VLOG(2) << "done with request";

  if (!s.ok()) {
    return s;
  }
  return response.is_constant();
}

StatusOr<std::unique_ptr<Literal>> ComputationBuilder::ComputeConstant(
    const ComputationDataHandle& operand, const Layout* output_layout,
    tensorflow::gtl::ArraySlice<Literal> parameters) {
  if (!first_error_.ok()) {
    return first_error_;
  }

  ComputeConstantRequest request;
  *request.mutable_computation() = computation_.handle();
  *request.mutable_operand() = operand;
  if (output_layout != nullptr) {
    *request.mutable_output_layout() = *output_layout;
  }
  for (const auto& param : parameters) {
    *request.add_parameters() = param.ToProto();
  }

  ComputeConstantResponse response;

  VLOG(2) << "making compute-constant request";
  Status s = client_->stub()->ComputeConstant(&request, &response);
  VLOG(2) << "done with request";

  if (!s.ok()) {
    return s;
  }

  VLOG(3) << "ComputeConstant: {" << response.DebugString() << "}";

  if (!response.has_literal()) {
    return InternalError(
        "no computed literal in the provided response in ComputeConstant "
        "request");
  }
  return MakeUnique<Literal>(response.literal());
}

ComputationDataHandle ComputationBuilder::Map(
    tensorflow::gtl::ArraySlice<ComputationDataHandle> operands,
    const Computation& computation,
    tensorflow::gtl::ArraySlice<int64> dimensions,
    tensorflow::gtl::ArraySlice<ComputationDataHandle> static_operands) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  MapRequest request;
  for (const ComputationDataHandle& operand : operands) {
    *request.add_operands() = operand;
  }
  *request.mutable_to_apply() = computation.handle();
  for (int64 dimension : dimensions) {
    request.add_dimensions(dimension);
  }
  for (const ComputationDataHandle& sop : static_operands) {
    *request.add_static_operands() = sop;
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_map_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making Map request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::RngNormal(
    const ComputationDataHandle& mu, const ComputationDataHandle& sigma,
    const Shape& shape) {
  return RngOp(RandomDistribution::RNG_NORMAL, {mu, sigma}, shape);
}

ComputationDataHandle ComputationBuilder::RngUniform(
    const ComputationDataHandle& a, const ComputationDataHandle& b,
    const Shape& shape) {
  return RngOp(RandomDistribution::RNG_UNIFORM, {a, b}, shape);
}

ComputationDataHandle ComputationBuilder::RngBernoulli(
    const ComputationDataHandle& mean, const Shape& shape) {
  return RngOp(RandomDistribution::RNG_BERNOULLI, {mean}, shape);
}

ComputationDataHandle ComputationBuilder::While(
    const Computation& condition, const Computation& body,
    const ComputationDataHandle& init) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  WhileRequest request;
  *request.mutable_condition() = condition.handle();
  *request.mutable_body() = body.handle();
  *request.mutable_init() = init;
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_while_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making while request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Conditional(
    const ComputationDataHandle& predicate,
    const ComputationDataHandle& true_operand,
    const Computation& true_computation,
    const ComputationDataHandle& false_operand,
    const Computation& false_computation) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  ConditionalRequest request;
  *request.mutable_predicate() = predicate;
  *request.mutable_true_operand() = true_operand;
  *request.mutable_true_computation() = true_computation.handle();
  *request.mutable_false_operand() = false_operand;
  *request.mutable_false_computation() = false_computation.handle();
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_conditional_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making conditional op request";
  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::Reduce(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& init_value, const Computation& computation,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  ReduceRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_init_value() = init_value;
  for (int64 dimension : dimensions_to_reduce) {
    request.add_dimensions(dimension);
  }
  *request.mutable_to_apply() = computation.handle();
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_reduce_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making reduce request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::ReduceAll(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& init_value, const Computation& computation) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> shape = GetShape(operand);
  if (!shape.ok()) {
    first_error_ = shape.status();
    return ComputationDataHandle();
  }

  std::vector<int64> all_dimnos(ShapeUtil::Rank(*shape.ValueOrDie()));
  std::iota(all_dimnos.begin(), all_dimnos.end(), 0);
  return Reduce(operand, init_value, computation, all_dimnos);
}

ComputationDataHandle ComputationBuilder::ReduceWindow(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& init_value, const Computation& computation,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding) {
  if (!first_error_.ok()) {
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> shape = GetShape(operand);
  if (!shape.ok()) {
    first_error_ = shape.status();
    return ComputationDataHandle();
  }

  Status padding_valid =
      ValidatePaddingValues(AsInt64Slice(shape.ValueOrDie()->dimensions()),
                            window_dimensions, window_strides);
  if (!padding_valid.ok()) {
    first_error_ = padding_valid;
    return ComputationDataHandle();
  }

  std::vector<std::pair<int64, int64>> padding_values =
      MakePadding(AsInt64Slice(shape.ValueOrDie()->dimensions()),
                  window_dimensions, window_strides, padding);
  return ReduceWindowWithGeneralPadding(operand, init_value, computation,
                                        window_dimensions, window_strides,
                                        padding_values);
}

ComputationDataHandle ComputationBuilder::ReduceWindowWithGeneralPadding(
    const ComputationDataHandle& operand,
    const ComputationDataHandle& init_value, const Computation& computation,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  ReduceWindowRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_to_apply() = computation.handle();
  *request.mutable_init_value() = init_value;

  if (!MakeWindow(window_dimensions, window_strides, padding, {}, {},
                  request.mutable_window())) {
    NoteError(InternalError("failed to make window"));
    return ComputationDataHandle();
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_reduce_window_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making reduce-window request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::BatchNormTraining(
    const ComputationDataHandle& operand, const ComputationDataHandle& scale,
    const ComputationDataHandle& offset, float epsilon, int64 feature_index) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }
  BatchNormTrainingRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_scale() = scale;
  *request.mutable_offset() = offset;
  request.set_epsilon(epsilon);
  request.set_feature_index(feature_index);

  OpRequest op_request;
  *op_request.mutable_batch_norm_training_request() = request;
  *op_request.mutable_computation() = computation_.handle();
  AddCommonFieldsToOpRequest(&op_request);

  OpResponse response;

  VLOG(2) << "making BatchNormTraining request";

  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::BatchNormInference(
    const ComputationDataHandle& operand, const ComputationDataHandle& scale,
    const ComputationDataHandle& offset, const ComputationDataHandle& mean,
    const ComputationDataHandle& variance, float epsilon, int64 feature_index) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }
  BatchNormInferenceRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_scale() = scale;
  *request.mutable_offset() = offset;
  *request.mutable_mean() = mean;
  *request.mutable_variance() = variance;
  request.set_epsilon(epsilon);
  request.set_feature_index(feature_index);

  OpRequest op_request;
  *op_request.mutable_batch_norm_inference_request() = request;
  *op_request.mutable_computation() = computation_.handle();
  AddCommonFieldsToOpRequest(&op_request);

  OpResponse response;

  VLOG(2) << "making BatchNormInference request";

  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::BatchNormGrad(
    const ComputationDataHandle& operand, const ComputationDataHandle& scale,
    const ComputationDataHandle& mean, const ComputationDataHandle& var,
    const ComputationDataHandle& grad_output, float epsilon,
    int64 feature_index) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }
  BatchNormGradRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_scale() = scale;
  *request.mutable_mean() = mean;
  *request.mutable_variance() = var;
  *request.mutable_grad_output() = grad_output;
  request.set_epsilon(epsilon);
  request.set_feature_index(feature_index);

  OpRequest op_request;
  *op_request.mutable_batch_norm_grad_request() = request;
  *op_request.mutable_computation() = computation_.handle();
  AddCommonFieldsToOpRequest(&op_request);

  OpResponse response;

  VLOG(2) << "making BatchNormGrad request";

  Status s = client_->stub()->Op(&op_request, &response);

  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::CrossReplicaSum(
    const ComputationDataHandle& operand) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  CrossReplicaSumRequest request;
  *request.mutable_operand() = operand;
  OpRequest op_request;
  *op_request.mutable_cross_replica_sum_request() = request;
  *op_request.mutable_computation() = computation_.handle();
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making cross-replica-sum request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::SelectAndScatter(
    const ComputationDataHandle& operand, const Computation& select,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
    const ComputationDataHandle& source,
    const ComputationDataHandle& init_value, const Computation& scatter) {
  if (!first_error_.ok()) {
    return ComputationDataHandle();
  }

  StatusOr<std::unique_ptr<Shape>> shape = GetShape(operand);
  if (!shape.ok()) {
    first_error_ = shape.status();
    return ComputationDataHandle();
  }
  return SelectAndScatterWithGeneralPadding(
      operand, select, window_dimensions, window_strides,
      MakePadding(AsInt64Slice(shape.ValueOrDie()->dimensions()),
                  window_dimensions, window_strides, padding),
      source, init_value, scatter);
}

ComputationDataHandle ComputationBuilder::SelectAndScatterWithGeneralPadding(
    const ComputationDataHandle& operand, const Computation& select,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    const ComputationDataHandle& source,
    const ComputationDataHandle& init_value, const Computation& scatter) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  SelectAndScatterRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_select() = select.handle();
  *request.mutable_source() = source;
  *request.mutable_init_value() = init_value;
  *request.mutable_scatter() = scatter.handle();

  if (!MakeWindow(window_dimensions, window_strides, padding, {}, {},
                  request.mutable_window())) {
    NoteError(InternalError("failed to make window"));
    return ComputationDataHandle();
  }
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_select_and_scatter_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making select-and-scatter request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

ComputationDataHandle ComputationBuilder::ReducePrecision(
    const ComputationDataHandle& operand, const int exponent_bits,
    const int mantissa_bits) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  ReducePrecisionRequest request;
  *request.mutable_operand() = operand;
  request.set_exponent_bits(exponent_bits);
  request.set_mantissa_bits(mantissa_bits);
  OpRequest op_request;
  *op_request.mutable_computation() = computation_.handle();
  *op_request.mutable_reduce_precision_request() = request;
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making reduce-precision request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

void ComputationBuilder::Send(const ComputationDataHandle& operand,
                              const ChannelHandle& handle) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return;
  }

  SendRequest request;
  *request.mutable_operand() = operand;
  *request.mutable_channel_handle() = handle;
  OpRequest op_request;
  *op_request.mutable_send_request() = request;
  *op_request.mutable_computation() = computation_.handle();
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making send request";
  Status s = client_->stub()->Op(&op_request, &response);
  VLOG(2) << "done with op request";

  if (!s.ok()) {
    NoteError(s);
    return;
  }
}

ComputationDataHandle ComputationBuilder::Recv(const Shape& shape,
                                               const ChannelHandle& handle) {
  if (!first_error_.ok() || !PrepareComputation().ok()) {
    return ComputationDataHandle();
  }

  RecvRequest request;
  *request.mutable_shape() = shape;
  *request.mutable_channel_handle() = handle;
  OpRequest op_request;
  *op_request.mutable_recv_request() = request;
  *op_request.mutable_computation() = computation_.handle();
  AddCommonFieldsToOpRequest(&op_request);
  OpResponse response;

  VLOG(2) << "making recv request";
  Status s = client_->stub()->Op(&op_request, &response);
  return ParseOpResponse(s, &response);
}

Computation ComputationBuilder::BuildAndNoteError() {
  DCHECK(parent_builder_ != nullptr);
  auto build_status = Build();
  if (!build_status.ok()) {
    parent_builder_->NoteError(
        AddStatus(build_status.status(),
                  tensorflow::strings::StrCat("error from: ", name_)));
    return Computation();
  }
  return build_status.ConsumeValueOrDie();
}

StatusOr<Computation> ComputationBuilder::Build() {
  if (!first_error_.ok()) {
    string backtrace;
    first_error_backtrace_.Dump(tensorflow::DebugWriteToString, &backtrace);
    return AppendStatus(first_error_, backtrace);
  }

  if (computation_.IsNull()) {
    return FailedPrecondition("no computation was built");
  }

  return {std::move(computation_)};
}

void ComputationBuilder::AddCommonFieldsToOpRequest(OpRequest* request) const {
  *request->mutable_metadata() = metadata_;
  if (sharding_) {
    *request->mutable_sharding() = *sharding_;
  }
}

/* static */ ConvolutionDimensionNumbers
ComputationBuilder::CreateDefaultConvDimensionNumbers(int num_spatial_dims) {
  ConvolutionDimensionNumbers dimension_numbers;
  dimension_numbers.set_input_batch_dimension(kConvBatchDimension);
  dimension_numbers.set_input_feature_dimension(kConvFeatureDimension);
  dimension_numbers.set_output_batch_dimension(kConvBatchDimension);
  dimension_numbers.set_output_feature_dimension(kConvFeatureDimension);
  dimension_numbers.set_kernel_output_feature_dimension(
      kConvKernelOutputDimension);
  dimension_numbers.set_kernel_input_feature_dimension(
      kConvKernelInputDimension);
  for (int i = 0; i < num_spatial_dims; ++i) {
    dimension_numbers.add_input_spatial_dimensions(i + 2);
    dimension_numbers.add_kernel_spatial_dimensions(i + 2);
    dimension_numbers.add_output_spatial_dimensions(i + 2);
  }
  return dimension_numbers;
}

/* static */ StatusOr<ConvolutionDimensionNumbers>
ComputationBuilder::CreateConvDimensionNumbers(
    int64 input_batch, int64 input_feature, int64 input_first_spatial,
    int64 input_second_spatial, int64 output_batch, int64 output_feature,
    int64 output_first_spatial, int64 output_second_spatial,
    int64 kernel_output_feature, int64 kernel_input_feature,
    int64 kernel_first_spatial, int64 kernel_second_spatial) {
  if (std::set<int64>({input_batch, input_feature, input_first_spatial,
                       input_second_spatial})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the input are not unique: (%lld, %lld, %lld, "
        "%lld)",
        input_batch, input_feature, input_first_spatial, input_second_spatial);
  }
  if (std::set<int64>({kernel_output_feature, kernel_input_feature,
                       kernel_first_spatial, kernel_second_spatial})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the weight are not unique: (%lld, %lld, %lld, "
        "%lld)",
        kernel_output_feature, kernel_input_feature, kernel_first_spatial,
        kernel_second_spatial);
  }
  if (std::set<int64>({output_batch, output_feature, output_first_spatial,
                       output_second_spatial})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the output are not unique: (%lld, %lld, %lld, "
        "%lld)",
        output_batch, output_feature, output_first_spatial,
        output_second_spatial);
  }
  ConvolutionDimensionNumbers dimension_numbers;
  dimension_numbers.set_input_batch_dimension(input_batch);
  dimension_numbers.set_input_feature_dimension(input_feature);
  dimension_numbers.add_input_spatial_dimensions(input_first_spatial);
  dimension_numbers.add_input_spatial_dimensions(input_second_spatial);
  dimension_numbers.set_kernel_output_feature_dimension(kernel_output_feature);
  dimension_numbers.set_kernel_input_feature_dimension(kernel_input_feature);
  dimension_numbers.add_kernel_spatial_dimensions(kernel_first_spatial);
  dimension_numbers.add_kernel_spatial_dimensions(kernel_second_spatial);
  dimension_numbers.set_output_batch_dimension(output_batch);
  dimension_numbers.set_output_feature_dimension(output_feature);
  dimension_numbers.add_output_spatial_dimensions(output_first_spatial);
  dimension_numbers.add_output_spatial_dimensions(output_second_spatial);
  return dimension_numbers;
}

}  // namespace xla
