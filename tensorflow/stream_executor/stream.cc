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

#include "tensorflow/stream_executor/stream.h"

#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/host_or_device_scalar.h"
#include "tensorflow/stream_executor/lib/stacktrace.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/rng.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace stream_executor {

namespace {
// Code to turn parameters to functions on stream into strings that
// will be VLOG'ed. We need overloads, instead of
// e.g. BatchDescriptorToVlogString(), as the code that calls these
// functions does not know what the type of the parameter is.
std::string ToVlogString(const dnn::BatchDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::FilterDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::ConvolutionDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::PoolingDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(const dnn::NormalizeDescriptor &descriptor) {
  return descriptor.ToShortString();
}

std::string ToVlogString(dnn::ActivationMode mode) {
  return dnn::ActivationModeString(mode);
}

std::string ToVlogString(const dnn::AlgorithmConfig &algo_config) {
  return algo_config.ToString();
}

std::string ToVlogString(dnn::ElementwiseOperation op) {
  return dnn::ElementwiseOperationString(op);
}

std::string ToVlogString(dnn::QuantizedActivationMode mode) {
  return dnn::QuantizedActivationModeString(mode);
}

std::string ToVlogString(blas::Transpose t) { return blas::TransposeString(t); }

std::string ToVlogString(blas::UpperLower ul) {
  return blas::UpperLowerString(ul);
}

std::string ToVlogString(blas::Diagonal d) { return blas::DiagonalString(d); }

std::string ToVlogString(blas::Side s) { return blas::SideString(s); }

std::string ToVlogString(blas::ComputationType ty) {
  return blas::ComputationTypeString(ty);
}

std::string ToVlogString(const void *ptr) {
  if (ptr == nullptr) {
    return "null";
  }

  // StrCat does not convert pointers to text.
  std::ostringstream out;
  out << ptr;
  return out.str();
}

template <class T>
std::string ToVlogString(const std::complex<T> &c) {
  // StrCat does not convert std::complex to text.
  std::ostringstream out;
  out << c;
  return out.str();
}

template <class T>
std::string ToVlogString(const std::function<T> &f) {
  return f == nullptr ? "null" : "<non-null function>";
}

std::string ToVlogString(const DeviceMemoryBase &memory) {
  return ToVlogString(memory.opaque());
}

std::string ToVlogString(const DeviceMemoryBase *memory) {
  return memory == nullptr ? "null" : ToVlogString(*memory);
}

std::string ToVlogString(const Eigen::half &h) {
  return absl::StrCat(static_cast<float>(h));
}

std::string ToVlogString(int i) { return absl::StrCat(i); }

std::string ToVlogString(uint32 i) { return absl::StrCat(i); }

std::string ToVlogString(uint64_t i) { return absl::StrCat(i); }

std::string ToVlogString(int64_t i) { return absl::StrCat(i); }

std::string ToVlogString(float f) { return absl::StrCat(f); }

std::string ToVlogString(double d) { return absl::StrCat(d); }

template <typename T>
std::string ToVlogString(const HostOrDeviceScalar<T> &memory_or_constant) {
  if (memory_or_constant.is_pointer()) {
    return ToVlogString(memory_or_constant.pointer());
  }
  return ToVlogString(memory_or_constant.value());
}

template <class T>
std::string ToVlogString(port::ArraySlice<T> elements) {
  std::string str = absl::StrCat(
      ToVlogString(reinterpret_cast<const void *>(elements.data())), "[",
      elements.size(), "]{");
  const char *separator = "";
  size_t max_to_show = std::numeric_limits<size_t>::max();
  if (!VLOG_IS_ON(2)) {
    max_to_show = 5;
  } else if (!VLOG_IS_ON(3)) {
    max_to_show = 20;
  } else if (!VLOG_IS_ON(11)) {
    max_to_show = 1000;
  }
  for (size_t i = 0; i < elements.size(); ++i) {
    if (i == max_to_show) {
      str += ", ...";
      break;
    }
    absl::StrAppend(&str, separator, ToVlogString(elements[i]));
    separator = ", ";
  }
  str += "}";
  return str;
}

template <class T>
std::string ToVlogString(port::MutableArraySlice<T> elements) {
  return ToVlogString(port::ArraySlice<T>(elements));
}

std::string ToVlogString(dnn::DepthToSpaceLayout depth_to_space_layout) {
  switch (depth_to_space_layout) {
    case dnn::DepthToSpaceLayout::DepthHeightWidth:
      return "DepthToSpaceLayout::DepthHeightWidth";
  }
  return "unknown DepthToSpaceLayout";
}

std::string ToVlogString(dnn::DataType data_type) {
  switch (data_type) {
    case dnn::DataType::kFloat:
      return "dnn::DataType::kFloat";
    case dnn::DataType::kDouble:
      return "dnn::DataType::kDouble";
    case dnn::DataType::kHalf:
      return "dnn::DataType::kHalf";
    case dnn::DataType::kInt8:
      return "dnn::DataType::kInt8";
    case dnn::DataType::kInt32:
      return "dnn::DataType::kInt32";
    default:
      return "unknown DataType";
  }
}

// Used together with PARAM to VLOG calls made to the stream. Intended
// to be used like this:
//
//   VLOG(1) << CallStr("MyFunction", this, {PARAM(a), PARAM(b)});
//
// where a and b are the parameters to MyFunction.
//
// See VLOG_CALL for a short-hand for this. This way of doing it saves
// a tremendous amount of boilerplate code given how many functions
// there are on Stream and how many parameters they each have.
std::string CallStr(const char *function_name, Stream *stream,
                    std::vector<std::pair<const char *, std::string>> params) {
  // Do not call this function unless VLOG is on since just
  // constructing all the strings in params is expensive.
  CHECK(VLOG_IS_ON(1));

  std::string str = absl::StrCat(stream->DebugStreamPointers(),
                                 " Called Stream::", function_name, "(");
  const char *separator = "";
  for (const auto &param : params) {
    absl::StrAppend(&str, separator, param.first, "=", param.second);
    separator = ", ";
  }
  absl::StrAppend(&str, ")");
  if (VLOG_IS_ON(10)) {
    absl::StrAppend(&str, " ", port::CurrentStackTrace(), "\n");
  }
  return str;
}

// Use this macro to avoid having to type every parameter twice to log
// it with VLOG and CallStr.
#define PARAM(parameter)                \
  {                                     \
#parameter, ToVlogString(parameter) \
  }

// Use this macro to avoid having to type out the name of each
// function and to save some boilerplate. Intended to be used like this:
//
//   VLOG_CALL(PARAM(a), PARAM(b))
//
// This saves a tremendous amount of boilerplate compared to the alternative:
//
//   VLOG(1) << "Calling MyFunction(a=" << ToVlogString(a)
//           << ", b=" << ToVlogString(b);
//
// Note here that most of the parameter names are not short and that
// most of the functions take many more than 2 parameters.
#define VLOG_CALL(...) VLOG(1) << CallStr(__func__, this, {__VA_ARGS__})

}  // namespace

Stream::Stream(StreamExecutor *parent)
    : parent_(parent),
      implementation_(parent->implementation()->GetStreamImplementation()),
      allocated_(false),
      status_(port::InternalError("Uninitialized stream")),
      temporary_memory_manager_(this) {
  VLOG_CALL(PARAM(parent));
}

Stream::~Stream() {
  VLOG_CALL();

  // Ensure the stream is completed.
  auto status = BlockHostUntilDone();
  if (!status.ok()) {
    LOG(WARNING) << "Error blocking host until done in stream destructor: "
                 << status;
  }
  temporary_memory_manager_.ForceDeallocateAll();
  RunAfterBlockHostUntilDoneCallbacks();

  if (allocated_) {
    parent_->DeallocateStream(this);
  }
}

port::Status Stream::RefreshStatus() {
  port::Status status = parent_->GetStatus(this);
  // We should not put the stream in an error state, just because the GetStatus
  // method is unimplemented.
  if (status != port::Status(port::error::UNIMPLEMENTED,
                             "GetStatus is not supported on this executor.")) {
    CheckStatus(status);
  }
  return status;
}

Stream &Stream::Init() {
  VLOG_CALL();

  absl::MutexLock lock(&mu_);
  CHECK_EQ(false, allocated_)
      << "stream appears to already have been initialized";
  CHECK(!status_.ok()) << "stream should be in !ok() state pre-initialization";

  if (parent_->AllocateStream(this)) {
    // Successful initialization!
    allocated_ = true;
    status_ = ::tensorflow::OkStatus();
  } else {
    LOG(ERROR) << "failed to allocate stream during initialization";
  }

  return *this;
}

Stream &Stream::InitTimer(Timer *timer) {
  VLOG_CALL(PARAM(timer));

  CheckError(parent_->AllocateTimer(timer));
  return *this;
}

Stream &Stream::InitWithTimer(Timer *timer) {
  VLOG_CALL(PARAM(timer));

  return Init().InitTimer(timer);
}

Stream &Stream::ThenRecordEvent(Event *event) {
  VLOG_CALL(PARAM(event));

  port::Status status = parent_->RecordEvent(this, event);
  if (!status.ok()) {
    LOG(ERROR) << "Error recording event in stream: " << status.error_message()
               << "; not marking stream as bad, as the Event object may be "
               << "at fault. Monitor for further errors.";
  }

  return *this;
}

Stream &Stream::ThenBatchNormalizationForward(
    const DeviceMemory<float> &x, const DeviceMemory<float> &scale,
    const DeviceMemory<float> &offset,
    const DeviceMemory<float> &estimated_mean,
    const DeviceMemory<float> &estimated_variance,
    const DeviceMemory<float> &side_input, const dnn::BatchDescriptor &x_desc,
    const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
    const double exponential_average_factor,
    dnn::ActivationMode activation_mode, DeviceMemory<float> *y,
    DeviceMemory<float> *batch_mean, DeviceMemory<float> *batch_var,
    DeviceMemory<float> *saved_mean, DeviceMemory<float> *saved_inv_var,
    bool is_training, ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator) {
  VLOG_CALL(PARAM(x), PARAM(scale), PARAM(offset), PARAM(x_desc),
            PARAM(scale_offset_desc), PARAM(epsilon), PARAM(y));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoBatchNormalizationForward(
        this, x, scale, offset, estimated_mean, estimated_variance, side_input,
        x_desc, scale_offset_desc, epsilon, exponential_average_factor,
        activation_mode, y, batch_mean, batch_var, saved_mean, saved_inv_var,
        is_training, reserve_space_allocator, workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenBatchNormalizationBackward(
    const DeviceMemory<float> &y_backprop, const DeviceMemory<float> &x,
    const DeviceMemory<float> &scale, const DeviceMemory<float> &offset,
    const DeviceMemory<float> &mean, const DeviceMemory<float> &inv_var,
    const DeviceMemory<float> &y, const dnn::BatchDescriptor &x_desc,
    const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<float> *x_backprop,
    DeviceMemory<float> *scale_backprop, DeviceMemory<float> *offset_backprop,
    DeviceMemory<float> *side_input_backprop,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator) {
  VLOG_CALL(PARAM(y_backprop), PARAM(x), PARAM(scale), PARAM(x_desc),
            PARAM(scale_offset_desc), PARAM(epsilon), PARAM(x_backprop),
            PARAM(scale_backprop), PARAM(offset_backprop));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoBatchNormalizationBackward(
        this, y_backprop, x, scale, offset, mean, inv_var, y, x_desc,
        scale_offset_desc, epsilon, activation_mode, x_backprop, scale_backprop,
        offset_backprop, side_input_backprop, reserve_space_data,
        workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenBatchNormalizationForward(
    const DeviceMemory<Eigen::half> &x, const DeviceMemory<float> &scale,
    const DeviceMemory<float> &offset,
    const DeviceMemory<float> &estimated_mean,
    const DeviceMemory<float> &estimated_variance,
    const DeviceMemory<Eigen::half> &side_input,
    const dnn::BatchDescriptor &x_desc,
    const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
    const double exponential_average_factor,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half> *y,
    DeviceMemory<float> *batch_mean, DeviceMemory<float> *batch_var,
    DeviceMemory<float> *saved_mean, DeviceMemory<float> *saved_inv_var,
    bool is_training, ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator) {
  VLOG_CALL(PARAM(x), PARAM(scale), PARAM(offset), PARAM(x_desc),
            PARAM(scale_offset_desc), PARAM(epsilon), PARAM(y));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoBatchNormalizationForward(
        this, x, scale, offset, estimated_mean, estimated_variance, side_input,
        x_desc, scale_offset_desc, epsilon, exponential_average_factor,
        activation_mode, y, batch_mean, batch_var, saved_mean, saved_inv_var,
        is_training, reserve_space_allocator, workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenBatchNormalizationBackward(
    const DeviceMemory<Eigen::half> &y_backprop,
    const DeviceMemory<Eigen::half> &x, const DeviceMemory<float> &scale,
    const DeviceMemory<float> &offset, const DeviceMemory<float> &mean,
    const DeviceMemory<float> &inv_var, const DeviceMemory<Eigen::half> &y,
    const dnn::BatchDescriptor &x_desc,
    const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
    dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half> *x_backprop,
    DeviceMemory<float> *scale_backprop, DeviceMemory<float> *offset_backprop,
    DeviceMemory<Eigen::half> *side_input_backprop,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator) {
  VLOG_CALL(PARAM(y_backprop), PARAM(x), PARAM(scale), PARAM(x_desc),
            PARAM(scale_offset_desc), PARAM(epsilon), PARAM(x_backprop),
            PARAM(scale_backprop), PARAM(offset_backprop));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoBatchNormalizationBackward(
        this, y_backprop, x, scale, offset, mean, inv_var, y, x_desc,
        scale_offset_desc, epsilon, activation_mode, x_backprop, scale_backprop,
        offset_backprop, side_input_backprop, reserve_space_data,
        workspace_allocator));

  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenConvolve(
    const dnn::BatchDescriptor &input_descriptor,
    const DeviceMemory<float> &input_data,
    const dnn::FilterDescriptor &filter_descriptor,
    const DeviceMemory<float> &filter_data,
    const dnn::ConvolutionDescriptor &convolution_descriptor,
    const dnn::BatchDescriptor &output_descriptor,
    DeviceMemory<float> *output) {
  if (ok()) {
    CheckError(ConvolveWithAlgorithm(
                   dnn::ConvolutionKind::FORWARD, input_descriptor, input_data,
                   filter_descriptor, filter_data, output_descriptor, *output,
                   convolution_descriptor,
                   /*scratch_allocator=*/nullptr, dnn::AlgorithmConfig(),
                   /*output_profile_result=*/nullptr)
                   .ok());
  }
  return *this;
}

Stream &Stream::ThenConvolveQuantized(
    const dnn::BatchDescriptor &input_descriptor,
    const DeviceMemory<float> &input_data,
    const dnn::FilterDescriptor &filter_descriptor,
    const DeviceMemory<int8> &filter_coefficients,
    const DeviceMemory<float> &coefficient_scales,
    const dnn::ConvolutionDescriptor &convolution_descriptor,
    const dnn::BatchDescriptor &output_descriptor,
    DeviceMemory<float> *output) {
  VLOG_CALL(PARAM(input_descriptor), PARAM(input_data),
            PARAM(filter_descriptor), PARAM(filter_coefficients),
            PARAM(coefficient_scales), PARAM(convolution_descriptor),
            PARAM(output_descriptor), PARAM(output));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoConvolveQuantized(
        this, input_descriptor, input_data, filter_descriptor,
        filter_coefficients, coefficient_scales, convolution_descriptor,
        output_descriptor, output));
  } else {
    SetError();
    LOG(WARNING) << "attempting to perform DNN operation using StreamExecutor "
                    "without DNN support";
  }
  return *this;
}

Stream &Stream::ThenConvolveQuantized(
    const dnn::BatchDescriptor &input_descriptor,
    const DeviceMemory<float> &input_data,
    const dnn::FilterDescriptor &filter_descriptor,
    const DeviceMemory<int16> &filter_coefficients,
    const DeviceMemory<float> &coefficient_scales,
    const dnn::ConvolutionDescriptor &convolution_descriptor,
    const dnn::BatchDescriptor &output_descriptor,
    DeviceMemory<float> *output) {
  VLOG_CALL(PARAM(input_descriptor), PARAM(input_data),
            PARAM(filter_descriptor), PARAM(filter_coefficients),
            PARAM(coefficient_scales), PARAM(convolution_descriptor),
            PARAM(output_descriptor), PARAM(output));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoConvolveQuantized(
        this, input_descriptor, input_data, filter_descriptor,
        filter_coefficients, coefficient_scales, convolution_descriptor,
        output_descriptor, output));
  } else {
    SetError();
    LOG(WARNING) << "attempting to perform DNN operation using StreamExecutor "
                    "without DNN support";
  }
  return *this;
}

Stream &Stream::ThenSeparableConvolve(
    const dnn::BatchDescriptor &batch_descriptor,
    const DeviceMemory<float> &input_data,
    const dnn::FilterDescriptor &filter_descriptor, int depth_multiplier,
    const DeviceMemory<float> &first_weights,
    const DeviceMemory<float> &second_weights,
    const dnn::ConvolutionDescriptor &convolution_descriptor,
    const dnn::BatchDescriptor &output_descriptor,
    DeviceMemory<float> *output) {
  VLOG_CALL(
      PARAM(batch_descriptor), PARAM(input_data), PARAM(filter_descriptor),
      PARAM(depth_multiplier), PARAM(first_weights), PARAM(second_weights),
      PARAM(convolution_descriptor), PARAM(output_descriptor), PARAM(output));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoSeparableConvolve(
        this, batch_descriptor, input_data, filter_descriptor, depth_multiplier,
        first_weights, second_weights, convolution_descriptor,
        output_descriptor, output));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMatMul(const DeviceMemory<float> &input_data,
                           const DeviceMemory<float> &weights,
                           const dnn::BatchDescriptor &input_dimensions,
                           const dnn::BatchDescriptor &output_dimensions,
                           DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(input_data), PARAM(weights), PARAM(input_dimensions),
            PARAM(output_dimensions), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMatMul(this, input_data, weights, input_dimensions,
                             output_dimensions, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMatMulQuantized(
    const DeviceMemory<float> &input_data, const DeviceMemory<int8> &weights,
    const DeviceMemory<float> &weight_scales,
    const dnn::BatchDescriptor &input_dimensions,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(input_data), PARAM(weights), PARAM(weight_scales),
            PARAM(input_dimensions), PARAM(output_dimensions),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMatMulQuantized(this, input_data, weights, weight_scales,
                                      input_dimensions, output_dimensions,
                                      output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMatMulQuantized(
    const DeviceMemory<float> &input_data, const DeviceMemory<int16> &weights,
    const DeviceMemory<float> &weight_scales,
    const dnn::BatchDescriptor &input_dimensions,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(input_data), PARAM(weights), PARAM(weight_scales),
            PARAM(input_dimensions), PARAM(output_dimensions),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMatMulQuantized(this, input_data, weights, weight_scales,
                                      input_dimensions, output_dimensions,
                                      output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenBiasAdd(const DeviceMemory<float> &input_data,
                            const DeviceMemory<float> &biases,
                            const dnn::BatchDescriptor &dimensions,
                            DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(input_data), PARAM(biases), PARAM(dimensions),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(
        dnn->DoBiasAdd(this, input_data, biases, dimensions, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenNormalizeWithDimensions(
    const dnn::NormalizeDescriptor &normalize_descriptor,
    const dnn::BatchDescriptor &dimensions,
    const DeviceMemory<float> &input_data, DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(normalize_descriptor), PARAM(dimensions), PARAM(input_data),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoNormalizeWithDimensions(
        this, normalize_descriptor, dimensions, input_data, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenNormalizeBackwardWithDimensions(
    const dnn::NormalizeDescriptor &normalize_descriptor,
    const dnn::BatchDescriptor &dimensions, const DeviceMemory<float> &raw_data,
    const DeviceMemory<float> &normalized_data,
    const DeviceMemory<float> &normalized_variable_gradient,
    DeviceMemory<float> *raw_variable_gradient,
    ScratchAllocator *workspace_allocator) {
  VLOG_CALL(PARAM(normalize_descriptor), PARAM(dimensions), PARAM(raw_data),
            PARAM(normalized_data), PARAM(normalized_variable_gradient),
            PARAM(raw_variable_gradient), PARAM(workspace_allocator));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoNormalizeBackwardWithDimensions(
        this, normalize_descriptor, dimensions, raw_data, normalized_data,
        normalized_variable_gradient, raw_variable_gradient,
        workspace_allocator));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenActivate(dnn::ActivationMode activation_mode,
                             const dnn::BatchDescriptor &dimensions,
                             const DeviceMemory<float> &input_data,
                             DeviceMemory<float> *output_data) {
  return ThenActivateWithOptions(activation_mode, dimensions, input_data,
                                 output_data, /*options=*/0);
}

Stream &Stream::ThenActivateWithOptions(dnn::ActivationMode activation_mode,
                                        const dnn::BatchDescriptor &dimensions,
                                        const DeviceMemory<float> &input_data,
                                        DeviceMemory<float> *output_data,
                                        uint64_t options) {
  VLOG_CALL(PARAM(activation_mode), PARAM(dimensions), PARAM(input_data),
            PARAM(output_data), PARAM(options));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoActivate(this, activation_mode, dimensions, input_data,
                               output_data, options));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenDepthConcatenate(
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float> *> input_data,
    DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data), PARAM(output_data));

  for (size_t i = 1; i < input_dimensions.size(); ++i) {
    if (input_dimensions[i].count() != input_dimensions[0].count() ||
        input_dimensions[i].height() != input_dimensions[0].height() ||
        input_dimensions[i].width() != input_dimensions[0].width()) {
      SetError();
      LOG(ERROR) << "Incompatible dimensions for depth concatenation.\n"
                 << "input_dimensions[0]: " << input_dimensions[0].ToString()
                 << "input_dimensions[" << i
                 << "]: " << input_dimensions[i].ToString();
      return *this;
    }
  }

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoDepthConcatenate(this, input_dimensions, input_data,
                                       output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenSpaceConcatenate(
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float> *> input_data,
    DeviceMemory<float> *output_data,
    dnn::SpaceConcatenateMode concat_direction) {
  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data), PARAM(output_data));

  // Check that the input dimensions of all the other batches match those of the
  // first batch.
  for (size_t i = 1; i < input_dimensions.size(); ++i) {
    if ((concat_direction == dnn::SpaceConcatenateMode::XDirection) &&
        (input_dimensions[i].count() != input_dimensions[0].count() ||
         input_dimensions[i].height() != input_dimensions[0].height() ||
         input_dimensions[i].feature_map_count() !=
             input_dimensions[0].feature_map_count())) {
      SetError();
      LOG(ERROR) << "Incompatible dimensions for X concatenation.\n"
                 << "input_dimensions[0]: " << input_dimensions[0].ToString()
                 << "input_dimensions[" << i
                 << "]: " << input_dimensions[i].ToString();
      return *this;
    }

    if ((concat_direction == dnn::SpaceConcatenateMode::YDirection) &&
        (input_dimensions[i].count() != input_dimensions[0].count() ||
         input_dimensions[i].width() != input_dimensions[0].width() ||
         input_dimensions[i].feature_map_count() !=
             input_dimensions[0].feature_map_count())) {
      SetError();
      LOG(ERROR) << "Incompatible dimensions for Y concatenation.\n"
                 << "input_dimensions[0]: " << input_dimensions[0].ToString()
                 << "input_dimensions[" << i
                 << "]: " << input_dimensions[i].ToString();
      return *this;
    }
  }
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoSpaceConcatenate(this, input_dimensions, input_data,
                                       output_data, concat_direction));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenReshape(const dnn::BatchDescriptor &input_dimensions,
                            const DeviceMemory<float> &input_data,
                            const dnn::BatchDescriptor &output_dimensions,
                            DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data),
            PARAM(output_dimensions), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoReshape(this, input_dimensions, input_data,
                              output_dimensions, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenDepthToSpace(
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<float> &input_data,
    const dnn::DepthToSpaceLayout &depth_to_space_layout,
    const int sqrt_depth_reduction, DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data),
            PARAM(depth_to_space_layout), PARAM(sqrt_depth_reduction),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoDepthToSpace(this, input_dimensions, input_data,
                                   depth_to_space_layout, sqrt_depth_reduction,
                                   output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenSpaceToDepth(
    const dnn::BatchDescriptor &input_dimensions,
    const DeviceMemory<float> &input_data,
    const dnn::DepthToSpaceLayout &space_to_depth_layout,
    const int sqrt_depth_increase, DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(input_dimensions), PARAM(input_data),
            PARAM(space_to_depth_layout), PARAM(sqrt_depth_increase),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoSpaceToDepth(this, input_dimensions, input_data,
                                   space_to_depth_layout, sqrt_depth_increase,
                                   output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenElementwiseOperate(
    dnn::ElementwiseOperation operation,
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float> *> input_data,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(operation), PARAM(input_dimensions), PARAM(input_data),
            PARAM(output_dimensions), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoElementwiseOperate(this, operation, input_dimensions,
                                         input_data, output_dimensions,
                                         output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenElementwiseOperateScaledQuantized(
    dnn::ElementwiseOperation operation,
    port::ArraySlice<int> input_multiplicands, int output_divisor,
    port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
    port::ArraySlice<const DeviceMemory<float> *> input_data,
    const dnn::BatchDescriptor &output_dimensions,
    DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(operation), PARAM(input_multiplicands), PARAM(output_divisor),
            PARAM(input_dimensions), PARAM(input_data),
            PARAM(output_dimensions), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoElementwiseOperateScaledQuantized(
        this, operation, input_multiplicands, output_divisor, input_dimensions,
        input_data, output_dimensions, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenXYPad(const dnn::BatchDescriptor &dimensions,
                          const DeviceMemory<float> &input_data,
                          int64_t left_pad, int64_t right_pad, int64_t top_pad,
                          int64_t bottom_pad,
                          DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(dimensions), PARAM(input_data), PARAM(left_pad),
            PARAM(right_pad), PARAM(top_pad), PARAM(bottom_pad),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoXYPad(this, dimensions, input_data, left_pad, right_pad,
                            top_pad, bottom_pad, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenXYSlice(const dnn::BatchDescriptor &dimensions,
                            const DeviceMemory<float> &input_data,
                            int64_t left_trim, int64_t right_trim,
                            int64_t top_trim, int64_t bottom_trim,
                            DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(dimensions), PARAM(input_data), PARAM(left_trim),
            PARAM(right_trim), PARAM(top_trim), PARAM(bottom_trim),
            PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoXYSlice(this, dimensions, input_data, left_trim,
                              right_trim, top_trim, bottom_trim, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenXYBroadcast(const dnn::BatchDescriptor &dimensions,
                                const DeviceMemory<float> &input_data,
                                int64_t replicate_x, int64_t replicate_y,
                                DeviceMemory<float> *output_data) {
  VLOG_CALL(PARAM(dimensions), PARAM(input_data), PARAM(replicate_x),
            PARAM(replicate_y), PARAM(output_data));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoXYBroadcast(this, dimensions, input_data, replicate_x,
                                  replicate_y, output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMemcpyD2HQuantized(
    const DeviceMemory<float> &gpu_unquantized_src,
    dnn::QuantizedActivationMode mode, void *host_dst, uint64_t size) {
  VLOG_CALL(PARAM(gpu_unquantized_src), PARAM(mode), PARAM(host_dst),
            PARAM(size));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMemcpyD2HQuantized(this, gpu_unquantized_src, mode,
                                         host_dst, size));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenMemcpyH2DQuantized(
    const void *host_src, uint64_t size, dnn::QuantizedActivationMode mode,
    DeviceMemory<float> *gpu_unquantized_dst) {
  VLOG_CALL(PARAM(host_src), PARAM(size), PARAM(mode),
            PARAM(gpu_unquantized_dst));

  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoMemcpyH2DQuantized(this, host_src, size, mode,
                                         gpu_unquantized_dst));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream *Stream::GetOrCreateSubStream() {
  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::vector<std::unique_ptr<Stream>> bad_streams;

  absl::MutexLock lock(&mu_);

  // Look for the first reusable sub_stream that is ok, dropping !ok sub_streams
  // we encounter along the way.
  for (size_t index = 0; index < sub_streams_.size();) {
    std::pair<std::unique_ptr<Stream>, bool> &pair = sub_streams_[index];
    if (pair.second) {
      // The sub_stream is reusable.
      Stream *sub_stream = pair.first.get();
      if (sub_stream->ok()) {
        VLOG(1) << DebugStreamPointers() << " reusing sub_stream "
                << sub_stream->DebugStreamPointers();
        pair.second = false;
        return sub_stream;
      }

      // The stream is reusable and not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      bad_streams.push_back(std::move(sub_streams_.back().first));
      sub_streams_.pop_back();
      VLOG(1) << DebugStreamPointers() << " dropped !ok sub_stream "
              << sub_stream->DebugStreamPointers();
    } else {
      // The sub_stream is not reusable, move on to the next one.
      ++index;
    }
  }

  // No streams are reusable; create a new stream.
  sub_streams_.emplace_back(std::unique_ptr<Stream>{new Stream{parent_}},
                            false);
  Stream *sub_stream = sub_streams_.back().first.get();
  sub_stream->Init();
  if (!sub_stream->ok()) {
    LOG(ERROR) << "sub-stream failed to be initialized";
  }
  VLOG(1) << DebugStreamPointers() << " created new sub_stream "
          << sub_stream->DebugStreamPointers();

  return sub_stream;
}

void Stream::ReturnSubStream(Stream *sub_stream) {
  // Do not destroy bad streams when holding mu_ because ~Stream() may
  // BlockHostUntilDone and it's host callbacks might attempt to acquire mu_.
  std::unique_ptr<Stream> bad_stream;

  absl::MutexLock lock(&mu_);

  // Look for the sub-stream.
  for (int64_t index = 0, end = sub_streams_.size(); index < end; ++index) {
    std::pair<std::unique_ptr<Stream>, bool> &pair = sub_streams_[index];
    if (pair.first.get() != sub_stream) {
      continue;
    }

    // Found the sub_stream.
    if (sub_stream->ok()) {
      VLOG(1) << DebugStreamPointers() << " returned ok sub_stream "
              << sub_stream->DebugStreamPointers();
      pair.second = true;
    } else {
      // The returned stream is not ok. Streams have a monotonic state
      // machine; the stream will remain in !ok forever. Swap it with the last
      // stream and pop it off.
      VLOG(1) << DebugStreamPointers() << " returned !ok sub_stream "
              << sub_stream->DebugStreamPointers();
      const int64_t last = sub_streams_.size() - 1;
      if (index != last) {
        std::swap(pair, sub_streams_[last]);
      }
      std::swap(bad_stream, sub_streams_.back().first);
      sub_streams_.pop_back();
    }
    return;
  }

  LOG(FATAL) << DebugStreamPointers()
             << " did not create the returned sub-stream "
             << sub_stream->DebugStreamPointers();
}

Stream &Stream::ThenStartTimer(Timer *t) {
  VLOG_CALL(PARAM(t));

  CheckError(parent_->StartTimer(this, t));
  return *this;
}

Stream &Stream::ThenStopTimer(Timer *t) {
  VLOG_CALL(PARAM(t));

  CheckError(parent_->StopTimer(this, t));
  return *this;
}

Stream &Stream::ThenWaitFor(Stream *other) {
  VLOG_CALL(PARAM(other));

  CHECK(this != other) << "stream cannot wait for itself";
  if (ok() && other->ok()) {
    CheckError(parent_->CreateStreamDependency(this, other));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers() << " did not wait for "
              << other->DebugStreamPointers();
  }
  return *this;
}

Stream &Stream::ThenWaitFor(Event *event) {
  VLOG_CALL(PARAM(event));

  if (ok()) {
    port::Status status = parent_->WaitForEvent(this, event);
    if (!status.ok()) {
      LOG(ERROR) << "Error waiting for event in stream: "
                 << status.error_message()
                 << "; not marking stream as bad, as the Event object may be "
                 << "at fault. Monitor for further errors.";
    }
  } else {
    LOG(INFO) << DebugStreamPointers() << " did not wait for an event.";
  }
  return *this;
}

// A functor that implements ThenBlasXXX interfaces, which calls DoBlasXXX
// functions and logs for errors.
template <typename... Args>
struct ThenBlasImpl {
  // blas_func is the DoBlasXXX member function pointer, and args are its
  // arguments except the first one of Stream* type.
  Stream &operator()(Stream *stream,
                     bool (blas::BlasSupport::*blas_func)(Stream *, Args...),
                     Args... args) {
    return Run(stream, blas_func, /*record_error=*/true, args...);
  }

  // Like operator(), but only calls stream->CheckError() if record_error is
  // true.
  Stream &Run(Stream *stream,
              bool (blas::BlasSupport::*blas_func)(Stream *, Args...),
              bool record_error, Args... args);
};

template <typename... Args>
Stream &ThenBlasImpl<Args...>::Run(
    Stream *stream, bool (blas::BlasSupport::*blas_func)(Stream *, Args...),
    bool record_error, Args... args) {
  if (stream->ok()) {
    bool ok;
    if (blas::BlasSupport *blas = stream->parent_->AsBlas()) {
      ok = (blas->*blas_func)(stream, args...);
    } else {
      LOG(WARNING)
          << "attempting to perform BLAS operation using StreamExecutor "
             "without BLAS support";
      ok = false;
    }
    if (record_error) {
      stream->CheckError(ok);
    }
  }
  return *stream;
}

Stream &Stream::ThenBlasAsum(uint64_t elem_count, const DeviceMemory<float> &x,
                             int incx, DeviceMemory<float> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAsum, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasAsum(uint64_t elem_count, const DeviceMemory<double> &x,
                             int incx, DeviceMemory<double> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAsum, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasAsum(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, DeviceMemory<float> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAsum, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasAsum(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, DeviceMemory<double> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAsum, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasAxpy(uint64_t elem_count, float alpha,
                             const DeviceMemory<float> &x, int incx,
                             DeviceMemory<float> *y, int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<uint64_t, float, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAxpy, elem_count, alpha, x, incx,
              y, incy);
}

Stream &Stream::ThenBlasAxpy(uint64_t elem_count, double alpha,
                             const DeviceMemory<double> &x, int incx,
                             DeviceMemory<double> *y, int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<uint64_t, double, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAxpy, elem_count, alpha, x, incx,
              y, incy);
}

Stream &Stream::ThenBlasAxpy(uint64_t elem_count, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, DeviceMemory<std::complex<float>> *y,
                             int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAxpy, elem_count, alpha, x, incx,
              y, incy);
}

Stream &Stream::ThenBlasAxpy(uint64_t elem_count, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, DeviceMemory<std::complex<double>> *y,
                             int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasAxpy, elem_count, alpha, x, incx,
              y, incy);
}

Stream &Stream::ThenBlasCopy(uint64_t elem_count, const DeviceMemory<float> &x,
                             int incx, DeviceMemory<float> *y, int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasCopy, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasCopy(uint64_t elem_count, const DeviceMemory<double> &x,
                             int incx, DeviceMemory<double> *y, int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasCopy, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasCopy(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, DeviceMemory<std::complex<float>> *y,
                             int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasCopy, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasCopy(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, DeviceMemory<std::complex<double>> *y,
                             int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasCopy, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasDot(uint64_t elem_count, const DeviceMemory<float> &x,
                            int incx, const DeviceMemory<float> &y, int incy,
                            DeviceMemory<float> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int,
               const DeviceMemory<float> &, int, DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDot, elem_count, x, incx, y, incy,
              result);
}

Stream &Stream::ThenBlasDot(uint64_t elem_count, const DeviceMemory<double> &x,
                            int incx, const DeviceMemory<double> &y, int incy,
                            DeviceMemory<double> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int,
               const DeviceMemory<double> &, int, DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDot, elem_count, x, incx, y, incy,
              result);
}

Stream &Stream::ThenBlasDotc(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy,
                             DeviceMemory<std::complex<float>> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDotc, elem_count, x, incx, y,
              incy, result);
}

Stream &Stream::ThenBlasDotc(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy,
                             DeviceMemory<std::complex<double>> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDotc, elem_count, x, incx, y,
              incy, result);
}

Stream &Stream::ThenBlasDotu(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy,
                             DeviceMemory<std::complex<float>> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDotu, elem_count, x, incx, y,
              incy, result);
}

Stream &Stream::ThenBlasDotu(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy,
                             DeviceMemory<std::complex<double>> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasDotu, elem_count, x, incx, y,
              incy, result);
}

Stream &Stream::ThenBlasNrm2(uint64_t elem_count, const DeviceMemory<float> &x,
                             int incx, DeviceMemory<float> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasNrm2, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasNrm2(uint64_t elem_count, const DeviceMemory<double> &x,
                             int incx, DeviceMemory<double> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasNrm2, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasNrm2(uint64_t elem_count,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, DeviceMemory<float> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasNrm2, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasNrm2(uint64_t elem_count,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, DeviceMemory<double> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasNrm2, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasRot(uint64_t elem_count, DeviceMemory<float> *x,
                            int incx, DeviceMemory<float> *y, int incy, float c,
                            float s) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(c), PARAM(s));

  ThenBlasImpl<uint64_t, DeviceMemory<float> *, int, DeviceMemory<float> *, int,
               float, float>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRot, elem_count, x, incx, y, incy,
              c, s);
}

Stream &Stream::ThenBlasRot(uint64_t elem_count, DeviceMemory<double> *x,
                            int incx, DeviceMemory<double> *y, int incy,
                            double c, double s) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(c), PARAM(s));

  ThenBlasImpl<uint64_t, DeviceMemory<double> *, int, DeviceMemory<double> *,
               int, double, double>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRot, elem_count, x, incx, y, incy,
              c, s);
}

Stream &Stream::ThenBlasRot(uint64_t elem_count,
                            DeviceMemory<std::complex<float>> *x, int incx,
                            DeviceMemory<std::complex<float>> *y, int incy,
                            float c, float s) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(c), PARAM(s));

  ThenBlasImpl<uint64_t, DeviceMemory<std::complex<float>> *, int,
               DeviceMemory<std::complex<float>> *, int, float, float>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRot, elem_count, x, incx, y, incy,
              c, s);
}

Stream &Stream::ThenBlasRot(uint64_t elem_count,
                            DeviceMemory<std::complex<double>> *x, int incx,
                            DeviceMemory<std::complex<double>> *y, int incy,
                            double c, double s) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(c), PARAM(s));

  ThenBlasImpl<uint64_t, DeviceMemory<std::complex<double>> *, int,
               DeviceMemory<std::complex<double>> *, int, double, double>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRot, elem_count, x, incx, y, incy,
              c, s);
}

Stream &Stream::ThenBlasRotg(DeviceMemory<float> *a, DeviceMemory<float> *b,
                             DeviceMemory<float> *c, DeviceMemory<float> *s) {
  VLOG_CALL(PARAM(a), PARAM(b), PARAM(c), PARAM(s));

  ThenBlasImpl<DeviceMemory<float> *, DeviceMemory<float> *,
               DeviceMemory<float> *, DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotg, a, b, c, s);
}

Stream &Stream::ThenBlasRotg(DeviceMemory<double> *a, DeviceMemory<double> *b,
                             DeviceMemory<double> *c, DeviceMemory<double> *s) {
  VLOG_CALL(PARAM(a), PARAM(b), PARAM(c), PARAM(s));

  ThenBlasImpl<DeviceMemory<double> *, DeviceMemory<double> *,
               DeviceMemory<double> *, DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotg, a, b, c, s);
}

Stream &Stream::ThenBlasRotg(DeviceMemory<std::complex<float>> *a,
                             DeviceMemory<std::complex<float>> *b,
                             DeviceMemory<float> *c,
                             DeviceMemory<std::complex<float>> *s) {
  VLOG_CALL(PARAM(a), PARAM(b), PARAM(c), PARAM(s));

  ThenBlasImpl<DeviceMemory<std::complex<float>> *,
               DeviceMemory<std::complex<float>> *, DeviceMemory<float> *,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotg, a, b, c, s);
}

Stream &Stream::ThenBlasRotg(DeviceMemory<std::complex<double>> *a,
                             DeviceMemory<std::complex<double>> *b,
                             DeviceMemory<double> *c,
                             DeviceMemory<std::complex<double>> *s) {
  VLOG_CALL(PARAM(a), PARAM(b), PARAM(c), PARAM(s));

  ThenBlasImpl<DeviceMemory<std::complex<double>> *,
               DeviceMemory<std::complex<double>> *, DeviceMemory<double> *,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotg, a, b, c, s);
}

Stream &Stream::ThenBlasRotm(uint64_t elem_count, DeviceMemory<float> *x,
                             int incx, DeviceMemory<float> *y, int incy,
                             const DeviceMemory<float> &param) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(param));

  ThenBlasImpl<uint64_t, DeviceMemory<float> *, int, DeviceMemory<float> *, int,
               const DeviceMemory<float> &>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotm, elem_count, x, incx, y,
              incy, param);
}

Stream &Stream::ThenBlasRotm(uint64_t elem_count, DeviceMemory<double> *x,
                             int incx, DeviceMemory<double> *y, int incy,
                             const DeviceMemory<double> &param) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy),
            PARAM(param));

  ThenBlasImpl<uint64_t, DeviceMemory<double> *, int, DeviceMemory<double> *,
               int, const DeviceMemory<double> &>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotm, elem_count, x, incx, y,
              incy, param);
}

Stream &Stream::ThenBlasRotmg(DeviceMemory<float> *d1, DeviceMemory<float> *d2,
                              DeviceMemory<float> *x1,
                              const DeviceMemory<float> &y1,
                              DeviceMemory<float> *param) {
  VLOG_CALL(PARAM(d1), PARAM(d2), PARAM(x1), PARAM(y1), PARAM(param));

  ThenBlasImpl<DeviceMemory<float> *, DeviceMemory<float> *,
               DeviceMemory<float> *, const DeviceMemory<float> &,
               DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotmg, d1, d2, x1, y1, param);
}

Stream &Stream::ThenBlasRotmg(DeviceMemory<double> *d1,
                              DeviceMemory<double> *d2,
                              DeviceMemory<double> *x1,
                              const DeviceMemory<double> &y1,
                              DeviceMemory<double> *param) {
  VLOG_CALL(PARAM(d1), PARAM(d2), PARAM(x1), PARAM(y1), PARAM(param));

  ThenBlasImpl<DeviceMemory<double> *, DeviceMemory<double> *,
               DeviceMemory<double> *, const DeviceMemory<double> &,
               DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasRotmg, d1, d2, x1, y1, param);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, float alpha,
                             DeviceMemory<float> *x, int incx) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, float, DeviceMemory<float> *, int> impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, double alpha,
                             DeviceMemory<double> *x, int incx) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, double, DeviceMemory<double> *, int> impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, float alpha,
                             DeviceMemory<std::complex<float>> *x, int incx) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, float, DeviceMemory<std::complex<float>> *, int> impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, double alpha,
                             DeviceMemory<std::complex<double>> *x, int incx) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, double, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, std::complex<float> alpha,
                             DeviceMemory<std::complex<float>> *x, int incx) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, std::complex<float>,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasScal(uint64_t elem_count, std::complex<double> alpha,
                             DeviceMemory<std::complex<double>> *x, int incx) {
  VLOG_CALL(PARAM(elem_count), PARAM(alpha), PARAM(x), PARAM(incx));

  ThenBlasImpl<uint64_t, std::complex<double>,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasScal, elem_count, alpha, x, incx);
}

Stream &Stream::ThenBlasSwap(uint64_t elem_count, DeviceMemory<float> *x,
                             int incx, DeviceMemory<float> *y, int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, DeviceMemory<float> *, int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSwap, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasSwap(uint64_t elem_count, DeviceMemory<double> *x,
                             int incx, DeviceMemory<double> *y, int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, DeviceMemory<double> *, int, DeviceMemory<double> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSwap, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasSwap(uint64_t elem_count,
                             DeviceMemory<std::complex<float>> *x, int incx,
                             DeviceMemory<std::complex<float>> *y, int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, DeviceMemory<std::complex<float>> *, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSwap, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasSwap(uint64_t elem_count,
                             DeviceMemory<std::complex<double>> *x, int incx,
                             DeviceMemory<std::complex<double>> *y, int incy) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(y), PARAM(incy));

  ThenBlasImpl<uint64_t, DeviceMemory<std::complex<double>> *, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSwap, elem_count, x, incx, y,
              incy);
}

Stream &Stream::ThenBlasIamax(uint64_t elem_count, const DeviceMemory<float> &x,
                              int incx, DeviceMemory<int> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int, DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamax, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamax(uint64_t elem_count,
                              const DeviceMemory<double> &x, int incx,
                              DeviceMemory<int> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int, DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamax, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamax(uint64_t elem_count,
                              const DeviceMemory<std::complex<float>> &x,
                              int incx, DeviceMemory<int> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamax, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamax(uint64_t elem_count,
                              const DeviceMemory<std::complex<double>> &x,
                              int incx, DeviceMemory<int> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamax, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamin(uint64_t elem_count, const DeviceMemory<float> &x,
                              int incx, DeviceMemory<int> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<float> &, int, DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamin, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamin(uint64_t elem_count,
                              const DeviceMemory<double> &x, int incx,
                              DeviceMemory<int> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<double> &, int, DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamin, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamin(uint64_t elem_count,
                              const DeviceMemory<std::complex<float>> &x,
                              int incx, DeviceMemory<int> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamin, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasIamin(uint64_t elem_count,
                              const DeviceMemory<std::complex<double>> &x,
                              int incx, DeviceMemory<int> *result) {
  VLOG_CALL(PARAM(elem_count), PARAM(x), PARAM(incx), PARAM(result));

  ThenBlasImpl<uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<int> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasIamin, elem_count, x, incx,
              result);
}

Stream &Stream::ThenBlasGbmv(blas::Transpose trans, uint64_t m, uint64 n,
                             uint64_t kl, uint64 ku, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(kl), PARAM(ku),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x), PARAM(incx),
            PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, uint64, uint64, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGbmv, trans, m, n, kl, ku, alpha,
              a, lda, x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGbmv(blas::Transpose trans, uint64_t m, uint64 n,
                             uint64_t kl, uint64 ku, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(kl), PARAM(ku),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x), PARAM(incx),
            PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, uint64, uint64, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGbmv, trans, m, n, kl, ku, alpha,
              a, lda, x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGbmv(blas::Transpose trans, uint64_t m, uint64 n,
                             uint64_t kl, uint64 ku, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(kl), PARAM(ku),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x), PARAM(incx),
            PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, uint64, uint64,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGbmv, trans, m, n, kl, ku, alpha,
              a, lda, x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGbmv(blas::Transpose trans, uint64_t m, uint64 n,
                             uint64_t kl, uint64 ku, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(kl), PARAM(ku),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x), PARAM(incx),
            PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, uint64, uint64,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGbmv, trans, m, n, kl, ku, alpha,
              a, lda, x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                             float alpha, const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemv, trans, m, n, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                             double alpha, const DeviceMemory<double> &a,
                             int lda, const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemv, trans, m, n, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemv, trans, m, n, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasImpl<blas::Transpose, uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemv, trans, m, n, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasGer(uint64_t m, uint64 n, float alpha,
                            const DeviceMemory<float> &x, int incx,
                            const DeviceMemory<float> &y, int incy,
                            DeviceMemory<float> *a, int lda) {
  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, float, const DeviceMemory<float> &, int,
               const DeviceMemory<float> &, int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGer, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGer(uint64_t m, uint64 n, double alpha,
                            const DeviceMemory<double> &x, int incx,
                            const DeviceMemory<double> &y, int incy,
                            DeviceMemory<double> *a, int lda) {
  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, double, const DeviceMemory<double> &, int,
               const DeviceMemory<double> &, int, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGer, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGerc(uint64_t m, uint64 n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy, DeviceMemory<std::complex<float>> *a,
                             int lda) {
  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGerc, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGerc(uint64_t m, uint64 n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy, DeviceMemory<std::complex<double>> *a,
                             int lda) {
  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGerc, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGeru(uint64_t m, uint64 n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy, DeviceMemory<std::complex<float>> *a,
                             int lda) {
  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGeru, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasGeru(uint64_t m, uint64 n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy, DeviceMemory<std::complex<double>> *a,
                             int lda) {
  VLOG_CALL(PARAM(m), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx), PARAM(y),
            PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGeru, m, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasHbmv(blas::UpperLower uplo, uint64_t n, uint64 k,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(k), PARAM(alpha), PARAM(a), PARAM(lda),
            PARAM(x), PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHbmv, uplo, n, k, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasHbmv(blas::UpperLower uplo, uint64_t n, uint64 k,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(k), PARAM(alpha), PARAM(a), PARAM(lda),
            PARAM(x), PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHbmv, uplo, n, k, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasHemv(blas::UpperLower uplo, uint64_t n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHemv, uplo, n, alpha, a, lda, x,
              incx, beta, y, incy);
}

Stream &Stream::ThenBlasHemv(blas::UpperLower uplo, uint64_t n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHemv, uplo, n, alpha, a, lda, x,
              incx, beta, y, incy);
}

Stream &Stream::ThenBlasHer(blas::UpperLower uplo, uint64_t n, float alpha,
                            const DeviceMemory<std::complex<float>> &x,
                            int incx, DeviceMemory<std::complex<float>> *a,
                            int lda) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, float,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer, uplo, n, alpha, x, incx, a,
              lda);
}

Stream &Stream::ThenBlasHer(blas::UpperLower uplo, uint64_t n, double alpha,
                            const DeviceMemory<std::complex<double>> &x,
                            int incx, DeviceMemory<std::complex<double>> *a,
                            int lda) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, double,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer, uplo, n, alpha, x, incx, a,
              lda);
}

Stream &Stream::ThenBlasHer2(blas::UpperLower uplo, uint64_t n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy, DeviceMemory<std::complex<float>> *a,
                             int lda) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer2, uplo, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasHer2(blas::UpperLower uplo, uint64_t n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy, DeviceMemory<std::complex<double>> *a,
                             int lda) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer2, uplo, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasHpmv(blas::UpperLower uplo, uint64_t n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &ap,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(ap), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &,
               const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpmv, uplo, n, alpha, ap, x, incx,
              beta, y, incy);
}

Stream &Stream::ThenBlasHpmv(blas::UpperLower uplo, uint64_t n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &ap,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(ap), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &,
               const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpmv, uplo, n, alpha, ap, x, incx,
              beta, y, incy);
}

Stream &Stream::ThenBlasHpr(blas::UpperLower uplo, uint64_t n, float alpha,
                            const DeviceMemory<std::complex<float>> &x,
                            int incx, DeviceMemory<std::complex<float>> *ap) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, float,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpr, uplo, n, alpha, x, incx, ap);
}

Stream &Stream::ThenBlasHpr(blas::UpperLower uplo, uint64_t n, double alpha,
                            const DeviceMemory<std::complex<double>> &x,
                            int incx, DeviceMemory<std::complex<double>> *ap) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, double,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpr, uplo, n, alpha, x, incx, ap);
}

Stream &Stream::ThenBlasHpr2(blas::UpperLower uplo, uint64_t n,
                             std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &x,
                             int incx,
                             const DeviceMemory<std::complex<float>> &y,
                             int incy, DeviceMemory<std::complex<float>> *ap) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpr2, uplo, n, alpha, x, incx, y,
              incy, ap);
}

Stream &Stream::ThenBlasHpr2(blas::UpperLower uplo, uint64_t n,
                             std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &x,
                             int incx,
                             const DeviceMemory<std::complex<double>> &y,
                             int incy, DeviceMemory<std::complex<double>> *ap) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHpr2, uplo, n, alpha, x, incx, y,
              incy, ap);
}

Stream &Stream::ThenBlasSbmv(blas::UpperLower uplo, uint64_t n, uint64 k,
                             float alpha, const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(k), PARAM(alpha), PARAM(a), PARAM(lda),
            PARAM(x), PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSbmv, uplo, n, k, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasSbmv(blas::UpperLower uplo, uint64_t n, uint64 k,
                             double alpha, const DeviceMemory<double> &a,
                             int lda, const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(k), PARAM(alpha), PARAM(a), PARAM(lda),
            PARAM(x), PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSbmv, uplo, n, k, alpha, a, lda,
              x, incx, beta, y, incy);
}

Stream &Stream::ThenBlasSpmv(blas::UpperLower uplo, uint64_t n, float alpha,
                             const DeviceMemory<float> &ap,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(ap), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               const DeviceMemory<float> &, int, float, DeviceMemory<float> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpmv, uplo, n, alpha, ap, x, incx,
              beta, y, incy);
}

Stream &Stream::ThenBlasSpmv(blas::UpperLower uplo, uint64_t n, double alpha,
                             const DeviceMemory<double> &ap,
                             const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(ap), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               const DeviceMemory<double> &, int, double,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpmv, uplo, n, alpha, ap, x, incx,
              beta, y, incy);
}

Stream &Stream::ThenBlasSpr(blas::UpperLower uplo, uint64_t n, float alpha,
                            const DeviceMemory<float> &x, int incx,
                            DeviceMemory<float> *ap) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpr, uplo, n, alpha, x, incx, ap);
}

Stream &Stream::ThenBlasSpr(blas::UpperLower uplo, uint64_t n, double alpha,
                            const DeviceMemory<double> &x, int incx,
                            DeviceMemory<double> *ap) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpr, uplo, n, alpha, x, incx, ap);
}

Stream &Stream::ThenBlasSpr2(blas::UpperLower uplo, uint64_t n, float alpha,
                             const DeviceMemory<float> &x, int incx,
                             const DeviceMemory<float> &y, int incy,
                             DeviceMemory<float> *ap) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, const DeviceMemory<float> &, int, DeviceMemory<float> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpr2, uplo, n, alpha, x, incx, y,
              incy, ap);
}

Stream &Stream::ThenBlasSpr2(blas::UpperLower uplo, uint64_t n, double alpha,
                             const DeviceMemory<double> &x, int incx,
                             const DeviceMemory<double> &y, int incy,
                             DeviceMemory<double> *ap) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(ap));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, const DeviceMemory<double> &, int, DeviceMemory<double> *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSpr2, uplo, n, alpha, x, incx, y,
              incy, ap);
}

Stream &Stream::ThenBlasSymv(blas::UpperLower uplo, uint64_t n, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &x, int incx, float beta,
                             DeviceMemory<float> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, const DeviceMemory<float> &, int, float,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymv, uplo, n, alpha, a, lda, x,
              incx, beta, y, incy);
}

Stream &Stream::ThenBlasSymv(blas::UpperLower uplo, uint64_t n, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             const DeviceMemory<double> &x, int incx,
                             double beta, DeviceMemory<double> *y, int incy) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(x),
            PARAM(incx), PARAM(beta), PARAM(y), PARAM(incy));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, const DeviceMemory<double> &, int, double,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymv, uplo, n, alpha, a, lda, x,
              incx, beta, y, incy);
}

Stream &Stream::ThenBlasSyr(blas::UpperLower uplo, uint64_t n, float alpha,
                            const DeviceMemory<float> &x, int incx,
                            DeviceMemory<float> *a, int lda) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr, uplo, n, alpha, x, incx, a,
              lda);
}

Stream &Stream::ThenBlasSyr(blas::UpperLower uplo, uint64_t n, double alpha,
                            const DeviceMemory<double> &x, int incx,
                            DeviceMemory<double> *a, int lda) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr, uplo, n, alpha, x, incx, a,
              lda);
}

Stream &Stream::ThenBlasSyr2(blas::UpperLower uplo, uint64_t n, float alpha,
                             const DeviceMemory<float> &x, int incx,
                             const DeviceMemory<float> &y, int incy,
                             DeviceMemory<float> *a, int lda) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, float, const DeviceMemory<float> &,
               int, const DeviceMemory<float> &, int, DeviceMemory<float> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2, uplo, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasSyr2(blas::UpperLower uplo, uint64_t n, double alpha,
                             const DeviceMemory<double> &x, int incx,
                             const DeviceMemory<double> &y, int incy,
                             DeviceMemory<double> *a, int lda) {
  VLOG_CALL(PARAM(uplo), PARAM(n), PARAM(alpha), PARAM(x), PARAM(incx),
            PARAM(y), PARAM(incy), PARAM(a), PARAM(lda));

  ThenBlasImpl<blas::UpperLower, uint64_t, double, const DeviceMemory<double> &,
               int, const DeviceMemory<double> &, int, DeviceMemory<double> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2, uplo, n, alpha, x, incx, y,
              incy, a, lda);
}

Stream &Stream::ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbmv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbmv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *x,
                             int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbmv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *x,
                             int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbmv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbsv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbsv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *x,
                             int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbsv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n, uint64 k,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *x,
                             int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(k),
            PARAM(a), PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               uint64_t, const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTbsv, uplo, trans, diag, n, k, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<float> &ap,
                             DeviceMemory<float> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<float> &, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpmv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<double> &ap,
                             DeviceMemory<double> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<double> &, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpmv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<float>> &ap,
                             DeviceMemory<std::complex<float>> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<float>> &,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpmv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<double>> &ap,
                             DeviceMemory<std::complex<double>> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<double>> &,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpmv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<float> &ap,
                             DeviceMemory<float> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<float> &, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpsv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<double> &ap,
                             DeviceMemory<double> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<double> &, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpsv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<float>> &ap,
                             DeviceMemory<std::complex<float>> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<float>> &,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpsv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<double>> &ap,
                             DeviceMemory<std::complex<double>> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(ap),
            PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<double>> &,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTpsv, uplo, trans, diag, n, ap, x,
              incx);
}

Stream &Stream::ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<float> &, int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<double> &, int, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *x,
                             int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *x,
                             int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<float> &, int, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *x, int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<double> &, int, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *x,
                             int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsv, uplo, trans, diag, n, a,
              lda, x, incx);
}

Stream &Stream::ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                             blas::Diagonal diag, uint64_t n,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *x,
                             int incx) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(diag), PARAM(n), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, blas::Diagonal, uint64_t,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsv, uplo, trans, diag, n, a,
              lda, x, incx);
}

namespace {
// Like ThenBlasImpl, except this expects the last argument of blas_func to be a
// blas::ProfileResult*.  This functor doesn't put the stream into an error
// state if the op fails and the profile result is non-null.  Instead, the
// error-ness is returned in the profile result itself.
template <typename... Args>
struct ThenBlasWithProfileImpl {
  Stream &operator()(Stream *stream,
                     bool (blas::BlasSupport::*blas_func)(
                         Stream *, Args..., blas::ProfileResult *),
                     Args... args, blas::ProfileResult *profile_result) {
    ThenBlasImpl<Args..., blas::ProfileResult *> Runner;
    bool record_error = profile_result == nullptr;
    return Runner.Run(stream, blas_func, record_error, args..., profile_result);
  }
};
}  // anonymous namespace

Stream &Stream::ThenBlasGemvWithProfiling(
    blas::Transpose trans, uint64_t m, uint64 n, float alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &x,
    int incx, float beta, DeviceMemory<float> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasWithProfileImpl<
      blas::Transpose, uint64_t, uint64_t, float, const DeviceMemory<float> &,
      int, const DeviceMemory<float> &, int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemvWithProfiling, trans, m, n,
              alpha, a, lda, x, incx, beta, y, incy, output_profile_result);
}

Stream &Stream::ThenBlasGemvWithProfiling(
    blas::Transpose trans, uint64_t m, uint64 n, double alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &x,
    int incx, double beta, DeviceMemory<double> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasWithProfileImpl<blas::Transpose, uint64_t, uint64_t, double,
                          const DeviceMemory<double> &, int,
                          const DeviceMemory<double> &, int, double,
                          DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemvWithProfiling, trans, m, n,
              alpha, a, lda, x, incx, beta, y, incy, output_profile_result);
}

Stream &Stream::ThenBlasGemvWithProfiling(
    blas::Transpose trans, uint64_t m, uint64 n, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &x, int incx,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasWithProfileImpl<
      blas::Transpose, uint64_t, uint64_t, std::complex<float>,
      const DeviceMemory<std::complex<float>> &, int,
      const DeviceMemory<std::complex<float>> &, int, std::complex<float>,
      DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemvWithProfiling, trans, m, n,
              alpha, a, lda, x, incx, beta, y, incy, output_profile_result);
}

Stream &Stream::ThenBlasGemvWithProfiling(
    blas::Transpose trans, uint64_t m, uint64 n, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &x, int incx,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(trans), PARAM(m), PARAM(n), PARAM(alpha), PARAM(a),
            PARAM(lda), PARAM(x), PARAM(incx), PARAM(beta), PARAM(y),
            PARAM(incy));

  ThenBlasWithProfileImpl<
      blas::Transpose, uint64_t, uint64_t, std::complex<double>,
      const DeviceMemory<std::complex<double>> &, int,
      const DeviceMemory<std::complex<double>> &, int, std::complex<double>,
      DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemvWithProfiling, trans, m, n,
              alpha, a, lda, x, incx, beta, y, incy, output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t,
                          uint64_t, float, const DeviceMemory<Eigen::half> &,
                          int, const DeviceMemory<Eigen::half> &, int, float,
                          DeviceMemory<Eigen::half> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t,
                          uint64_t, float, const DeviceMemory<float> &, int,
                          const DeviceMemory<float> &, int, float,
                          DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t,
                          uint64_t, double, const DeviceMemory<double> &, int,
                          const DeviceMemory<double> &, int, double,
                          DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<
      blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
      std::complex<float>, const DeviceMemory<std::complex<float>> &, int,
      const DeviceMemory<std::complex<float>> &, int, std::complex<float>,
      DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasGemmWithProfiling(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasWithProfileImpl<
      blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
      std::complex<double>, const DeviceMemory<std::complex<double>> &, int,
      const DeviceMemory<std::complex<double>> &, int, std::complex<double>,
      DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmWithProfiling, transa, transb,
              m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
              output_profile_result);
}

Stream &Stream::ThenBlasHemm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &b,
                             int ldb, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *c, int ldc) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHemm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasHemm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &b,
                             int ldb, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *c, int ldc) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHemm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasHerk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, float alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, float beta,
                             DeviceMemory<std::complex<float>> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, float,
               const DeviceMemory<std::complex<float>> &, int, float,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHerk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasHerk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, double alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, double beta,
                             DeviceMemory<std::complex<double>> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, double,
               const DeviceMemory<std::complex<double>> &, int, double,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHerk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasHer2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, std::complex<float> alpha,
                              const DeviceMemory<std::complex<float>> &a,
                              int lda,
                              const DeviceMemory<std::complex<float>> &b,
                              int ldb, float beta,
                              DeviceMemory<std::complex<float>> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int, float,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasHer2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, std::complex<double> alpha,
                              const DeviceMemory<std::complex<double>> &a,
                              int lda,
                              const DeviceMemory<std::complex<double>> &b,
                              int ldb, double beta,
                              DeviceMemory<std::complex<double>> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int, double,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasHer2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             const DeviceMemory<float> &b, int ldb, float beta,
                             DeviceMemory<float> *c, int ldc) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             const DeviceMemory<double> &b, int ldb,
                             double beta, DeviceMemory<double> *c, int ldc) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda,
                             const DeviceMemory<std::complex<float>> &b,
                             int ldb, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *c, int ldc) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64_t m,
                             uint64_t n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda,
                             const DeviceMemory<std::complex<double>> &b,
                             int ldb, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *c, int ldc) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(m), PARAM(n), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::Side, blas::UpperLower, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSymm, side, uplo, m, n, alpha, a,
              lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, float alpha,
                             const DeviceMemory<float> &a, int lda, float beta,
                             DeviceMemory<float> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, float, DeviceMemory<float> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyrk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             double beta, DeviceMemory<double> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, double,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyrk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, std::complex<float> beta,
                             DeviceMemory<std::complex<float>> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, std::complex<float>, DeviceMemory<std::complex<float>> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyrk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans,
                             uint64_t n, uint64 k, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, std::complex<double> beta,
                             DeviceMemory<std::complex<double>> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(beta), PARAM(c), PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, std::complex<double>, DeviceMemory<std::complex<double>> *,
               int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyrk, uplo, trans, n, k, alpha, a,
              lda, beta, c, ldc);
}

Stream &Stream::ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, float alpha,
                              const DeviceMemory<float> &a, int lda,
                              const DeviceMemory<float> &b, int ldb, float beta,
                              DeviceMemory<float> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, float,
               const DeviceMemory<float> &, int, const DeviceMemory<float> &,
               int, float, DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, double alpha,
                              const DeviceMemory<double> &a, int lda,
                              const DeviceMemory<double> &b, int ldb,
                              double beta, DeviceMemory<double> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t, double,
               const DeviceMemory<double> &, int, const DeviceMemory<double> &,
               int, double, DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, std::complex<float> alpha,
                              const DeviceMemory<std::complex<float>> &a,
                              int lda,
                              const DeviceMemory<std::complex<float>> &b,
                              int ldb, std::complex<float> beta,
                              DeviceMemory<std::complex<float>> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<float>, const DeviceMemory<std::complex<float>> &,
               int, const DeviceMemory<std::complex<float>> &, int,
               std::complex<float>, DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans,
                              uint64_t n, uint64 k, std::complex<double> alpha,
                              const DeviceMemory<std::complex<double>> &a,
                              int lda,
                              const DeviceMemory<std::complex<double>> &b,
                              int ldb, std::complex<double> beta,
                              DeviceMemory<std::complex<double>> *c, int ldc) {
  VLOG_CALL(PARAM(uplo), PARAM(trans), PARAM(n), PARAM(k), PARAM(alpha),
            PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb), PARAM(beta), PARAM(c),
            PARAM(ldc));

  ThenBlasImpl<blas::UpperLower, blas::Transpose, uint64_t, uint64_t,
               std::complex<double>, const DeviceMemory<std::complex<double>> &,
               int, const DeviceMemory<std::complex<double>> &, int,
               std::complex<double>, DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasSyr2k, uplo, trans, n, k, alpha,
              a, lda, b, ldb, beta, c, ldc);
}

Stream &Stream::ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *b, int ldb) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, float, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *b, int ldb) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, double, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *b,
                             int ldb) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *b,
                             int ldb) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrmm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, float alpha,
                             const DeviceMemory<float> &a, int lda,
                             DeviceMemory<float> *b, int ldb) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, float, const DeviceMemory<float> &, int,
               DeviceMemory<float> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, double alpha,
                             const DeviceMemory<double> &a, int lda,
                             DeviceMemory<double> *b, int ldb) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, double, const DeviceMemory<double> &, int,
               DeviceMemory<double> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, std::complex<float> alpha,
                             const DeviceMemory<std::complex<float>> &a,
                             int lda, DeviceMemory<std::complex<float>> *b,
                             int ldb) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float>> &, int,
               DeviceMemory<std::complex<float>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                             blas::Transpose transa, blas::Diagonal diag,
                             uint64_t m, uint64 n, std::complex<double> alpha,
                             const DeviceMemory<std::complex<double>> &a,
                             int lda, DeviceMemory<std::complex<double>> *b,
                             int ldb) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double>> &, int,
               DeviceMemory<std::complex<double>> *, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsm, side, uplo, transa, diag, m,
              n, alpha, a, lda, b, ldb);
}

Stream &Stream::ThenBlasTrsmBatched(blas::Side side, blas::UpperLower uplo,
                                    blas::Transpose transa, blas::Diagonal diag,
                                    uint64_t m, uint64 n, float alpha,
                                    const DeviceMemory<float *> &as, int lda,
                                    DeviceMemory<float *> *bs, int ldb,
                                    int batch_count) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(as), PARAM(lda), PARAM(bs),
            PARAM(ldb), PARAM(batch_count));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, float, const DeviceMemory<float *> &, int,
               DeviceMemory<float *> *, int, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsmBatched, side, uplo, transa,
              diag, m, n, alpha, as, lda, bs, ldb, batch_count);
}

Stream &Stream::ThenBlasTrsmBatched(blas::Side side, blas::UpperLower uplo,
                                    blas::Transpose transa, blas::Diagonal diag,
                                    uint64_t m, uint64 n, double alpha,
                                    const DeviceMemory<double *> &as, int lda,
                                    DeviceMemory<double *> *bs, int ldb,
                                    int batch_count) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(as), PARAM(lda), PARAM(bs),
            PARAM(ldb), PARAM(batch_count));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, double, const DeviceMemory<double *> &, int,
               DeviceMemory<double *> *, int, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsmBatched, side, uplo, transa,
              diag, m, n, alpha, as, lda, bs, ldb, batch_count);
}

Stream &Stream::ThenBlasTrsmBatched(
    blas::Side side, blas::UpperLower uplo, blas::Transpose transa,
    blas::Diagonal diag, uint64_t m, uint64 n, std::complex<float> alpha,
    const DeviceMemory<std::complex<float> *> &as, int lda,
    DeviceMemory<std::complex<float> *> *bs, int ldb, int batch_count) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(as), PARAM(lda), PARAM(bs),
            PARAM(ldb), PARAM(batch_count));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<float>,
               const DeviceMemory<std::complex<float> *> &, int,
               DeviceMemory<std::complex<float> *> *, int, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsmBatched, side, uplo, transa,
              diag, m, n, alpha, as, lda, bs, ldb, batch_count);
}

Stream &Stream::ThenBlasTrsmBatched(
    blas::Side side, blas::UpperLower uplo, blas::Transpose transa,
    blas::Diagonal diag, uint64_t m, uint64 n, std::complex<double> alpha,
    const DeviceMemory<std::complex<double> *> &as, int lda,
    DeviceMemory<std::complex<double> *> *bs, int ldb, int batch_count) {
  VLOG_CALL(PARAM(side), PARAM(uplo), PARAM(transa), PARAM(diag), PARAM(m),
            PARAM(n), PARAM(alpha), PARAM(as), PARAM(lda), PARAM(bs),
            PARAM(ldb), PARAM(batch_count));

  ThenBlasImpl<blas::Side, blas::UpperLower, blas::Transpose, blas::Diagonal,
               uint64_t, uint64_t, std::complex<double>,
               const DeviceMemory<std::complex<double> *> &, int,
               DeviceMemory<std::complex<double> *> *, int, int>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasTrsmBatched, side, uplo, transa,
              diag, m, n, alpha, as, lda, bs, ldb, batch_count);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &b, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &c, int ldc,
    int batch_count) {
  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &b, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               float, const port::ArraySlice<DeviceMemory<Eigen::half> *> &,
               int, const port::ArraySlice<DeviceMemory<Eigen::half> *> &, int,
               float, const port::ArraySlice<DeviceMemory<Eigen::half> *> &,
               int, int, ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, const port::ArraySlice<DeviceMemory<float> *> &a,
    int lda, const port::ArraySlice<DeviceMemory<float> *> &b, int ldb,
    float beta, const port::ArraySlice<DeviceMemory<float> *> &c, int ldc,
    int batch_count) {
  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, float alpha, const port::ArraySlice<DeviceMemory<float> *> &a,
    int lda, const port::ArraySlice<DeviceMemory<float> *> &b, int ldb,
    float beta, const port::ArraySlice<DeviceMemory<float> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               float, const port::ArraySlice<DeviceMemory<float> *> &, int,
               const port::ArraySlice<DeviceMemory<float> *> &, int, float,
               const port::ArraySlice<DeviceMemory<float> *> &, int, int,
               ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, double alpha, const port::ArraySlice<DeviceMemory<double> *> &a,
    int lda, const port::ArraySlice<DeviceMemory<double> *> &b, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c, int ldc,
    int batch_count) {
  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, double alpha, const port::ArraySlice<DeviceMemory<double> *> &a,
    int lda, const port::ArraySlice<DeviceMemory<double> *> &b, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               double, const port::ArraySlice<DeviceMemory<double> *> &, int,
               const port::ArraySlice<DeviceMemory<double> *> &, int, double,
               const port::ArraySlice<DeviceMemory<double> *> &, int, int,
               ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b, int ldb,
    std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c, int ldc,
    int batch_count) {
  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b, int ldb,
    std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               std::complex<float>,
               const port::ArraySlice<DeviceMemory<std::complex<float>> *> &,
               int,
               const port::ArraySlice<DeviceMemory<std::complex<float>> *> &,
               int, std::complex<float>,
               const port::ArraySlice<DeviceMemory<std::complex<float>> *> &,
               int, int, ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

Stream &Stream::ThenBlasGemmBatched(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b, int ldb,
    std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c, int ldc,
    int batch_count) {
  return ThenBlasGemmBatchedWithScratch(transa, transb, m, n, k, alpha, a, lda,
                                        b, ldb, beta, c, ldc, batch_count,
                                        /*scratch_allocator=*/nullptr);
}

Stream &Stream::ThenBlasGemmBatchedWithScratch(
    blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
    uint64_t k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b, int ldb,
    std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  VLOG_CALL(PARAM(transa), PARAM(transb), PARAM(m), PARAM(n), PARAM(k),
            PARAM(alpha), PARAM(a), PARAM(lda), PARAM(b), PARAM(ldb),
            PARAM(beta), PARAM(c), PARAM(ldc), PARAM(batch_count));

  ThenBlasImpl<blas::Transpose, blas::Transpose, uint64_t, uint64_t, uint64,
               std::complex<double>,
               const port::ArraySlice<DeviceMemory<std::complex<double>> *> &,
               int,
               const port::ArraySlice<DeviceMemory<std::complex<double>> *> &,
               int, std::complex<double>,
               const port::ArraySlice<DeviceMemory<std::complex<double>> *> &,
               int, int, ScratchAllocator *>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasGemmBatched, transa, transb, m, n,
              k, alpha, a, lda, b, ldb, beta, c, ldc, batch_count,
              scratch_allocator);
}

template <typename ABType, typename CType>
Stream &Stream::ThenBlasLtMatmulImpl(
    const blas::IBlasLtMatmulPlan *plan, const HostOrDeviceScalar<CType> &alpha,
    const DeviceMemory<ABType> &a, const DeviceMemory<ABType> &b,
    const HostOrDeviceScalar<CType> &beta, DeviceMemory<CType> *c,
    ScratchAllocator *scratch_allocator,
    const blas::IBlasLtMatmulAlgorithm *algorithm,
    const DeviceMemory<CType> &bias,
    blas::ProfileResult *output_profile_result) {
  VLOG_CALL(PARAM(plan), PARAM(alpha), PARAM(a), PARAM(b), PARAM(beta),
            PARAM(scratch_allocator), PARAM(c), PARAM(algorithm), PARAM(bias));

  ThenBlasWithProfileImpl<
      const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<CType> &,
      const DeviceMemory<ABType> &, const DeviceMemory<ABType> &,
      const HostOrDeviceScalar<CType> &, DeviceMemory<CType> *,
      ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
      const DeviceMemory<CType> &>
      impl;
  return impl(this, &blas::BlasSupport::DoBlasLtMatmul, plan, alpha, a, b, beta,
              c, scratch_allocator, algorithm, bias, output_profile_result);
}

// Explicit template instantiations for each supported type combination.
template Stream &Stream::ThenBlasLtMatmulImpl<int8, int32>(
    const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<int32> &,
    const DeviceMemory<int8> &, const DeviceMemory<int8> &,
    const HostOrDeviceScalar<int32> &, DeviceMemory<int32> *,
    ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<int32> &, blas::ProfileResult *);

template Stream &Stream::ThenBlasLtMatmulImpl<Eigen::half, Eigen::half>(
    const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<Eigen::half> &,
    const DeviceMemory<Eigen::half> &, const DeviceMemory<Eigen::half> &,
    const HostOrDeviceScalar<Eigen::half> &, DeviceMemory<Eigen::half> *,
    ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<Eigen::half> &, blas::ProfileResult *);

template Stream &Stream::ThenBlasLtMatmulImpl<float, float>(
    const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<float> &,
    const DeviceMemory<float> &, const DeviceMemory<float> &,
    const HostOrDeviceScalar<float> &, DeviceMemory<float> *,
    ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<float> &, blas::ProfileResult *);

template Stream &Stream::ThenBlasLtMatmulImpl<double, double>(
    const blas::IBlasLtMatmulPlan *, const HostOrDeviceScalar<double> &,
    const DeviceMemory<double> &, const DeviceMemory<double> &,
    const HostOrDeviceScalar<double> &, DeviceMemory<double> *,
    ScratchAllocator *, const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<double> &, blas::ProfileResult *);

template Stream &
Stream::ThenBlasLtMatmulImpl<std::complex<float>, std::complex<float>>(
    const blas::IBlasLtMatmulPlan *,
    const HostOrDeviceScalar<std::complex<float>> &,
    const DeviceMemory<std::complex<float>> &,
    const DeviceMemory<std::complex<float>> &,
    const HostOrDeviceScalar<std::complex<float>> &,
    DeviceMemory<std::complex<float>> *, ScratchAllocator *,
    const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<std::complex<float>> &, blas::ProfileResult *);

template Stream &
Stream::ThenBlasLtMatmulImpl<std::complex<double>, std::complex<double>>(
    const blas::IBlasLtMatmulPlan *,
    const HostOrDeviceScalar<std::complex<double>> &,
    const DeviceMemory<std::complex<double>> &,
    const DeviceMemory<std::complex<double>> &,
    const HostOrDeviceScalar<std::complex<double>> &,
    DeviceMemory<std::complex<double>> *, ScratchAllocator *,
    const blas::IBlasLtMatmulAlgorithm *,
    const DeviceMemory<std::complex<double>> &, blas::ProfileResult *);

Stream &Stream::ThenSetRngSeed(const uint8 *seed, uint64_t seed_bytes) {
  VLOG_CALL(PARAM(seed), PARAM(seed_bytes));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->SetSeed(this, seed, seed_bytes));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers() << " unable to initialize RNG";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandUniform(DeviceMemory<float> *values) {
  VLOG_CALL(PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandUniform(this, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandGaussian(float mean, float sd,
                                         DeviceMemory<float> *values) {
  VLOG_CALL(PARAM(mean), PARAM(sd), PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandGaussian(this, mean, sd, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandGaussian(double mean, double sd,
                                         DeviceMemory<double> *values) {
  VLOG_CALL(PARAM(mean), PARAM(sd), PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandGaussian(this, mean, sd, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandUniform(DeviceMemory<double> *values) {
  VLOG_CALL(PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandUniform(this, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandUniform(
    DeviceMemory<std::complex<float>> *values) {
  VLOG_CALL(PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandUniform(this, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenPopulateRandUniform(
    DeviceMemory<std::complex<double>> *values) {
  VLOG_CALL(PARAM(values));

  if (rng::RngSupport *rng = parent_->AsRng()) {
    CheckError(rng->DoPopulateRandUniform(this, values));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform RNG operation using StreamExecutor"
                 " without RNG support.";
  }
  return *this;
}

Stream &Stream::ThenMemcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                           uint64_t size) {
  VLOG_CALL(PARAM(host_dst), PARAM(gpu_src), PARAM(size));

  CheckError(parent_->Memcpy(this, host_dst, gpu_src, size));
  return *this;
}

Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                           uint64_t size) {
  VLOG_CALL(PARAM(gpu_dst), PARAM(host_src), PARAM(size));

  CheckError(parent_->Memcpy(this, gpu_dst, host_src, size));
  return *this;
}

Stream &Stream::ThenMemcpy(DeviceMemoryBase *gpu_dst,
                           const DeviceMemoryBase &gpu_src, uint64_t size) {
  VLOG_CALL(PARAM(gpu_dst), PARAM(gpu_src), PARAM(size));

  CheckError(parent_->MemcpyDeviceToDevice(this, gpu_dst, gpu_src, size));
  return *this;
}

Stream &Stream::ThenMemZero(DeviceMemoryBase *location, uint64_t size) {
  VLOG_CALL(PARAM(location), PARAM(size));

  CheckStatus(parent_->MemZero(this, location, size));
  return *this;
}

Stream &Stream::ThenMemset32(DeviceMemoryBase *location, uint32 pattern,
                             uint64_t size) {
  VLOG_CALL(PARAM(location), PARAM(pattern), PARAM(size));

  CheckStatus(parent_->Memset32(this, location, pattern, size));
  return *this;
}

Stream &Stream::ThenRnnForward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<Eigen::half> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<Eigen::half> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<Eigen::half> &input_c_data,
    const DeviceMemory<Eigen::half> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    DeviceMemory<Eigen::half> *output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    DeviceMemory<Eigen::half> *output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    DeviceMemory<Eigen::half> *output_c_data, bool is_training,
    ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnForward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        is_training, reserve_space_allocator, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenRnnForward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<float> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<float> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<float> &input_c_data, const DeviceMemory<float> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    DeviceMemory<float> *output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    DeviceMemory<float> *output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    DeviceMemory<float> *output_c_data, bool is_training,
    ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnForward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        is_training, reserve_space_allocator, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenRnnForward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<double> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<double> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<double> &input_c_data,
    const DeviceMemory<double> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    DeviceMemory<double> *output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    DeviceMemory<double> *output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    DeviceMemory<double> *output_c_data, bool is_training,
    ScratchAllocator *reserve_space_allocator,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnForward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        is_training, reserve_space_allocator, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenRnnBackward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<Eigen::half> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<Eigen::half> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<Eigen::half> &input_c_data,
    const DeviceMemory<Eigen::half> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    const DeviceMemory<Eigen::half> &output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    const DeviceMemory<Eigen::half> &output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    const DeviceMemory<Eigen::half> &output_c_data,
    const DeviceMemory<Eigen::half> &output_backprop_data,
    const DeviceMemory<Eigen::half> &output_h_backprop_data,
    const DeviceMemory<Eigen::half> &output_c_backprop_data,
    DeviceMemory<Eigen::half> *input_backprop_data,
    DeviceMemory<Eigen::half> *input_h_backprop_data,
    DeviceMemory<Eigen::half> *input_c_backprop_data,
    DeviceMemory<Eigen::half> *params_backprop_data,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnBackward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        output_backprop_data, output_h_backprop_data, output_c_backprop_data,
        input_backprop_data, input_h_backprop_data, input_c_backprop_data,
        params_backprop_data, reserve_space_data, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetError();
    LOG(WARNING) << "Attempting to call ThenRnnBackward without DNN support";
  }
  return *this;
}

Stream &Stream::ThenRnnBackward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<float> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<float> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<float> &input_c_data, const DeviceMemory<float> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    const DeviceMemory<float> &output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    const DeviceMemory<float> &output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    const DeviceMemory<float> &output_c_data,
    const DeviceMemory<float> &output_backprop_data,
    const DeviceMemory<float> &output_h_backprop_data,
    const DeviceMemory<float> &output_c_backprop_data,
    DeviceMemory<float> *input_backprop_data,
    DeviceMemory<float> *input_h_backprop_data,
    DeviceMemory<float> *input_c_backprop_data,
    DeviceMemory<float> *params_backprop_data,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnBackward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        output_backprop_data, output_h_backprop_data, output_c_backprop_data,
        input_backprop_data, input_h_backprop_data, input_c_backprop_data,
        params_backprop_data, reserve_space_data, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetError();
    LOG(WARNING) << "Attempting to call ThenRnnBackward without DNN support";
  }
  return *this;
}

Stream &Stream::ThenRnnBackward(
    const dnn::RnnDescriptor &rnn_desc,
    const dnn::RnnSequenceTensorDescriptor &input_desc,
    const DeviceMemory<double> &input_data,
    const DeviceMemory<int> &seq_lengths_data,
    const dnn::RnnStateTensorDescriptor &input_h_desc,
    const DeviceMemory<double> &input_h_data,
    const dnn::RnnStateTensorDescriptor &input_c_desc,
    const DeviceMemory<double> &input_c_data,
    const DeviceMemory<double> &params,
    const dnn::RnnSequenceTensorDescriptor &output_desc,
    const DeviceMemory<double> &output_data,
    const dnn::RnnStateTensorDescriptor &output_h_desc,
    const DeviceMemory<double> &output_h_data,
    const dnn::RnnStateTensorDescriptor &output_c_desc,
    const DeviceMemory<double> &output_c_data,
    const DeviceMemory<double> &output_backprop_data,
    const DeviceMemory<double> &output_h_backprop_data,
    const DeviceMemory<double> &output_c_backprop_data,
    DeviceMemory<double> *input_backprop_data,
    DeviceMemory<double> *input_h_backprop_data,
    DeviceMemory<double> *input_c_backprop_data,
    DeviceMemory<double> *params_backprop_data,
    DeviceMemory<uint8> *reserve_space_data,
    ScratchAllocator *workspace_allocator,
    dnn::ProfileResult *output_profile_result) {
  // TODO(zhengxq): add VLOG PARAM calls.
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    auto status = dnn->DoRnnBackward(
        this, rnn_desc, input_desc, input_data, seq_lengths_data, input_h_desc,
        input_h_data, input_c_desc, input_c_data, params, output_desc,
        output_data, output_h_desc, output_h_data, output_c_desc, output_c_data,
        output_backprop_data, output_h_backprop_data, output_c_backprop_data,
        input_backprop_data, input_h_backprop_data, input_c_backprop_data,
        params_backprop_data, reserve_space_data, workspace_allocator,
        output_profile_result);
    if (!status && !output_profile_result) {
      SetError();
    }
  } else {
    SetError();
    LOG(WARNING) << "Attempting to call ThenRnnBackward without DNN support";
  }
  return *this;
}

Stream &Stream::ThenCtcLoss(const dnn::RnnStateTensorDescriptor &probs_desc,
                            const DeviceMemory<float> &probs_data,
                            absl::Span<const int> labels_data,
                            absl::Span<const int> labels_lengths_data,
                            absl::Span<const int> input_lengths_data,
                            DeviceMemory<float> *costs_data,
                            const dnn::RnnStateTensorDescriptor &grads_desc,
                            DeviceMemory<float> *grads_data,
                            ScratchAllocator *workspace_allocator) {
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    DeviceMemory<uint8> scratch_memory;
    int ctc_loss_algo_id;
    auto status =
        dnn->PrepareForCtcLoss(this, probs_desc, probs_data, grads_desc,
                               labels_data, labels_lengths_data,
                               input_lengths_data, workspace_allocator,
                               &scratch_memory, &ctc_loss_algo_id)
            .ok();
    if (status) {
      status = dnn->DoCtcLoss(this, probs_desc, probs_data, labels_data,
                              labels_lengths_data, input_lengths_data,
                              costs_data, grads_desc, grads_data,
                              &scratch_memory, ctc_loss_algo_id);
    }
    if (!status) {
      SetError();
    }
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenTransformTensor(const dnn::BatchDescriptor &input_desc,
                                    dnn::DataType input_type,
                                    const DeviceMemoryBase &input_data,
                                    const dnn::BatchDescriptor &output_desc,
                                    dnn::DataType output_type, float scale,
                                    DeviceMemoryBase *output_data) {
  VLOG_CALL(PARAM(input_desc), PARAM(input_type), PARAM(input_data),
            PARAM(output_desc), PARAM(output_type), PARAM(scale),
            PARAM(output_data));
  if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
    CheckError(dnn->DoTransformTensor(this, input_desc, input_type, input_data,
                                      output_desc, output_type, scale,
                                      output_data));
  } else {
    SetErrorAndLogNoDnnSupport();
  }
  return *this;
}

Stream &Stream::ThenDoHostCallback(std::function<void()> callback) {
  VLOG_CALL(PARAM(callback));

  if (!ok()) {
    LOG(INFO) << DebugStreamPointers()
              << " was in error state before adding host callback";
  }
  CheckError(parent_->HostCallback(this, std::move(callback)));
  return *this;
}

Stream &Stream::ThenDoHostCallbackWithStatus(
    std::function<port::Status()> callback) {
  VLOG_CALL(PARAM(callback));

  if (!ok()) {
    LOG(INFO) << DebugStreamPointers()
              << " was in error state before adding host callback";
  }
  CheckError(parent_->HostCallback(this, std::move(callback)));
  return *this;
}

Stream &Stream::ThenRunAfterNextBlockHostUntilDone(
    std::function<void()> callback) {
  VLOG_CALL(PARAM(callback));

  if (!ok()) {
    LOG(INFO) << DebugStreamPointers()
              << " was in error state before adding callback to be run after "
                 "next block-host-until-done.";
  }
  absl::MutexLock lock(&mu_);
  after_block_host_until_done_callbacks_.push_back(std::move(callback));
  return *this;
}

void Stream::CheckError(bool operation_retcode) {
  if (operation_retcode) {
    return;
  }
  absl::MutexLock lock(&mu_);
  status_ = port::InternalError("Unknown error");
}

Stream &Stream::ThenFft(fft::Plan *plan,
                        const DeviceMemory<std::complex<float>> &input,
                        DeviceMemory<std::complex<float>> *output) {
  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan,
                        const DeviceMemory<std::complex<double>> &input,
                        DeviceMemory<std::complex<double>> *output) {
  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan, const DeviceMemory<float> &input,
                        DeviceMemory<std::complex<float>> *output) {
  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan, const DeviceMemory<double> &input,
                        DeviceMemory<std::complex<double>> *output) {
  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan,
                        const DeviceMemory<std::complex<float>> &input,
                        DeviceMemory<float> *output) {
  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

Stream &Stream::ThenFft(fft::Plan *plan,
                        const DeviceMemory<std::complex<double>> &input,
                        DeviceMemory<double> *output) {
  VLOG_CALL(PARAM(plan), PARAM(input), PARAM(output));

  if (fft::FftSupport *fft = parent_->AsFft()) {
    CheckError(fft->DoFft(this, plan, input, output));
  } else {
    SetError();
    LOG(INFO) << DebugStreamPointers()
              << " attempting to perform FFT operation using StreamExecutor"
                 " without FFT support";
  }
  return *this;
}

// It looks confusing, but all this is doing is inserting a callback at the
// present point in the stream to then enqueue a task on the host executor.
Stream &Stream::ThenEnqueueOnBackgroundThread(
    std::function<void(StreamExecutor *)> task) {
  VLOG_CALL(PARAM(task));

  StreamExecutor *stream_executor = this->parent_;
  std::function<void()> bound_task = std::bind(task, stream_executor);

  return ThenDoHostCallback([stream_executor, bound_task]() {
    stream_executor->EnqueueOnBackgroundThread(bound_task);
  });
}

port::Status Stream::BlockHostUntilDone() {
  VLOG_CALL();

  if (!ok()) {
    absl::MutexLock lock(&mu_);
    LOG(INFO) << status_.ToString();
    port::Status status = port::Status(
        port::error::INTERNAL,
        "stream did not block host until done; was already in an error state");
    LOG(INFO) << DebugStreamPointers() << " " << status;
    return status;
  }

  temporary_memory_manager_.DeallocateFinalizedTemporaries();

  port::Status error = parent_->BlockHostUntilDone(this);
  CheckError(error.ok());

  RunAfterBlockHostUntilDoneCallbacks();
  return error;
}

void Stream::RunAfterBlockHostUntilDoneCallbacks() {
  std::vector<std::function<void()>> callbacks;
  {
    absl::MutexLock lock(&mu_);
    std::swap(callbacks, after_block_host_until_done_callbacks_);
  }
  for (const auto &fn : callbacks) {
    fn();
  }
}

std::string Stream::DebugStreamPointers() const {
  // Relies on the ToVlogString(const void*) overload above.
  return absl::StrCat("[stream=", ToVlogString(this),
                      ",impl=", ToVlogString(implementation_.get()), "]");
}

void Stream::CheckStatus(port::Status status) {
  if (status.ok()) {
    return;
  }
  LOG(ERROR) << status;
  absl::MutexLock lock(&mu_);
  status_ = status;
}

}  // namespace stream_executor
