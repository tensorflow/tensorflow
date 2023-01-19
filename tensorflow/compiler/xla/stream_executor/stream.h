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

// The Stream is used in conjunction with the StreamExecutor "parent" to
// perform actions with a linear stream of dependencies. Dependencies can also
// be created between Streams to do task management (i.e. limit which tasks
// can be performed concurrently and specify what task dependencies exist).

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_STREAM_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_STREAM_H_

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/stream_executor/event.h"
#include "tensorflow/compiler/xla/stream_executor/fft.h"
#include "tensorflow/compiler/xla/stream_executor/kernel.h"
#include "tensorflow/compiler/xla/stream_executor/launch_dim.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor_pimpl.h"
#include "tensorflow/compiler/xla/stream_executor/temporary_memory_manager.h"

namespace stream_executor {

namespace host {
class HostBlas;
class HostFft;
class HostRng;
class HostTimer;
}  // namespace host

namespace ocl {
class CLBlas;
}  // namespace ocl

namespace internal {
class StreamInterface;
}  // namespace internal

class DeviceMemoryBase;
template <typename ElemT>
class DeviceMemory;

class Timer;

namespace dnn {
class BatchDescriptor;
class FilterDescriptor;
class ConvolutionDescriptor;
class ProfileResult;
class AlgorithmDesc;
}  // namespace dnn

class StreamExecutor;
class ScratchAllocator;

namespace detail {

// Helper class to prevent a template function argument from being deduced. This
// is identical to std::type_identity in C++20.
template <typename T>
struct NonDeduced {
  using type = T;
};
template <typename T>
using NonDeducedType = typename NonDeduced<T>::type;

// Helper to return if `T` is the same type as `First` or any or `Rest`.
template <typename T>
constexpr bool is_any_of() {
  return false;
}

template <typename T, typename First, typename... Rest>
constexpr bool is_any_of() {
  return std::is_same_v<T, First> || is_any_of<T, Rest...>();
}

}  // namespace detail

// Convert a type to the corresponding QuantizedActivationMode.
template <typename ElementType>
struct Quantization;

// Represents a stream of dependent computations on a GPU device.
//
// The operations within a stream execute linearly and asynchronously until
// BlockHostUntilDone() is invoked, which synchronously joins host code with
// the execution of the stream.
//
// If any given operation fails when entraining work for the stream, ok() will
// indicate that an error has occurred. After initialization, once a stream is
// !ok(), it will never be ok().
//
// Thread-safe post-initialization.
class Stream {
 public:
  // Instantiate a stream tied to parent as a platform executor. Work
  // entrained onto this stream will be launched/managed on that
  // StreamExecutor's platform.
  explicit Stream(StreamExecutor *parent);

  // Deallocates any stream resources that the parent StreamExecutor has
  // bestowed
  // upon this object.
  ~Stream();

  // Returns whether any errors have occurred while entraining work for this
  // stream.
  bool ok() const { return !InErrorState(); }

  // Retrieves execution status back into the stream from the underlying
  // implementation without blocking the stream.
  //
  // Normally, Stream::BlockHostUntilDone is used to get execution status.
  // However, some devices use out-of-band mechnanisms to ensure their streams
  // have finished on-device work, without needing to block the streams. (These
  // devices should also override AllowsSyncOnCompletion to return false.) For
  // these devices, this method can be used after work is finished to retrieve
  // execution status.
  tsl::Status RefreshStatus() TF_LOCKS_EXCLUDED(mu_);

  // Initialize the stream. This must be performed before entraining any other
  // operations.
  Stream &Init() TF_LOCKS_EXCLUDED(mu_);

  // Initializes timer t via the StreamExecutor.
  Stream &InitTimer(Timer *t);

  // Convenience wrapper around Init() and InitTimer().
  Stream &InitWithTimer(Timer *t);

  // Get or create a sub-stream from this stream. If there is any sub-stream in
  // the pool that can be reused then just return this sub-stream.  Otherwise
  // create a new sub-stream.
  //
  // TODO(b/112196569): The semantics of failed sub-streams is error-prone.
  Stream *GetOrCreateSubStream() TF_LOCKS_EXCLUDED(mu_);

  // Return the sub-stream back to the host stream so that it can be reused
  // later. Sub-streams that are !ok() will not be reused.
  //
  // TODO(b/112196569): The semantics of failed sub-streams is error-prone.
  void ReturnSubStream(Stream *sub_stream) TF_LOCKS_EXCLUDED(mu_);

  // Allocate temporary memories. The stream will deallocate them when blocked
  // or destroyed.
  template <typename T>
  tsl::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>>
  AllocateTemporaryArray(uint64_t element_count);

  // Entrains onto the stream of operations: a kernel launch with the given
  // (variadic) parameters for the invocation. These arguments can be things
  // like DeviceMemory or primitive types such as int. What arguments you may
  // pass to a given kernel are noted as the template parameters to the
  // TypedKernel type that the machocc compiler generates.
  //
  // Template parameters:
  //  Params...   The type list of formal parameters that the typed kernel
  //              expects, which is matched against Args...
  //  Args...     The deduced type list for passed actual arguments
  //
  // Implementation: A compile-time compatibility check is performed that has
  // some leniency versus an exact parameter pack match -- for example,
  // `const DeviceMemory<T>` is considered "pack compatible" with a
  // `const DeviceMemory<T>&` formal parameter; in part, because we don't have
  // perfect forwarding support without rvalue references. It also attempts to
  // spit out helpful static_assert error traces with information as to the
  // argument number and types that were mismatched.
  template <typename... Params, typename... Args>
  tsl::Status ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                         const TypedKernel<Params...> &kernel, Args... args);

  // Record a "start" event for the interval timer at this point in the
  // stream's execution (relative to the previously and subsequently enqueued
  // items in the stream's execution). Streams may be started/stopped multiple
  // times.
  Stream &ThenStartTimer(Timer *t);

  // Record a "stop" event for the interval timer at this point in the
  // stream's execution. See also Stream::ThenStartTimer.
  Stream &ThenStopTimer(Timer *t);

  // TODO(leary) If work is added to the stream that is being depended upon,
  //              then what? Have to describe what happens.
  template <typename... Params>
  Stream &ThenWaitFor(Stream *other, Params... more_streams) {
    return ThenWaitFor(more_streams...).ThenWaitFor(other);
  }

  // Create a dependency for this stream's next work on the other stream
  // completing. Does not take ownership of other, and other must not be
  // null.
  //
  // Checks that a stream does not wait for itself, and it is up to the
  // user to guarantee that a stream does not come to wait on itself in a
  // cyclic manner; in that case, behavior is undefined.
  //
  // N.B. Base recursion case for the variadic ThenWaitFor.
  Stream &ThenWaitFor(Stream *other);

  // Waits for all streams values in others.
  // Checks that there is no shallow circular wait (i.e. that "this" is not in
  // others)
  template <typename P>
  Stream &ThenWaitFor(P others) {
    for (auto &stream : *others) {
      CHECK_NE(stream.get(), this);
      ThenWaitFor(stream.get());
    }
    return *this;
  }

  // Waits for an event object to be set.
  // Note that ThenRecordEvent must have been called on the event before
  // you call this function; otherwise the event will be considered complete
  // and this wait will do nothing.
  Stream &ThenWaitFor(Event *event);

  // Inserts the specified event into the end of this stream. Once the stream
  // has processed all events prior to the insertion point, the event will be
  // marked as completed.
  // The stream does not take ownership of event - meaning that event's lifetime
  // must extend past the point at which it is marked complete!
  Stream &ThenRecordEvent(Event *event);

  ////////////////
  // DNN support
  //
  // See DnnSupport::* for comments on the following methods.

  Stream &ThenBatchNormalizationForward(
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
      ScratchAllocator *workspace_allocator);

  Stream &ThenBatchNormalizationBackward(
      const DeviceMemory<float> &y_backprop, const DeviceMemory<float> &x,
      const DeviceMemory<float> &scale, const DeviceMemory<float> &offset,
      const DeviceMemory<float> &mean, const DeviceMemory<float> &inv_var,
      const DeviceMemory<float> &y, const dnn::BatchDescriptor &x_desc,
      const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
      dnn::ActivationMode activation_mode, DeviceMemory<float> *x_backprop,
      DeviceMemory<float> *scale_backprop, DeviceMemory<float> *offset_backprop,
      DeviceMemory<float> *side_input_backprop,
      DeviceMemory<uint8_t> *reserve_space_data,
      ScratchAllocator *workspace_allocator);

  Stream &ThenBatchNormalizationForward(
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
      ScratchAllocator *workspace_allocator);

  Stream &ThenBatchNormalizationBackward(
      const DeviceMemory<Eigen::half> &y_backprop,
      const DeviceMemory<Eigen::half> &x, const DeviceMemory<float> &scale,
      const DeviceMemory<float> &offset, const DeviceMemory<float> &mean,
      const DeviceMemory<float> &inv_var, const DeviceMemory<Eigen::half> &y,
      const dnn::BatchDescriptor &x_desc,
      const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
      dnn::ActivationMode activation_mode,
      DeviceMemory<Eigen::half> *x_backprop,
      DeviceMemory<float> *scale_backprop, DeviceMemory<float> *offset_backprop,
      DeviceMemory<Eigen::half> *side_input_backprop,
      DeviceMemory<uint8_t> *reserve_space_data,
      ScratchAllocator *workspace_allocator);

  Stream &ThenBatchNormalizationForward(
      const DeviceMemory<Eigen::bfloat16> &x, const DeviceMemory<float> &scale,
      const DeviceMemory<float> &offset,
      const DeviceMemory<float> &estimated_mean,
      const DeviceMemory<float> &estimated_variance,
      const DeviceMemory<Eigen::bfloat16> &side_input,
      const dnn::BatchDescriptor &x_desc,
      const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
      const double exponential_average_factor,
      dnn::ActivationMode activation_mode, DeviceMemory<Eigen::bfloat16> *y,
      DeviceMemory<float> *batch_mean, DeviceMemory<float> *batch_var,
      DeviceMemory<float> *saved_mean, DeviceMemory<float> *saved_inv_var,
      bool is_training, ScratchAllocator *reserve_space_allocator,
      ScratchAllocator *workspace_allocator);

  Stream &ThenBatchNormalizationBackward(
      const DeviceMemory<Eigen::bfloat16> &y_backprop,
      const DeviceMemory<Eigen::bfloat16> &x, const DeviceMemory<float> &scale,
      const DeviceMemory<float> &offset, const DeviceMemory<float> &mean,
      const DeviceMemory<float> &inv_var,
      const DeviceMemory<Eigen::bfloat16> &y,
      const dnn::BatchDescriptor &x_desc,
      const dnn::BatchDescriptor &scale_offset_desc, const double epsilon,
      dnn::ActivationMode activation_mode,
      DeviceMemory<Eigen::bfloat16> *x_backprop,
      DeviceMemory<float> *scale_backprop, DeviceMemory<float> *offset_backprop,
      DeviceMemory<Eigen::bfloat16> *side_input_backprop,
      DeviceMemory<uint8_t> *reserve_space_data,
      ScratchAllocator *workspace_allocator);

  Stream &ThenConvolve(const dnn::BatchDescriptor &input_descriptor,
                       const DeviceMemory<float> &input_data,
                       const dnn::FilterDescriptor &filter_descriptor,
                       const DeviceMemory<float> &filter_data,
                       const dnn::ConvolutionDescriptor &convolution_descriptor,
                       const dnn::BatchDescriptor &output_descriptor,
                       DeviceMemory<float> *output);

  Stream &ThenConvolveQuantized(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<float> &input_data,
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<int8_t> &filter_coefficients,
      const DeviceMemory<float> &coefficient_scales,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> *output_data);

  Stream &ThenConvolveQuantized(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<float> &input_data,
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<int16> &filter_coefficients,
      const DeviceMemory<float> &coefficient_scales,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> *output_data);

  template <typename InputType, typename OutputType>
  tsl::Status ConvolveWithAlgorithm(
      dnn::ConvolutionKind kind, const dnn::BatchDescriptor &input_descriptor,
      DeviceMemory<InputType> input_data,
      const dnn::FilterDescriptor &filter_descriptor,
      DeviceMemory<InputType> filter_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<OutputType> output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      ScratchAllocator *scratch_allocator,
      const dnn::AlgorithmConfig &algorithm_config,
      dnn::ProfileResult *output_profile_result) {
    DeviceMemory<uint8_t> scratch_memory;
    dnn::AlgorithmDesc algorithm_desc;
    if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
      TF_RETURN_IF_ERROR(dnn->PrepareForConvolution(
          kind, this, input_descriptor, input_data, filter_descriptor,
          filter_data, output_descriptor, output_data, convolution_descriptor,
          algorithm_config, scratch_allocator, &algorithm_desc,
          &scratch_memory));
      return dnn->DoConvolve(kind, dnn::ToDataType<InputType>::value,
                             dnn::ToDataType<OutputType>::value, this,
                             input_descriptor, input_data, filter_descriptor,
                             filter_data, output_descriptor, output_data,
                             convolution_descriptor, algorithm_desc,
                             scratch_memory, output_profile_result);
    }
    return tsl::errors::Unimplemented("DNN library is not found.");
  }

  template <typename InputT, typename ScaleT, typename SideInputT,
            typename BiasT, typename OutputT>
  tsl::Status FusedConvolveWithAlgorithm(
      const dnn::BatchDescriptor &conv_input_descriptor,
      const DeviceMemory<InputT> &conv_input_data, ScaleT conv_input_scale,
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<InputT> &filter_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const DeviceMemory<SideInputT> &side_input_data, ScaleT side_input_scale,
      const dnn::BatchDescriptor &bias_descriptor,
      const DeviceMemory<BiasT> &biases, dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<OutputT> *output, ScratchAllocator *scratch_allocator,
      const dnn::AlgorithmConfig &algorithm_config,
      dnn::ProfileResult *output_profile_result) {
    if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
      return dnn->DoFusedConvolve(
          this, dnn::ToDataType<InputT>::value,
          dnn::ToDataType<SideInputT>::value, dnn::ToDataType<BiasT>::value,
          dnn::ToDataType<OutputT>::value, conv_input_descriptor,
          conv_input_data, conv_input_scale, filter_descriptor, filter_data,
          convolution_descriptor, side_input_data, side_input_scale,
          bias_descriptor, biases, activation_mode, output_descriptor, *output,
          scratch_allocator, algorithm_config, output_profile_result);
    }
    return tsl::errors::Unimplemented("DNN library is not found.");
  }

  tsl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> ConvolveRunnerFromDesc(
      const dnn::AlgorithmDesc &algorithm_desc, dnn::ConvolutionKind kind,
      dnn::DataType element_type, dnn::DataType output_type,
      const dnn::BatchDescriptor &input_descriptor,
      const dnn::FilterDescriptor &filter_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      const dnn::ConvolutionDescriptor &convolution_descriptor) {
    dnn::DnnSupport *dnn_support = parent_->AsDnn();
    if (!dnn_support) {
      return tsl::errors::Unimplemented("DNN library is not found.");
    }
    return dnn_support->ConvolveRunnerFromDesc(
        this, algorithm_desc, kind, element_type, output_type, input_descriptor,
        filter_descriptor, output_descriptor, convolution_descriptor);
  }

  tsl::StatusOr<std::unique_ptr<const dnn::FusedConvRunner>>
  FusedConvolveRunnerFromDesc(
      const dnn::AlgorithmDesc &algorithm_desc, dnn::ConvolutionKind kind,
      dnn::DataType element_type, dnn::DataType bias_type,
      dnn::DataType output_type, double conv_input_scale,
      double side_input_scale, double leakyrelu_alpha,
      const dnn::BatchDescriptor &input_descriptor,
      const dnn::FilterDescriptor &filter_descriptor,
      const dnn::BatchDescriptor &bias_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      dnn::ActivationMode activation_mode) {
    dnn::DnnSupport *dnn_support = parent_->AsDnn();
    if (!dnn_support) {
      return tsl::errors::Unimplemented("DNN library is not found.");
    }
    return dnn_support->FusedConvolveRunnerFromDesc(
        this, algorithm_desc, kind, element_type, bias_type, output_type,
        conv_input_scale, side_input_scale, leakyrelu_alpha, input_descriptor,
        filter_descriptor, bias_descriptor, output_descriptor,
        convolution_descriptor, activation_mode);
  }

  Stream &ThenSeparableConvolve(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<float> &input_data,
      const dnn::FilterDescriptor &filter_descriptor, int depth_multiplier,
      const DeviceMemory<float> &first_weights,
      const DeviceMemory<float> &second_weights,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> *output);

  Stream &ThenMatMul(const DeviceMemory<float> &input_data,
                     const DeviceMemory<float> &weights,
                     const dnn::BatchDescriptor &input_dimensions,
                     const dnn::BatchDescriptor &output_dimensions,
                     DeviceMemory<float> *output_data);

  Stream &ThenMatMulQuantized(const DeviceMemory<float> &input_data,
                              const DeviceMemory<int8_t> &weights,
                              const DeviceMemory<float> &weight_scales,
                              const dnn::BatchDescriptor &input_dimensions,
                              const dnn::BatchDescriptor &output_dimensions,
                              DeviceMemory<float> *output_data);

  Stream &ThenMatMulQuantized(const DeviceMemory<float> &input_data,
                              const DeviceMemory<int16> &weights,
                              const DeviceMemory<float> &weight_scales,
                              const dnn::BatchDescriptor &input_dimensions,
                              const dnn::BatchDescriptor &output_dimensions,
                              DeviceMemory<float> *output_data);

  Stream &ThenBiasAdd(const DeviceMemory<float> &input_data,
                      const DeviceMemory<float> &biases,
                      const dnn::BatchDescriptor &dimensions,
                      DeviceMemory<float> *output_data);

  template <typename ElementType>
  tsl::Status ThenPoolForward(const dnn::PoolingDescriptor &pooling_dimensions,
                              const dnn::BatchDescriptor &input_dimensions,
                              const DeviceMemory<ElementType> &input_data,
                              const dnn::BatchDescriptor &output_dimensions,
                              DeviceMemory<ElementType> *output_data,
                              ScratchAllocator *workspace_allocator = nullptr) {
    if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
      return dnn->DoPoolForward(dnn::ToDataType<ElementType>::value, this,
                                pooling_dimensions, input_dimensions,
                                input_data, output_dimensions, *output_data,
                                workspace_allocator);
    }
    return tsl::errors::Unimplemented("DNN library is not found.");
  }

  template <typename ElementType>
  tsl::Status ThenPoolBackward(
      const dnn::PoolingDescriptor &pooling_dimensions,
      const dnn::BatchDescriptor &input_dimensions,
      const DeviceMemory<ElementType> &input_data,
      const dnn::BatchDescriptor &output_dimensions,
      const DeviceMemory<ElementType> &output_data,
      const DeviceMemory<ElementType> &input_diff_data,
      DeviceMemory<ElementType> *output_diff_data,
      ScratchAllocator *workspace_allocator = nullptr) {
    if (dnn::DnnSupport *dnn = parent_->AsDnn()) {
      return dnn->DoPoolBackward(
          dnn::ToDataType<ElementType>::value, this, pooling_dimensions,
          input_dimensions, input_data, output_dimensions, output_data,
          input_diff_data, *output_diff_data, workspace_allocator);
    }
    return tsl::errors::Unimplemented("DNN library is not found.");
  }

  Stream &ThenNormalizeWithDimensions(
      const dnn::NormalizeDescriptor &normalize_descriptor,
      const dnn::BatchDescriptor &dimensions,
      const DeviceMemory<float> &input_data, DeviceMemory<float> *output_data);

  Stream &ThenNormalizeBackwardWithDimensions(
      const dnn::NormalizeDescriptor &normalize_descriptor,
      const dnn::BatchDescriptor &dimensions,
      const DeviceMemory<float> &raw_data,
      const DeviceMemory<float> &normalized_data,
      const DeviceMemory<float> &normalized_variable_gradient,
      DeviceMemory<float> *raw_variable_gradient,
      ScratchAllocator *workspace_allocator = nullptr);

  Stream &ThenActivate(dnn::ActivationMode activation_mode,
                       const dnn::BatchDescriptor &dimensions,
                       const DeviceMemory<float> &input_data,
                       DeviceMemory<float> *output_data);

  // Same as ThenActivate, but also takes an options argument that can be used
  // for platform-specific option flags.
  Stream &ThenActivateWithOptions(dnn::ActivationMode activation_mode,
                                  const dnn::BatchDescriptor &dimensions,
                                  const DeviceMemory<float> &input_data,
                                  DeviceMemory<float> *output_data,
                                  uint64_t options);

  Stream &ThenDepthConcatenate(
      absl::Span<const dnn::BatchDescriptor> input_dimensions,
      absl::Span<const DeviceMemory<float> *const> input_data,
      DeviceMemory<float> *output_data);

  Stream &ThenSpaceConcatenate(
      absl::Span<const dnn::BatchDescriptor> input_dimensions,
      absl::Span<const DeviceMemory<float> *const> input_data,
      DeviceMemory<float> *output_data,
      dnn::SpaceConcatenateMode concat_direction);

  // Change the layout of the data by shrinking one dimension (or set of
  // dimensions) and growing another dimension (or set of dimensions), while
  // keeping the total number of data elements constant, and maintaining the
  // current data ordering.
  Stream &ThenReshape(const dnn::BatchDescriptor &input_dimensions,
                      const DeviceMemory<float> &input_data,
                      const dnn::BatchDescriptor &output_dimensions,
                      DeviceMemory<float> *output_data);

  // Depth to space takes an X by Y image with depth D*M² and changes it to an
  // MX x MY image with depth D. Each input location (x,y) with depth D*M² in
  // the input image is changed to an MxM contiguous area in the output image,
  // with the values being laid out in raster order specified by
  // DepthToSpaceLayout, and will have a new depth of D.
  // See the DoDepthToSpace comment for more information.
  Stream &ThenDepthToSpace(const dnn::BatchDescriptor &input_dimensions,
                           const DeviceMemory<float> &input_data,
                           const dnn::DepthToSpaceLayout &depth_to_space_layout,
                           const int sqrt_depth_reduction,
                           DeviceMemory<float> *output_data);

  // Space to depth is the inverse of depth to space. Space to depth takes each
  // non-overlapping M by M patch (in the X and Y dimensions) with depth D of
  // the input, and transforms it to a 1 by 1 patch with depth D*M². If the
  // input has size (MX, MY, D), the output has size (X, Y, D*M²). The number of
  // data elements is not changed.
  Stream &ThenSpaceToDepth(const dnn::BatchDescriptor &input_dimensions,
                           const DeviceMemory<float> &input_data,
                           const dnn::DepthToSpaceLayout &space_to_depth_layout,
                           const int sqrt_depth_increase,
                           DeviceMemory<float> *output_data);

  Stream &ThenElementwiseOperate(
      dnn::ElementwiseOperation operation,
      absl::Span<const dnn::BatchDescriptor> input_dimensions,
      absl::Span<const DeviceMemory<float> *const> input_data,
      const dnn::BatchDescriptor &output_dimensions,
      DeviceMemory<float> *output_data);

  Stream &ThenElementwiseOperateScaledQuantized(
      dnn::ElementwiseOperation operation,
      absl::Span<const int> input_multiplicands, int output_divisor,
      absl::Span<const dnn::BatchDescriptor> input_dimensions,
      absl::Span<const DeviceMemory<float> *const> input_data,
      const dnn::BatchDescriptor &output_dimensions,
      DeviceMemory<float> *output_data);

  Stream &ThenXYPad(const dnn::BatchDescriptor &dimensions,
                    const DeviceMemory<float> &input_data, int64_t left_pad,
                    int64_t right_pad, int64_t top_pad, int64_t bottom_pad,
                    DeviceMemory<float> *output_data);

  Stream &ThenXYSlice(const dnn::BatchDescriptor &dimensions,
                      const DeviceMemory<float> &input_data, int64_t left_trim,
                      int64_t right_trim, int64_t top_trim, int64_t bottom_trim,
                      DeviceMemory<float> *output_data);

  // Grows the input tensor by replicating the X and Y dimensions. The batch and
  // depth/feature_map dimensions are unchanged. Currently, the input tensor is
  // limited to X=1 and Y=1.
  Stream &ThenXYBroadcast(const dnn::BatchDescriptor &dimensions,
                          const DeviceMemory<float> &input_data,
                          int64_t replicate_x, int64_t replicate_y,
                          DeviceMemory<float> *output_data);

  // See DnnSupport::DoMemcpyD2HQuantized.
  Stream &ThenMemcpyD2HQuantized(const DeviceMemory<float> &gpu_unquantized_src,
                                 dnn::QuantizedActivationMode mode,
                                 void *host_dst, uint64_t size);

  // Template version of ThenMemcpyD2HQuantized that takes a mutable span and
  // uses the Quantization trait to call the generic version of
  // ThenMemcpyD2HQuantized with the correct QuantizedActivationMode.
  template <typename ElementType>
  Stream &ThenMemcpyD2HQuantized(const DeviceMemory<float> &gpu_unquantized_src,
                                 absl::Span<ElementType> host_dst) {
    return ThenMemcpyD2HQuantized(
        gpu_unquantized_src, Quantization<ElementType>::kModeId,
        host_dst.data(), host_dst.size() * sizeof(ElementType));
  }

  // See DnnSupport::DoMemcpyH2DQuantized.
  Stream &ThenMemcpyH2DQuantized(const void *host_src, uint64_t size,
                                 dnn::QuantizedActivationMode mode,
                                 DeviceMemory<float> *gpu_unquantized_dst);

  // Template version of ThenMemcpyH2DQuantized that takes an array slice
  // and uses the Quantization trait to call the generic version of
  // ThenMemcpyH2DQuantized with the correct QuantizedActivationMode.
  template <typename ElementType>
  Stream &ThenMemcpyH2DQuantized(absl::Span<const ElementType> host_src,
                                 DeviceMemory<float> *gpu_unquantized_dst) {
    return ThenMemcpyH2DQuantized(
        host_src.data(), host_src.size() * sizeof(ElementType),
        Quantization<ElementType>::kModeId, gpu_unquantized_dst);
  }

  // See DnnSupport::DoCopyHostBuffer2Device.
  Stream &ThenCopyHostBuffer2Device(HostBuffer *buffer_src,
                                    DeviceMemory<float> *gpu_unquantized_dst);

  // See DnnSupport::DoCopyDevice2HostBuffer.
  Stream &ThenCopyDevice2HostBuffer(
      const DeviceMemory<float> &gpu_unquantized_src, HostBuffer *buffer_dst);

  /////////////////
  // BLAS support

  // See BlasSupport::DoBlasAxpy. Note that, even for the case where alpha is
  // present in DeviceMemory, it must be an execution-time constant (i.e. a
  // value
  // that the stream does not change or populate during the course of
  // execution). The value is effectively captured at stream-enqueue time.
  Stream &ThenBlasAxpy(uint64_t elem_count, float alpha,
                       const DeviceMemory<float> &x, int incx,
                       DeviceMemory<float> *y, int incy);
  Stream &ThenBlasAxpy(uint64_t elem_count, double alpha,
                       const DeviceMemory<double> &x, int incx,
                       DeviceMemory<double> *y, int incy);
  Stream &ThenBlasAxpy(uint64_t elem_count, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasAxpy(uint64_t elem_count, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       DeviceMemory<std::complex<double>> *y, int incy);

  // See BlasSupport::DoBlasCopy.
  Stream &ThenBlasCopy(uint64_t elem_count, const DeviceMemory<float> &x,
                       int incx, DeviceMemory<float> *y, int incy);
  Stream &ThenBlasCopy(uint64_t elem_count, const DeviceMemory<double> &x,
                       int incx, DeviceMemory<double> *y, int incy);
  Stream &ThenBlasCopy(uint64_t elem_count,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasCopy(uint64_t elem_count,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       DeviceMemory<std::complex<double>> *y, int incy);

  // See BlasSupport::DoBlasScal.
  Stream &ThenBlasScal(uint64_t elem_count, float alpha, DeviceMemory<float> *x,
                       int incx);
  Stream &ThenBlasScal(uint64_t elem_count, double alpha,
                       DeviceMemory<double> *x, int incx);
  Stream &ThenBlasScal(uint64_t elem_count, float alpha,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasScal(uint64_t elem_count, double alpha,
                       DeviceMemory<std::complex<double>> *x, int incx);
  Stream &ThenBlasScal(uint64_t elem_count, std::complex<float> alpha,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasScal(uint64_t elem_count, std::complex<double> alpha,
                       DeviceMemory<std::complex<double>> *x, int incx);

  // See BlasSupport::DoBlasGemv.
  Stream &ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n, float alpha,
                       const DeviceMemory<float> &a, int lda,
                       const DeviceMemory<float> &x, int incx, float beta,
                       DeviceMemory<float> *y, int incy);
  Stream &ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                       double alpha, const DeviceMemory<double> &a, int lda,
                       const DeviceMemory<double> &x, int incx, double beta,
                       DeviceMemory<double> *y, int incy);
  Stream &ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                       std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasGemv(blas::Transpose trans, uint64_t m, uint64 n,
                       std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *y, int incy);

  Stream &ThenBlasGemvWithProfiling(blas::Transpose trans, uint64_t m, uint64 n,
                                    float alpha, const DeviceMemory<float> &a,
                                    int lda, const DeviceMemory<float> &x,
                                    int incx, float beta,
                                    DeviceMemory<float> *y, int incy,
                                    blas::ProfileResult *output_profile_result);
  Stream &ThenBlasGemvWithProfiling(blas::Transpose trans, uint64_t m, uint64 n,
                                    double alpha, const DeviceMemory<double> &a,
                                    int lda, const DeviceMemory<double> &x,
                                    int incx, double beta,
                                    DeviceMemory<double> *y, int incy,
                                    blas::ProfileResult *output_profile_result);
  Stream &ThenBlasGemvWithProfiling(
      blas::Transpose trans, uint64_t m, uint64 n, std::complex<float> alpha,
      const DeviceMemory<std::complex<float>> &a, int lda,
      const DeviceMemory<std::complex<float>> &x, int incx,
      std::complex<float> beta, DeviceMemory<std::complex<float>> *y, int incy,
      blas::ProfileResult *output_profile_result);
  Stream &ThenBlasGemvWithProfiling(
      blas::Transpose trans, uint64_t m, uint64 n, std::complex<double> alpha,
      const DeviceMemory<std::complex<double>> &a, int lda,
      const DeviceMemory<std::complex<double>> &x, int incx,
      std::complex<double> beta, DeviceMemory<std::complex<double>> *y,
      int incy, blas::ProfileResult *output_profile_result);

  // See BlasSupport::DoBlasSbmv.
  Stream &ThenBlasSbmv(blas::UpperLower uplo, uint64_t n, uint64 k, float alpha,
                       const DeviceMemory<float> &a, int lda,
                       const DeviceMemory<float> &x, int incx, float beta,
                       DeviceMemory<float> *y, int incy);
  Stream &ThenBlasSbmv(blas::UpperLower uplo, uint64_t n, uint64 k,
                       double alpha, const DeviceMemory<double> &a, int lda,
                       const DeviceMemory<double> &x, int incx, double beta,
                       DeviceMemory<double> *y, int incy);

  template <typename InputType>
  tsl::Status ThenBlasGemm(blas::Transpose transa, blas::Transpose transb,
                           uint64_t m, uint64 n, uint64 k,
                           const DeviceMemory<InputType> &a, int lda,
                           const DeviceMemory<InputType> &b, int ldb,
                           DeviceMemory<InputType> *c, int ldc,
                           blas::ComputePrecision precision) {
    InputType alpha{1.0};
    InputType beta{0.0};
    return ThenBlasGemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, precision);
  }

  // TODO(parkers): Update all callers to pass kDefaultComputePrecision.
  template <typename InputType>
  tsl::Status ThenBlasGemm(blas::Transpose transa, blas::Transpose transb,
                           uint64_t m, uint64 n, uint64 k,
                           const DeviceMemory<InputType> &a, int lda,
                           const DeviceMemory<InputType> &b, int ldb,
                           DeviceMemory<InputType> *c, int ldc) {
    return ThenBlasGemm(transa, transb, m, n, k, a, lda, b, ldb, c, ldc,
                        blas::kDefaultComputePrecision);
  }

  template <typename InputType, typename ConstantType>
  tsl::Status ThenBlasGemm(blas::Transpose transa, blas::Transpose transb,
                           uint64_t m, uint64 n, uint64 k, ConstantType alpha,
                           const DeviceMemory<InputType> &a, int lda,
                           const DeviceMemory<InputType> &b, int ldb,
                           ConstantType beta, DeviceMemory<InputType> *c,
                           int ldc, blas::ComputePrecision precision) {
    static_assert(
        detail::is_any_of<InputType, Eigen::half, Eigen::bfloat16, float,
                          double, std::complex<float>, std::complex<double>>(),
        "Input can be half, bf16, float, double, std::complex<float> or "
        "std::complex<double>");
    static_assert(!std::is_same_v<InputType, Eigen::half> ||
                      detail::is_any_of<ConstantType, float, Eigen::half>(),
                  "If input is Eigen::half, constant has to be either "
                  "Eigen::half or float");
    static_assert(
        detail::is_any_of<InputType, Eigen::half, ConstantType>(),
        "If input is not Eigen::half, constant and input types have to match");
    blas::BlasSupport *blas = parent()->AsBlas();
    if (!blas) {
      return tsl::errors::Internal(
          "Attempting to perform BLAS operation using "
          "StreamExecutor without BLAS support");
    }

    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);

    return blas->DoBlasGemm(this, transa, transb, m, n, k,
                            blas::ToDataType<InputType>::value, alpha_ptr, a,
                            lda, b, ldb, beta_ptr, c, ldc, precision);
  }

  // TODO(parkers): Update all callers to pass kDefaultComputePrecision.
  template <typename InputType, typename ConstantType>
  tsl::Status ThenBlasGemm(blas::Transpose transa, blas::Transpose transb,
                           uint64_t m, uint64 n, uint64 k, ConstantType alpha,
                           const DeviceMemory<InputType> &a, int lda,
                           const DeviceMemory<InputType> &b, int ldb,
                           ConstantType beta, DeviceMemory<InputType> *c,
                           int ldc) {
    return ThenBlasGemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
                        ldc, blas::kDefaultComputePrecision);
  }

  Stream &ThenBlasGemmWithProfiling(blas::Transpose transa,
                                    blas::Transpose transb, uint64_t m,
                                    uint64 n, uint64_t k, float alpha,
                                    const DeviceMemory<Eigen::half> &a, int lda,
                                    const DeviceMemory<Eigen::half> &b, int ldb,
                                    float beta, DeviceMemory<Eigen::half> *c,
                                    int ldc,
                                    blas::ProfileResult *output_profile_result);
  Stream &ThenBlasGemmWithProfiling(blas::Transpose transa,
                                    blas::Transpose transb, uint64_t m,
                                    uint64 n, uint64_t k, float alpha,
                                    const DeviceMemory<float> &a, int lda,
                                    const DeviceMemory<float> &b, int ldb,
                                    float beta, DeviceMemory<float> *c, int ldc,
                                    blas::ProfileResult *output_profile_result);
  Stream &ThenBlasGemmWithProfiling(blas::Transpose transa,
                                    blas::Transpose transb, uint64_t m,
                                    uint64 n, uint64_t k, double alpha,
                                    const DeviceMemory<double> &a, int lda,
                                    const DeviceMemory<double> &b, int ldb,
                                    double beta, DeviceMemory<double> *c,
                                    int ldc,
                                    blas::ProfileResult *output_profile_result);
  Stream &ThenBlasGemmWithProfiling(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, std::complex<float> alpha,
      const DeviceMemory<std::complex<float>> &a, int lda,
      const DeviceMemory<std::complex<float>> &b, int ldb,
      std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
      blas::ProfileResult *output_profile_result);
  Stream &ThenBlasGemmWithProfiling(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, std::complex<double> alpha,
      const DeviceMemory<std::complex<double>> &a, int lda,
      const DeviceMemory<std::complex<double>> &b, int ldb,
      std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
      blas::ProfileResult *output_profile_result);

  template <typename InputType, typename OutputType>
  tsl::Status ThenBlasGemmWithAlgorithm(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, const DeviceMemory<InputType> &a, int lda,
      const DeviceMemory<InputType> &b, int ldb, DeviceMemory<OutputType> *c,
      int ldc, blas::ComputationType computation_type,
      blas::AlgorithmType algorithm,
      blas::ProfileResult *output_profile_result) {
    OutputType alpha{1};
    OutputType beta{0};
    return ThenBlasGemmWithAlgorithm(transa, transb, m, n, k, alpha, a, lda, b,
                                     ldb, beta, c, ldc, computation_type,
                                     algorithm, blas::kDefaultComputePrecision,
                                     output_profile_result);
  }

  template <typename InputType, typename OutputType, typename ConstantType>
  tsl::Status ThenBlasGemmWithAlgorithm(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, ConstantType alpha, const DeviceMemory<InputType> &a, int lda,
      const DeviceMemory<InputType> &b, int ldb, ConstantType beta,
      DeviceMemory<OutputType> *c, int ldc,
      blas::ComputationType computation_type, blas::AlgorithmType algorithm,
      blas::ComputePrecision precision,
      blas::ProfileResult *output_profile_result) {
    TF_RETURN_IF_ERROR(
        CheckTypesForExtendedBlas<InputType, OutputType, ConstantType>(
            computation_type));

    blas::BlasSupport *blas = parent()->AsBlas();
    if (!blas) {
      return tsl::errors::Internal(
          "Attempting to perform BLAS operation using "
          "StreamExecutor without BLAS support");
    }

    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);

    tsl::Status st = blas->DoBlasGemmWithAlgorithm(
        this, transa, transb, m, n, k, alpha_ptr, a,
        blas::ToDataType<InputType>::value, lda, b,
        blas::ToDataType<InputType>::value, ldb, beta_ptr, c,
        blas::ToDataType<OutputType>::value, ldc, computation_type, algorithm,
        precision, output_profile_result);
    if (output_profile_result) {
      // The error is recorded in the profile.
      return ::tsl::OkStatus();
    }
    return st;
  }

  template <typename InputType, typename OutputType, typename ConstantType>
  tsl::Status ThenBlasGemmStridedBatchedWithAlgorithm(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, ConstantType alpha, const DeviceMemory<InputType> &a, int lda,
      int64_t stride_a, const DeviceMemory<InputType> &b, int ldb,
      int64_t stride_b, ConstantType beta, DeviceMemory<OutputType> *c, int ldc,
      int64_t stride_c, int batch_count, blas::ComputationType computation_type,
      blas::AlgorithmType algorithm, blas::ComputePrecision precision,
      blas::ProfileResult *output_profile_result) {
    TF_RETURN_IF_ERROR(
        CheckTypesForExtendedBlas<InputType, OutputType, ConstantType>(
            computation_type));

    blas::BlasSupport *blas = parent()->AsBlas();
    if (!blas) {
      return tsl::errors::Internal(
          "Attempting to perform BLAS operation using "
          "StreamExecutor without BLAS support");
    }
    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);
    tsl::Status st = blas->DoBlasGemmStridedBatchedWithAlgorithm(
        this, transa, transb, m, n, k, alpha_ptr, a,
        blas::ToDataType<InputType>::value, lda, stride_a, b,
        blas::ToDataType<InputType>::value, ldb, stride_b, beta_ptr, c,
        blas::ToDataType<OutputType>::value, ldc, stride_c, batch_count,
        computation_type, algorithm, precision, output_profile_result);
    if (output_profile_result) {
      // The error is recorded in the profile.
      return ::tsl::OkStatus();
    }
    return st;
  }

  template <typename T>
  using DeviceMemorySlice = absl::Span<DeviceMemory<T> *const>;

  // See BlasSupport::DoBlasGemmBatched.
  Stream &ThenBlasGemmBatched(blas::Transpose transa, blas::Transpose transb,
                              uint64_t m, uint64 n, uint64_t k, float alpha,
                              DeviceMemorySlice<Eigen::half> a, int lda,
                              DeviceMemorySlice<Eigen::half> b, int ldb,
                              float beta, DeviceMemorySlice<Eigen::half> c,
                              int ldc, int batch_count);
  Stream &ThenBlasGemmBatched(blas::Transpose transa, blas::Transpose transb,
                              uint64_t m, uint64 n, uint64 k, float alpha,
                              DeviceMemorySlice<float> a, int lda,
                              DeviceMemorySlice<float> b, int ldb, float beta,
                              DeviceMemorySlice<float> c, int ldc,
                              int batch_count);
  Stream &ThenBlasGemmBatched(blas::Transpose transa, blas::Transpose transb,
                              uint64_t m, uint64 n, uint64 k, double alpha,
                              DeviceMemorySlice<double> a, int lda,
                              DeviceMemorySlice<double> b, int ldb, double beta,
                              DeviceMemorySlice<double> c, int ldc,
                              int batch_count);
  Stream &ThenBlasGemmBatched(blas::Transpose transa, blas::Transpose transb,
                              uint64_t m, uint64 n, uint64_t k,
                              std::complex<float> alpha,
                              DeviceMemorySlice<std::complex<float>> a, int lda,
                              DeviceMemorySlice<std::complex<float>> b, int ldb,
                              std::complex<float> beta,
                              DeviceMemorySlice<std::complex<float>> c, int ldc,
                              int batch_count);
  Stream &ThenBlasGemmBatched(blas::Transpose transa, blas::Transpose transb,
                              uint64_t m, uint64 n, uint64_t k,
                              std::complex<double> alpha,
                              DeviceMemorySlice<std::complex<double>> a,
                              int lda,
                              DeviceMemorySlice<std::complex<double>> b,
                              int ldb, std::complex<double> beta,
                              DeviceMemorySlice<std::complex<double>> c,
                              int ldc, int batch_count);
  Stream &ThenBlasGemmBatchedWithScratch(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, float alpha, DeviceMemorySlice<Eigen::half> a, int lda,
      DeviceMemorySlice<Eigen::half> b, int ldb, float beta,
      DeviceMemorySlice<Eigen::half> c, int ldc, int batch_count,
      ScratchAllocator *scratch_allocator);
  Stream &ThenBlasGemmBatchedWithScratch(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, float alpha, DeviceMemorySlice<Eigen::bfloat16> a, int lda,
      DeviceMemorySlice<Eigen::bfloat16> b, int ldb, float beta,
      DeviceMemorySlice<Eigen::bfloat16> c, int ldc, int batch_count,
      ScratchAllocator *scratch_allocator);
  Stream &ThenBlasGemmBatchedWithScratch(blas::Transpose transa,
                                         blas::Transpose transb, uint64_t m,
                                         uint64 n, uint64_t k, float alpha,
                                         DeviceMemorySlice<float> a, int lda,
                                         DeviceMemorySlice<float> b, int ldb,
                                         float beta, DeviceMemorySlice<float> c,
                                         int ldc, int batch_count,
                                         ScratchAllocator *scratch_allocator);
  Stream &ThenBlasGemmBatchedWithScratch(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, double alpha, DeviceMemorySlice<double> a, int lda,
      DeviceMemorySlice<double> b, int ldb, double beta,
      DeviceMemorySlice<double> c, int ldc, int batch_count,
      ScratchAllocator *scratch_allocator);
  Stream &ThenBlasGemmBatchedWithScratch(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, std::complex<float> alpha,
      DeviceMemorySlice<std::complex<float>> a, int lda,
      DeviceMemorySlice<std::complex<float>> b, int ldb,
      std::complex<float> beta, DeviceMemorySlice<std::complex<float>> c,
      int ldc, int batch_count, ScratchAllocator *scratch_allocator);
  Stream &ThenBlasGemmBatchedWithScratch(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, std::complex<double> alpha,
      DeviceMemorySlice<std::complex<double>> a, int lda,
      DeviceMemorySlice<std::complex<double>> b, int ldb,
      std::complex<double> beta, DeviceMemorySlice<std::complex<double>> c,
      int ldc, int batch_count, ScratchAllocator *scratch_allocator);

  template <typename InputType, typename ConstantType>
  tsl::Status ThenBlasGemmStridedBatched(
      blas::Transpose transa, blas::Transpose transb, uint64_t m, uint64 n,
      uint64_t k, ConstantType alpha, const DeviceMemory<InputType> &a, int lda,
      int64_t stride_a, const DeviceMemory<InputType> &b, int ldb,
      int64_t stride_b, ConstantType beta, DeviceMemory<InputType> *c, int ldc,
      int64_t stride_c, int batch_count, blas::ComputePrecision precision) {
    static_assert(
        detail::is_any_of<InputType, float, Eigen::half, Eigen::bfloat16,
                          double, std::complex<float>, std::complex<double>>(),
        "Unsupported input type");
    static_assert(
        std::is_same_v<ConstantType, InputType> ||
            (detail::is_any_of<InputType, Eigen::half, Eigen::bfloat16>() &&
             std::is_same_v<ConstantType, float>),
        "Mismatched input and alpha/beta types");
    blas::BlasSupport *blas = parent()->AsBlas();
    if (!blas) {
      return tsl::errors::Internal(
          "Attempting to perform BLAS operation using "
          "StreamExecutor without BLAS support");
    }

    void *alpha_ptr = &alpha;
    void *beta_ptr = &beta;
    float alpha_storage, beta_storage;
    UpcastHalfToFloat<ConstantType>(&alpha_ptr, &beta_ptr, &alpha_storage,
                                    &beta_storage);

    return blas->DoBlasGemmStridedBatched(
        this, transa, transb, m, n, k, blas::ToDataType<InputType>::value,
        alpha_ptr, a, lda, stride_a, b, ldb, stride_b, beta_ptr, c, ldc,
        stride_c, batch_count, precision);
  }

  // See BlasSupport::DoBlasTrsm.
  Stream &ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64_t m,
                       uint64_t n, float alpha, const DeviceMemory<float> &a,
                       int lda, DeviceMemory<float> *b, int ldb);
  Stream &ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64_t m,
                       uint64_t n, double alpha, const DeviceMemory<double> &a,
                       int lda, DeviceMemory<double> *b, int ldb);
  Stream &ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64_t m,
                       uint64_t n, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       DeviceMemory<std::complex<float>> *b, int ldb);
  Stream &ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64_t m,
                       uint64_t n, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       DeviceMemory<std::complex<double>> *b, int ldb);

  // See BlasSupport::DoBlasTrsmBatched.
  Stream &ThenBlasTrsmBatched(blas::Side side, blas::UpperLower uplo,
                              blas::Transpose transa, blas::Diagonal diag,
                              uint64_t m, uint64 n, float alpha,
                              const DeviceMemory<float *> &as, int lda,
                              DeviceMemory<float *> *bs, int ldb,
                              int batch_count);
  Stream &ThenBlasTrsmBatched(blas::Side side, blas::UpperLower uplo,
                              blas::Transpose transa, blas::Diagonal diag,
                              uint64_t m, uint64 n, double alpha,
                              const DeviceMemory<double *> &as, int lda,
                              DeviceMemory<double *> *bs, int ldb,
                              int batch_count);
  Stream &ThenBlasTrsmBatched(blas::Side side, blas::UpperLower uplo,
                              blas::Transpose transa, blas::Diagonal diag,
                              uint64_t m, uint64 n, std::complex<float> alpha,
                              const DeviceMemory<std::complex<float> *> &as,
                              int lda, DeviceMemory<std::complex<float> *> *bs,
                              int ldb, int batch_count);
  Stream &ThenBlasTrsmBatched(blas::Side side, blas::UpperLower uplo,
                              blas::Transpose transa, blas::Diagonal diag,
                              uint64_t m, uint64 n, std::complex<double> alpha,
                              const DeviceMemory<std::complex<double> *> &as,
                              int lda, DeviceMemory<std::complex<double> *> *bs,
                              int ldb, int batch_count);

  // See FftSupport::DoFft.
  Stream &ThenFft(fft::Plan *plan,
                  const DeviceMemory<std::complex<float>> &input,
                  DeviceMemory<std::complex<float>> *output);
  Stream &ThenFft(fft::Plan *plan,
                  const DeviceMemory<std::complex<double>> &input,
                  DeviceMemory<std::complex<double>> *output);
  Stream &ThenFft(fft::Plan *plan, const DeviceMemory<float> &input,
                  DeviceMemory<std::complex<float>> *output);
  Stream &ThenFft(fft::Plan *plan, const DeviceMemory<double> &input,
                  DeviceMemory<std::complex<double>> *output);
  Stream &ThenFft(fft::Plan *plan,
                  const DeviceMemory<std::complex<float>> &input,
                  DeviceMemory<float> *output);
  Stream &ThenFft(fft::Plan *plan,
                  const DeviceMemory<std::complex<double>> &input,
                  DeviceMemory<double> *output);

  // Makes the RNG use the provided value as the basis for further generation.
  // /dev/urandom (good) and /dev/random (better, but sometimes slow) are good
  // sources of seed data if the default (high quality) sources are not
  // desired.
  // For most use cases, this function will not be necessary; each provided
  // back-end implementation will be appropriately seeded by default.
  // At a minimum 16 bytes of data are required in the seed buffer.
  //
  // To seed with good (non-reproducible) data:
  //   File* f = File::Open("/dev/random", "r");
  //   int64_t bytes_read = f->Read(seed_data, bytes_to_read);
  //   < error checking >
  //   stream.ThenSetRngSeed(seed_data, bytes_read);
  //
  // To seed with reproducible data:
  //   uint64_t seed_data[2] = { <data> };
  //   stream.ThenSetRngSeed(seed_data, 16);
  Stream &ThenSetRngSeed(const uint8 *seed, uint64_t seed_bytes);

  // Populates the memory indicated by values with uniform-random-distribution
  // values. TODO(leary) seeding API/description
  //
  // Uses the type and size of the DeviceMemory to infer what data should be
  // populated.
  Stream &ThenPopulateRandUniform(DeviceMemory<float> *values);
  Stream &ThenPopulateRandUniform(DeviceMemory<double> *values);
  Stream &ThenPopulateRandUniform(DeviceMemory<std::complex<float>> *values);
  Stream &ThenPopulateRandUniform(DeviceMemory<std::complex<double>> *values);
  Stream &ThenPopulateRandGaussian(float mean, float stddev,
                                   DeviceMemory<float> *values);
  Stream &ThenPopulateRandGaussian(double mean, double stddev,
                                   DeviceMemory<double> *values);

  // Entrain onto the stream: a memcpy to a host destination from a GPU source
  // of the given target size. host_dst must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate or otherwise allocated and
  // then registered with StreamExecutor::HostMemoryRegister.
  Stream &ThenMemcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                     uint64_t size);

  // Entrain onto the stream: a memcpy to a GPU destination from a host source
  // of the given target size. host_src must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate or otherwise allocated and
  // then registered with StreamExecutor::HostMemoryRegister.
  Stream &ThenMemcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                     uint64_t size);

  // Alternative interface for memcpying from device to host that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  Stream &ThenMemcpyD2H(const DeviceMemory<T> &gpu_src,
                        absl::Span<T> host_dst) {
    auto host_size = host_dst.size() * sizeof(T);
    CHECK(gpu_src.size() == 0 || host_size >= gpu_src.size());
    return ThenMemcpy(host_dst.begin(), gpu_src, host_size);
  }

  // Alternative interface for memcpying from host to device that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  Stream &ThenMemcpyH2D(absl::Span<const T> host_src,
                        DeviceMemory<T> *gpu_dst) {
    auto host_size = host_src.size() * sizeof(T);
    CHECK(gpu_dst->size() == 0 || gpu_dst->size() >= host_size);
    return ThenMemcpy(gpu_dst, host_src.begin(), host_size);
  }

  // Entrain onto the stream: a memcpy to a GPU destination from a GPU source
  // of the given target size. gpu_src/dst must be pointers to GPU memory and
  // peer access must be enabled between their owning StreamExecutors.
  Stream &ThenMemcpy(DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src,
                     uint64_t size);

  // Calls to the device-to-device copy overload of ThenMemcpy -- useful for
  // ensuring that the host pointer isn't getting confused accidentally with a
  // device pointer if you're not doing metaprogramming against the API.
  Stream &ThenMemcpyD2D(DeviceMemoryBase *gpu_dst,
                        const DeviceMemoryBase &gpu_src, uint64_t size) {
    return ThenMemcpy(gpu_dst, gpu_src, size);
  }

  // Entrain onto the stream: a memset of zero at a GPU location of size bytes.
  // The location must not be null.
  Stream &ThenMemZero(DeviceMemoryBase *location, uint64_t size);

  // Entrain onto the stream: a memset of a 32-bit pattern at a GPU location of
  // size bytes, where bytes must be evenly 32-bit sized (i.e. evenly divisible
  // by 4). The location must not be null.
  Stream &ThenMemset32(DeviceMemoryBase *location, uint32_t pattern,
                       uint64_t size);

  // Enqueue a forward operation of the RNN model onto the stream.
  // See DnnSupport::DoRnnForward for more details.
  Stream &ThenRnnForward(const dnn::RnnDescriptor &rnn_desc,
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
                         DeviceMemory<Eigen::half> *output_c_data,
                         bool is_training,
                         ScratchAllocator *reserve_space_allocator,
                         ScratchAllocator *workspace_allocator,
                         dnn::ProfileResult *output_profile_result);

  Stream &ThenRnnForward(const dnn::RnnDescriptor &rnn_desc,
                         const dnn::RnnSequenceTensorDescriptor &input_desc,
                         const DeviceMemory<float> &input_data,
                         const DeviceMemory<int> &seq_lengths_data,
                         const dnn::RnnStateTensorDescriptor &input_h_desc,
                         const DeviceMemory<float> &input_h_data,
                         const dnn::RnnStateTensorDescriptor &input_c_desc,
                         const DeviceMemory<float> &input_c_data,
                         const DeviceMemory<float> &params,
                         const dnn::RnnSequenceTensorDescriptor &output_desc,
                         DeviceMemory<float> *output_data,
                         const dnn::RnnStateTensorDescriptor &output_h_desc,
                         DeviceMemory<float> *output_h_data,
                         const dnn::RnnStateTensorDescriptor &output_c_desc,
                         DeviceMemory<float> *output_c_data, bool is_training,
                         ScratchAllocator *reserve_space_allocator,
                         ScratchAllocator *workspace_allocator,
                         dnn::ProfileResult *output_profile_result);

  Stream &ThenRnnForward(const dnn::RnnDescriptor &rnn_desc,
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
                         dnn::ProfileResult *output_profile_result);

  // Enqueue a backward operation of the RNN model onto the stream.
  // See DnnSupport::DoRnnBackward for more details.
  Stream &ThenRnnBackward(
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
      DeviceMemory<uint8_t> *reserve_space_data,
      ScratchAllocator *workspace_allocator,
      dnn::ProfileResult *output_profile_result);

  Stream &ThenRnnBackward(const dnn::RnnDescriptor &rnn_desc,
                          const dnn::RnnSequenceTensorDescriptor &input_desc,
                          const DeviceMemory<float> &input_data,
                          const DeviceMemory<int> &seq_lengths_data,
                          const dnn::RnnStateTensorDescriptor &input_h_desc,
                          const DeviceMemory<float> &input_h_data,
                          const dnn::RnnStateTensorDescriptor &input_c_desc,
                          const DeviceMemory<float> &input_c_data,
                          const DeviceMemory<float> &params,
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
                          DeviceMemory<uint8_t> *reserve_space_data,
                          ScratchAllocator *workspace_allocator,
                          dnn::ProfileResult *output_profile_result);

  Stream &ThenRnnBackward(const dnn::RnnDescriptor &rnn_desc,
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
                          DeviceMemory<uint8_t> *reserve_space_data,
                          ScratchAllocator *workspace_allocator,
                          dnn::ProfileResult *output_profile_result);

  // Enqueue a CTCLoss operation onto the stream.
  // See DnnSupport::DoCtcLoss for more details.
  Stream &ThenCtcLoss(const dnn::RnnStateTensorDescriptor &probs_desc,
                      const DeviceMemory<float> &probs_data,
                      absl::Span<const int> labels_data,
                      absl::Span<const int> labels_lengths_data,
                      absl::Span<const int> input_lengths_data,
                      DeviceMemory<float> *costs_data,
                      const dnn::RnnStateTensorDescriptor &grads_desc,
                      DeviceMemory<float> *grads_data,
                      ScratchAllocator *workspace_allocator);

  // Enqueue onto the stream a operation that transforms a tensor.
  // See DnnSupport::DoTransformTensor for more details.
  Stream &ThenTransformTensor(const dnn::BatchDescriptor &input_desc,
                              dnn::DataType input_type,
                              const DeviceMemoryBase &input_data,
                              const dnn::BatchDescriptor &output_desc,
                              dnn::DataType output_type, float scale,
                              DeviceMemoryBase *output_data);

  // The templated version of the above ThenTransformTensor. Useful when the
  // input and output types are statically known.
  template <typename InElemT, typename OutElemT>
  Stream &ThenTransformTensor(const dnn::BatchDescriptor &input_desc,
                              const DeviceMemory<InElemT> &input_data,
                              const dnn::BatchDescriptor &output_desc,
                              DeviceMemory<OutElemT> *output_data) {
    return ThenTransformTensor(input_desc, dnn::ToDataType<InElemT>(),
                               input_data, output_desc,
                               dnn::ToDataType<OutElemT>(), output_data);
  }

  // (Synchronously) block the host code waiting for the operations
  // entrained on the stream (enqueued to this point in program
  // execution) to complete.
  //
  // Returns an OK status if the blocking was successful and the stream is ok().
  // Otherwise returns an error describing why the blocking failed.
  tsl::Status BlockHostUntilDone() TF_LOCKS_EXCLUDED(mu_);

  // Warning! This method interacts with internal threads in
  // sometimes-unpredictable ways and is intended for GPU-Executor-internal
  // use
  // only. Please check with a member of the FASTR team before making use of
  // this method.
  //
  // Entrains onto the stream a function to be executed on the host at some
  // point in the future.
  // Async host callbacks DO NOT block the stream as device functions (or as
  // synchronous host callbacks). No synchronization is possible with
  // asynchronous callbacks; they are strictly fire-and-forget.
  // This method is private due to the potential for undefined behavior with
  // synchronization using OpenCL user events.
  // The ONLY lifetime guarantee in these calls is that the StreamExecutor
  // parameter will still be valid - this Stream may not be!
  // Any callbacks requiring device API calls must use this method.
  Stream &ThenEnqueueOnBackgroundThread(
      std::function<void(StreamExecutor *)> task);

  // Returns the (opaque) platform-specific backing object. Ownership is not
  // transferred to the caller.
  internal::StreamInterface *implementation() { return implementation_.get(); }

  // Entrains onto the stream a callback to the host (from the device).
  // Behaves as ThenDoHostCallbackWithStatus below, but the callback should
  // never fail or its failure is inconsequential.
  //
  // This is kept for backward compatibility. Future code should use
  // ThenDoHostCallbackWithStatus and explicitly return a success status.
  // TODO(b/112125301): Eventually remove this method.
  Stream &ThenDoHostCallback(absl::AnyInvocable<void() &&> callback);

  // Entrains onto the stream a callback to the host (from the device).
  // Host callbacks block/occupy the stream just as device functions
  // (execute one at a time, block later stream operations).
  // Whether the callback return status affects the result of BlockHostUntilDone
  // is platform-dependent.
  //
  // Behavior is undefined when synchronizing using OpenCL user events.
  // Behavior is undefined if host callbacks call device routines or insert
  // them into any stream.
  //
  // On certain platforms, ThenDoHostCallback is expected to have significant
  // negative effects on performance.
  Stream &ThenDoHostCallbackWithStatus(
      absl::AnyInvocable<tsl::Status() &&> callback);

  // Runs the given callback after the next call to BlockHostUntilDone on this
  // stream (or after the Stream does BlockHostUntilDone in its destructor).
  // This can act as a faster alternative to ThenDoHostCallbackWithStatus for
  // some use cases.
  Stream &ThenRunAfterNextBlockHostUntilDone(
      absl::AnyInvocable<void() &&> callback);

  // Returns the StreamExecutor (parent object) associated with this stream.
  StreamExecutor *parent() const {
    CHECK(parent_ != nullptr);
    return parent_;
  }

  //
  CudaComputeCapability GetCudaComputeCapability() const {
    return parent()->GetDeviceDescription().cuda_compute_capability();
  }

  RocmComputeCapability GetRocmComputeCapability() const {
    return parent()->GetDeviceDescription().rocm_compute_capability();
  }
  // Returns the (internal usage) temporary-memory-allocation manager associated
  // with this stream.
  internal::TemporaryMemoryManager *temporary_memory_manager();

  // Returns a debugging string "[stream=0x...,impl=0x...]".
  std::string DebugStreamPointers() const;

 private:
  friend class host::HostBlas;  // for parent_.
  friend class host::HostFft;   // for parent_.
  friend class host::HostRng;   // for parent_.
  template <typename... Args>
  friend struct ThenBlasImpl;  // for implementing ThenBlasXXX.
  friend class ocl::CLBlas;    // for parent_.

  // Checks whether types match before a call to extended BLAS version.
  template <typename ABType, typename CType, typename ScaleType>
  tsl::Status CheckTypesForExtendedBlas(
      blas::ComputationType computation_type) {
    static_assert(
        detail::is_any_of<ABType, Eigen::half, Eigen::bfloat16, float, double,
                          int8_t, std::complex<float>, std::complex<double>>(),
        "The only buffer types supported are: Eigen::half, float, "
        "double, int8, std::complex<float> and std::complex<double>");
    static_assert(
        std::is_same_v<ABType, CType> ||
            (std::is_same_v<ABType, int8_t> && std::is_same_v<CType, int32_t>),
        "Input and output buffer types should be the same unless input is "
        "int8 and output is int32");
    static_assert(
        std::is_same_v<ScaleType, CType> ||
            (std::is_same_v<ScaleType, float> &&
             detail::is_any_of<CType, Eigen::half, Eigen::bfloat16>()),
        "Mismatched alpha/beta and output types");

    bool valid_computation_type = [computation_type] {
      switch (computation_type) {
        case blas::ComputationType::kF16:
          return std::is_same_v<CType, Eigen::half>;
        case blas::ComputationType::kF32:
          return detail::is_any_of<CType, Eigen::half, Eigen::bfloat16, float,
                                   std::complex<float>>();
        case blas::ComputationType::kF64:
          return detail::is_any_of<CType, double, std::complex<double>>();
        case blas::ComputationType::kI32:
          return std::is_same_v<CType, int32_t>;
        case blas::ComputationType::kF16AsF32:   // fall-through
        case blas::ComputationType::kBF16AsF32:  // fall-through
        case blas::ComputationType::kTF32AsF32:
          return detail::is_any_of<CType, float, std::complex<float>>();
      }
    }();

    if (!valid_computation_type) {
      return tsl::errors::Internal(
          "Invalid computation type ",
          blas::ComputationTypeString(computation_type), " for output type: ",
          blas::DataTypeString(blas::ToDataType<CType>::value));
    }
    return ::tsl::OkStatus();
  }

  bool InErrorState() const TF_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock lock(&mu_);
    return !status_.ok();
  }

  // Sets the error state if operation_retcode is false.
  // This is a useful shorthand for many stream routines.
  void CheckError(bool operation_retcode) TF_LOCKS_EXCLUDED(mu_);

  // Checks the status and logs the error message, if any.
  void CheckStatus(tsl::Status status) TF_LOCKS_EXCLUDED(mu_);

  void SetError() { CheckError(false /* = operation_retcode */); }

  void SetErrorAndLogNoDnnSupport() {
    SetError();
    LOG(WARNING) << "attempting to perform DNN operation using StreamExecutor "
                    "without DNN support";
  }

  // Runs the set of callbacks that are intended to run after
  // BlockHostUntilDone.
  void RunAfterBlockHostUntilDoneCallbacks();

  // The StreamExecutor that supports the operation of this stream.
  StreamExecutor *parent_;

  // The platform-dependent implementation that the StreamExecutor interface
  // delegates to.
  std::unique_ptr<internal::StreamInterface> implementation_;

  // mutex that guards the allocation / error state flags.
  // Mutable so that it can be obtained via const reader lock.
  mutable absl::Mutex mu_;

  // Whether Init() was successfully called to allocate this stream on the
  // underlying platform. It simply flips from 0 to 1 with a sanity check.
  // See StreamExecutor::AllocateStream.
  bool allocated_ ABSL_GUARDED_BY(mu_);

  // The last error (if any) of all method calls.
  tsl::Status status_ ABSL_GUARDED_BY(mu_);

  // Sub-streams that are generated from this stream. Each element has a pointer
  // to sub-stream and a boolean value indicating if this substream is ready to
  // be reused.
  std::vector<std::pair<std::unique_ptr<Stream>, bool>> sub_streams_
      ABSL_GUARDED_BY(mu_);

  // Streams can allocate temporary memories to help with work they enqueue
  // (e.g. for scratch memory spaces). This member tracks those allocations and
  // notes when they can be reclaimed -- reclamation is attempted when
  // BlockHostUntilDone() is called.
  internal::TemporaryMemoryManager temporary_memory_manager_;

  // Callbacks enqueued to be run after the next call to BlockHostUntilDone().
  std::vector<absl::AnyInvocable<void() &&>>
      after_block_host_until_done_callbacks_ ABSL_GUARDED_BY(mu_);

  // Non-extended BLAS interface requires alpha/beta to be floats when input
  // type is Eigen::half. However, for consistency purposes it is convenient
  // for the interface to accept Eigen::half.
  template <typename T>
  void UpcastHalfToFloat(void **alpha_ptr, void **beta_ptr,
                         float *alpha_storage, float *beta_storage) {
    if (std::is_same<T, Eigen::half>::value) {
      *alpha_storage =
          static_cast<float>(*reinterpret_cast<Eigen::half *>(*alpha_ptr));
      *beta_storage =
          static_cast<float>(*reinterpret_cast<Eigen::half *>(*beta_ptr));
      *alpha_ptr = alpha_storage;
      *beta_ptr = beta_storage;
    } else if (std::is_same<T, Eigen::bfloat16>::value) {
      *alpha_storage =
          static_cast<float>(*reinterpret_cast<Eigen::bfloat16 *>(*alpha_ptr));
      *beta_storage =
          static_cast<float>(*reinterpret_cast<Eigen::bfloat16 *>(*beta_ptr));
      *alpha_ptr = alpha_storage;
      *beta_ptr = beta_storage;
    }
  }

  SE_DISALLOW_COPY_AND_ASSIGN(Stream);
};

////////////
// Inlines

template <typename... Params, typename... Args>
inline tsl::Status Stream::ThenLaunch(ThreadDim thread_dims,
                                      BlockDim block_dims,
                                      const TypedKernel<Params...> &kernel,
                                      Args... args) {
  KernelInvocationChecker<std::tuple<Params...>,
                          std::tuple<Args...>>::CheckAllStaticAssert();

  // This is the core that allows type-safe kernel launching.
  // Since the platforms take kernel arguments as tuples of (void *, size),
  // we pack the variadic parameters passed as ...args into the desired
  // tuple form and pass that packed form to the StreamExecutor::Launch()
  // implementation.
  KernelArgsArray<sizeof...(args)> kernel_args;
  kernel.PackParams(&kernel_args, args...);
  TF_RETURN_IF_ERROR(
      parent_->Launch(this, thread_dims, block_dims, kernel, kernel_args));
  return ::tsl::OkStatus();
}

template <typename T>
inline tsl::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>>
Stream::AllocateTemporaryArray(uint64_t element_count) {
  return temporary_memory_manager_.AllocateArray<T>(element_count);
}

inline internal::TemporaryMemoryManager *Stream::temporary_memory_manager() {
  return &temporary_memory_manager_;
}

template <>
struct Quantization<uint8_t> {
  static constexpr dnn::QuantizedActivationMode kModeId =
      dnn::QuantizedActivationMode::k8Bit;
};

template <>
struct Quantization<uint16_t> {
  static constexpr dnn::QuantizedActivationMode kModeId =
      dnn::QuantizedActivationMode::k16Bit;
};

template <>
struct Quantization<int32_t> {
  static constexpr dnn::QuantizedActivationMode kModeId =
      dnn::QuantizedActivationMode::k32Bit;
};

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_STREAM_H_
