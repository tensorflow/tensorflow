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

#ifndef TENSORFLOW_STREAM_EXECUTOR_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_STREAM_H_

#include <complex>
#include <functional>
#include <memory>

#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/fft.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/launch_dim.h"
#include "tensorflow/stream_executor/lib/array_slice.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/temporary_memory_manager.h"

namespace perftools {
namespace gputools {

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
class BatchDescriptor;
class FilterDescriptor;
class ConvolutionDescriptor;
class ProfileResult;
typedef int64 AlgorithmType;
}  // namespace dnn

class StreamExecutor;
class ScratchAllocator;

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

  // Test only. Use an externally-populated value (like a mock) for the
  // platform-specific stream implementation.
  Stream(StreamExecutor *parent, internal::StreamInterface *implementation);

  // Deallocates any stream resources that the parent StreamExecutor has
  // bestowed
  // upon this object.
  ~Stream();

  // Returns whether any errors have occurred while entraining work for this
  // stream.
  bool ok() const { return !InErrorState(); }

  // Initialize the stream. This must be performed before entraining any other
  // operations.
  Stream &Init();

  // Initializes timer t via the StreamExecutor.
  Stream &InitTimer(Timer *t);

  // Convenience wrapper around Init() and InitTimer().
  Stream &InitWithTimer(Timer *t);

  // Warning! After calling BlockHostUntilDone(), all sub-streams will be
  // returned and hence invalid. This may be a temporary solution to the issue
  // b/18070215.
  // Get or create a sub-stream from this stream. If there is any sub-stream
  // in the pool that can be reused then just return this sub-stream.
  // Otherwise
  // create a new sub-stream.
  Stream *GetOrCreateSubStream();

  // Return the sub-stream back to the host stream so that it can be reused
  // later.
  void ReturnSubStream(Stream *sub_stream);

  // Allocate temporary memories. The stream will deallocate them when blocked
  // or destroyed.
  template <typename T>
  port::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>>
  AllocateTemporaryArray(uint64 element_count);

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
  Stream &ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                     const TypedKernel<Params...> &kernel, Args... args);

  // Record a "start" event for the interval timer at this point in the
  // stream's
  // execution (relative to the previously and subsequently enqueued items in
  // the stream's execution). Streams may be started/stopped multiple times.
  Stream &ThenStartTimer(Timer *t);

  // Record a "stop" event for the interval timer at this point in the
  // stream's
  // execution. See also Stream::ThenStartTimer.
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
  // cyclic
  // manner; in that case, behavior is undefined.
  //
  // N.B. Base recursion case for the variadic ThenWaitFor.
  Stream &ThenWaitFor(Stream *other);

  // Waits for all streams values in others.
  // Checks that there is no shallow circular wait (i.e. that "this" is not in
  // others).
  Stream &ThenWaitFor(std::vector<std::unique_ptr<Stream>> *others);

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

  // TODO(leary) add double-precision version of this interface.
  Stream &ThenConvolve(const dnn::BatchDescriptor &input_descriptor,
                       const DeviceMemory<float> &input_data,
                       const dnn::FilterDescriptor &filter_descriptor,
                       const DeviceMemory<float> &filter_data,
                       const dnn::ConvolutionDescriptor &convolution_descriptor,
                       const dnn::BatchDescriptor &output_descriptor,
                       DeviceMemory<float> *output);

  Stream &ThenConvolveWithScratch(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<Eigen::half> &input_data,
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<Eigen::half> &filter_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<Eigen::half> *output, ScratchAllocator *scratch_allocator);

  Stream &ThenConvolveWithScratch(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<float> &input_data,
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<float> &filter_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> *output, ScratchAllocator *scratch_allocator);

  Stream &ThenConvolveWithAlgorithm(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<float> &input_data,
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<float> &filter_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> *output, ScratchAllocator *scratch_allocator,
      const dnn::AlgorithmConfig &algorithm_config,
      dnn::ProfileResult *output_profile_result);

  Stream &ThenConvolveWithAlgorithm(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<Eigen::half> &input_data,
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<Eigen::half> &filter_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<Eigen::half> *output, ScratchAllocator *scratch_allocator,
      const dnn::AlgorithmConfig &algorithm_config,
      dnn::ProfileResult *output_profile_result);

  Stream &ThenSeparableConvolve(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<float> &input_data,
      const dnn::FilterDescriptor &filter_descriptor, int depth_multiplier,
      const DeviceMemory<float> &first_weights,
      const DeviceMemory<float> &second_weights,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> *output);

  Stream &ThenConvolveBackwardData(
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<float> &filter_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &input_descriptor,
      DeviceMemory<float> *backward_input_data);

  Stream &ThenConvolveBackwardDataWithScratch(
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<float> &filter_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &input_descriptor,
      DeviceMemory<float> *backward_input_data,
      ScratchAllocator *scratch_allocator);

  Stream &ThenConvolveBackwardDataWithScratch(
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<Eigen::half> &filter_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<Eigen::half> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &input_descriptor,
      DeviceMemory<Eigen::half> *backward_input_data,
      ScratchAllocator *scratch_allocator);

  Stream &ThenConvolveBackwardDataWithAlgorithm(
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<float> &filter_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &input_descriptor,
      DeviceMemory<float> *backward_input_data,
      ScratchAllocator *scratch_allocator,
      const dnn::AlgorithmConfig &algorithm_config,
      dnn::ProfileResult *output_profile_result);

  Stream &ThenConvolveBackwardDataWithAlgorithm(
      const dnn::FilterDescriptor &filter_descriptor,
      const DeviceMemory<Eigen::half> &filter_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<Eigen::half> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::BatchDescriptor &input_descriptor,
      DeviceMemory<Eigen::half> *backward_input_data,
      ScratchAllocator *scratch_allocator,
      const dnn::AlgorithmConfig &algorithm_config,
      dnn::ProfileResult *output_profile_result);

  Stream &ThenConvolveBackwardFilter(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<float> &input_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::FilterDescriptor &filter_descriptor,
      DeviceMemory<float> *backward_filter_data);

  Stream &ThenConvolveBackwardFilterWithScratch(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<float> &input_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::FilterDescriptor &filter_descriptor,
      DeviceMemory<float> *backward_filter_data,
      ScratchAllocator *scratch_allocator);

  Stream &ThenConvolveBackwardFilterWithScratch(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<Eigen::half> &input_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<Eigen::half> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::FilterDescriptor &filter_descriptor,
      DeviceMemory<Eigen::half> *backward_filter_data,
      ScratchAllocator *scratch_allocator);

  Stream &ThenConvolveBackwardFilterWithAlgorithm(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<float> &input_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::FilterDescriptor &filter_descriptor,
      DeviceMemory<float> *backward_filter_data,
      ScratchAllocator *scratch_allocator,
      const dnn::AlgorithmConfig &algorithm_config,
      dnn::ProfileResult *output_profile_result);

  Stream &ThenConvolveBackwardFilterWithAlgorithm(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<Eigen::half> &input_data,
      const dnn::BatchDescriptor &output_descriptor,
      DeviceMemory<Eigen::half> backward_output_data,
      const dnn::ConvolutionDescriptor &convolution_descriptor,
      const dnn::FilterDescriptor &filter_descriptor,
      DeviceMemory<Eigen::half> *backward_filter_data,
      ScratchAllocator *scratch_allocator,
      const dnn::AlgorithmConfig &algorithm_config,
      dnn::ProfileResult *output_profile_result);

  Stream &ThenConvolveBackwardBias(const dnn::BatchDescriptor &input_descriptor,
                                   const DeviceMemory<double> &input_data,
                                   const dnn::BatchDescriptor &bias_descriptor,
                                   DeviceMemory<double> *backward_bias_data);

  Stream &ThenConvolveBackwardBias(const dnn::BatchDescriptor &input_descriptor,
                                   const DeviceMemory<float> &input_data,
                                   const dnn::BatchDescriptor &bias_descriptor,
                                   DeviceMemory<float> *backward_bias_data);

  Stream &ThenConvolveBackwardBias(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<Eigen::half> &input_data,
      const dnn::BatchDescriptor &bias_descriptor,
      DeviceMemory<Eigen::half> *backward_bias_data);

  Stream &ThenMatMul(const DeviceMemory<float> &input_data,
                     const DeviceMemory<float> &weights,
                     const dnn::BatchDescriptor &input_dimensions,
                     const dnn::BatchDescriptor &output_dimensions,
                     DeviceMemory<float> *output_data);

  Stream &ThenMatMulQuantized(const DeviceMemory<float> &input_data,
                              const DeviceMemory<int8> &weights,
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

  Stream &ThenPoolForward(const dnn::PoolingDescriptor &pooling_dimensions,
                          const dnn::BatchDescriptor &input_dimensions,
                          const DeviceMemory<float> &input_data,
                          const dnn::BatchDescriptor &output_dimensions,
                          DeviceMemory<float> *output_data);

  Stream &ThenPoolForward(const dnn::PoolingDescriptor &pooling_dimensions,
                          const dnn::BatchDescriptor &input_dimensions,
                          const DeviceMemory<Eigen::half> &input_data,
                          const dnn::BatchDescriptor &output_dimensions,
                          DeviceMemory<Eigen::half> *output_data);

  Stream &ThenPoolBackward(const dnn::PoolingDescriptor &pooling_dimensions,
                           const dnn::BatchDescriptor &input_dimensions,
                           const DeviceMemory<float> &input_data,
                           const dnn::BatchDescriptor &output_dimensions,
                           const DeviceMemory<float> &output_data,
                           const DeviceMemory<float> &input_diff_data,
                           DeviceMemory<float> *output_diff_data);

  Stream &ThenPoolBackward(const dnn::PoolingDescriptor &pooling_dimensions,
                           const dnn::BatchDescriptor &input_dimensions,
                           const DeviceMemory<Eigen::half> &input_data,
                           const dnn::BatchDescriptor &output_dimensions,
                           const DeviceMemory<Eigen::half> &output_data,
                           const DeviceMemory<Eigen::half> &input_diff_data,
                           DeviceMemory<Eigen::half> *output_diff_data);

  Stream &ThenNormalize(const dnn::NormalizeDescriptor &normalize_descriptor,
                        const DeviceMemory<float> &input_data,
                        DeviceMemory<float> *output_data);

  // Similar to ThenNormalize, but normalizes across feature maps and allows for
  // specifying the dimensions of the tensor.
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
      DeviceMemory<float> *raw_variable_gradient);

  Stream &ThenActivate(dnn::ActivationMode activation_mode,
                       const dnn::BatchDescriptor &dimensions,
                       const DeviceMemory<float> &input_data,
                       DeviceMemory<float> *output_data);

  Stream &ThenDepthConcatenate(
      port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
      port::ArraySlice<const DeviceMemory<float> *> input_data,
      DeviceMemory<float> *output_data);

  Stream &ThenElementwiseOperate(
      dnn::ElementwiseOperation operation,
      port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
      port::ArraySlice<const DeviceMemory<float> *> input_data,
      const dnn::BatchDescriptor &output_dimensions,
      DeviceMemory<float> *output_data);

  Stream &ThenXYPad(const dnn::BatchDescriptor &dimensions,
                    const DeviceMemory<float> &input_data, int64 left_pad,
                    int64 right_pad, int64 top_pad, int64 bottom_pad,
                    DeviceMemory<float> *output_data);

  Stream &ThenXYSlice(const dnn::BatchDescriptor &dimensions,
                      const DeviceMemory<float> &input_data, int64 left_trim,
                      int64 right_trim, int64 top_trim, int64 bottom_trim,
                      DeviceMemory<float> *output_data);

  // See DnnSupport::DoMemcpyD2HQuantized.
  Stream &ThenMemcpyD2HQuantized(const DeviceMemory<float> &gpu_unquantized_src,
                                 dnn::QuantizedActivationMode mode,
                                 void *host_dst, uint64 size);

  // Template version of ThenMemcpyD2HQuantized that takes a MutableArraySlice
  // and uses the Quantization trait to call the generic version of
  // ThenMemcpyD2HQuantized with the correct QuantizedActivationMode.
  template <typename ElementType>
  Stream &ThenMemcpyD2HQuantized(
      const DeviceMemory<float> &gpu_unquantized_src,
      port::MutableArraySlice<ElementType> host_dst) {
    return ThenMemcpyD2HQuantized(
        gpu_unquantized_src, Quantization<ElementType>::kModeId,
        host_dst.data(), host_dst.size() * sizeof(ElementType));
  }

  // See DnnSupport::DoMemcpyH2DQuantized.
  Stream &ThenMemcpyH2DQuantized(const void *host_src, uint64 size,
                                 dnn::QuantizedActivationMode mode,
                                 DeviceMemory<float> *gpu_unquantized_dst);

  // Template version of ThenMemcpyH2DQuantized that takes an ArraySlice
  // and uses the Quantization trait to call the generic version of
  // ThenMemcpyH2DQuantized with the correct QuantizedActivationMode.
  template <typename ElementType>
  Stream &ThenMemcpyH2DQuantized(port::ArraySlice<ElementType> host_src,
                                 DeviceMemory<float> *gpu_unquantized_dst) {
    return ThenMemcpyH2DQuantized(
        host_src.data(), host_src.size() * sizeof(ElementType),
        Quantization<ElementType>::kModeId, gpu_unquantized_dst);
  }

  /////////////////
  // BLAS support

  // See BlasSupport::DoBlasAsum.
  Stream &ThenBlasAsum(uint64 elem_count, const DeviceMemory<float> &x,
                       int incx, DeviceMemory<float> *result);
  Stream &ThenBlasAsum(uint64 elem_count, const DeviceMemory<double> &x,
                       int incx, DeviceMemory<double> *result);
  Stream &ThenBlasAsum(uint64 elem_count,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       DeviceMemory<float> *result);
  Stream &ThenBlasAsum(uint64 elem_count,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       DeviceMemory<double> *result);

  // See BlasSupport::DoBlasAxpy. Note that, even for the case where alpha is
  // present in DeviceMemory, it must be an execution-time constant (i.e. a
  // value
  // that the stream does not change or populate during the course of
  // execution). The value is effectively captured at stream-enqueue time.
  Stream &ThenBlasAxpy(uint64 elem_count, float alpha,
                       const DeviceMemory<float> &x, int incx,
                       DeviceMemory<float> *y, int incy);
  Stream &ThenBlasAxpy(uint64 elem_count, double alpha,
                       const DeviceMemory<double> &x, int incx,
                       DeviceMemory<double> *y, int incy);
  Stream &ThenBlasAxpy(uint64 elem_count, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasAxpy(uint64 elem_count, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       DeviceMemory<std::complex<double>> *y, int incy);

  // See BlasSupport::DoBlasCopy.
  Stream &ThenBlasCopy(uint64 elem_count, const DeviceMemory<float> &x,
                       int incx, DeviceMemory<float> *y, int incy);
  Stream &ThenBlasCopy(uint64 elem_count, const DeviceMemory<double> &x,
                       int incx, DeviceMemory<double> *y, int incy);
  Stream &ThenBlasCopy(uint64 elem_count,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasCopy(uint64 elem_count,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       DeviceMemory<std::complex<double>> *y, int incy);

  // See BlasSupport::DoBlasDot.
  Stream &ThenBlasDot(uint64 elem_count, const DeviceMemory<float> &x, int incx,
                      const DeviceMemory<float> &y, int incy,
                      DeviceMemory<float> *result);
  Stream &ThenBlasDot(uint64 elem_count, const DeviceMemory<double> &x,
                      int incx, const DeviceMemory<double> &y, int incy,
                      DeviceMemory<double> *result);

  // See BlasSupport::DoBlasDotc.
  Stream &ThenBlasDotc(uint64 elem_count,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       const DeviceMemory<std::complex<float>> &y, int incy,
                       DeviceMemory<std::complex<float>> *result);
  Stream &ThenBlasDotc(uint64 elem_count,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       const DeviceMemory<std::complex<double>> &y, int incy,
                       DeviceMemory<std::complex<double>> *result);

  // See BlasSupport::DoBlasDotu.
  Stream &ThenBlasDotu(uint64 elem_count,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       const DeviceMemory<std::complex<float>> &y, int incy,
                       DeviceMemory<std::complex<float>> *result);
  Stream &ThenBlasDotu(uint64 elem_count,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       const DeviceMemory<std::complex<double>> &y, int incy,
                       DeviceMemory<std::complex<double>> *result);

  // See BlasSupport::DoBlasNrm2.
  Stream &ThenBlasNrm2(uint64 elem_count, const DeviceMemory<float> &x,
                       int incx, DeviceMemory<float> *result);
  Stream &ThenBlasNrm2(uint64 elem_count, const DeviceMemory<double> &x,
                       int incx, DeviceMemory<double> *result);
  Stream &ThenBlasNrm2(uint64 elem_count,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       DeviceMemory<float> *result);
  Stream &ThenBlasNrm2(uint64 elem_count,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       DeviceMemory<double> *result);

  // See BlasSupport::DoBlasRot.
  Stream &ThenBlasRot(uint64 elem_count, DeviceMemory<float> *x, int incx,
                      DeviceMemory<float> *y, int incy, float c, float s);
  Stream &ThenBlasRot(uint64 elem_count, DeviceMemory<double> *x, int incx,
                      DeviceMemory<double> *y, int incy, double c, double s);
  Stream &ThenBlasRot(uint64 elem_count, DeviceMemory<std::complex<float>> *x,
                      int incx, DeviceMemory<std::complex<float>> *y, int incy,
                      float c, float s);
  Stream &ThenBlasRot(uint64 elem_count, DeviceMemory<std::complex<double>> *x,
                      int incx, DeviceMemory<std::complex<double>> *y, int incy,
                      double c, double s);

  // See BlasSupport::DoBlasRotg.
  Stream &ThenBlasRotg(DeviceMemory<float> *a, DeviceMemory<float> *b,
                       DeviceMemory<float> *c, DeviceMemory<float> *s);
  Stream &ThenBlasRotg(DeviceMemory<double> *a, DeviceMemory<double> *b,
                       DeviceMemory<double> *c, DeviceMemory<double> *s);
  Stream &ThenBlasRotg(DeviceMemory<std::complex<float>> *a,
                       DeviceMemory<std::complex<float>> *b,
                       DeviceMemory<float> *c,
                       DeviceMemory<std::complex<float>> *s);
  Stream &ThenBlasRotg(DeviceMemory<std::complex<double>> *a,
                       DeviceMemory<std::complex<double>> *b,
                       DeviceMemory<double> *c,
                       DeviceMemory<std::complex<double>> *s);

  // See BlasSupport::DoBlasRotm.
  Stream &ThenBlasRotm(uint64 elem_count, DeviceMemory<float> *x, int incx,
                       DeviceMemory<float> *y, int incy,
                       const DeviceMemory<float> &param);
  Stream &ThenBlasRotm(uint64 elem_count, DeviceMemory<double> *x, int incx,
                       DeviceMemory<double> *y, int incy,
                       const DeviceMemory<double> &param);

  // See BlasSupport::DoBlasRotmg.
  Stream &ThenBlasRotmg(DeviceMemory<float> *d1, DeviceMemory<float> *d2,
                        DeviceMemory<float> *x1, const DeviceMemory<float> &y1,
                        DeviceMemory<float> *param);
  Stream &ThenBlasRotmg(DeviceMemory<double> *d1, DeviceMemory<double> *d2,
                        DeviceMemory<double> *x1,
                        const DeviceMemory<double> &y1,
                        DeviceMemory<double> *param);

  // See BlasSupport::DoBlasScal.
  Stream &ThenBlasScal(uint64 elem_count, float alpha, DeviceMemory<float> *x,
                       int incx);
  Stream &ThenBlasScal(uint64 elem_count, double alpha, DeviceMemory<double> *x,
                       int incx);
  Stream &ThenBlasScal(uint64 elem_count, float alpha,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasScal(uint64 elem_count, double alpha,
                       DeviceMemory<std::complex<double>> *x, int incx);
  Stream &ThenBlasScal(uint64 elem_count, std::complex<float> alpha,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasScal(uint64 elem_count, std::complex<double> alpha,
                       DeviceMemory<std::complex<double>> *x, int incx);

  // See BlasSupport::DoBlasSwap.
  Stream &ThenBlasSwap(uint64 elem_count, DeviceMemory<float> *x, int incx,
                       DeviceMemory<float> *y, int incy);
  Stream &ThenBlasSwap(uint64 elem_count, DeviceMemory<double> *x, int incx,
                       DeviceMemory<double> *y, int incy);
  Stream &ThenBlasSwap(uint64 elem_count, DeviceMemory<std::complex<float>> *x,
                       int incx, DeviceMemory<std::complex<float>> *y,
                       int incy);
  Stream &ThenBlasSwap(uint64 elem_count, DeviceMemory<std::complex<double>> *x,
                       int incx, DeviceMemory<std::complex<double>> *y,
                       int incy);

  // See BlasSupport::DoBlasIamax.
  Stream &ThenBlasIamax(uint64 elem_count, const DeviceMemory<float> &x,
                        int incx, DeviceMemory<int> *result);
  Stream &ThenBlasIamax(uint64 elem_count, const DeviceMemory<double> &x,
                        int incx, DeviceMemory<int> *result);
  Stream &ThenBlasIamax(uint64 elem_count,
                        const DeviceMemory<std::complex<float>> &x, int incx,
                        DeviceMemory<int> *result);
  Stream &ThenBlasIamax(uint64 elem_count,
                        const DeviceMemory<std::complex<double>> &x, int incx,
                        DeviceMemory<int> *result);

  // See BlasSupport::DoBlasIamin.
  Stream &ThenBlasIamin(uint64 elem_count, const DeviceMemory<float> &x,
                        int incx, DeviceMemory<int> *result);
  Stream &ThenBlasIamin(uint64 elem_count, const DeviceMemory<double> &x,
                        int incx, DeviceMemory<int> *result);
  Stream &ThenBlasIamin(uint64 elem_count,
                        const DeviceMemory<std::complex<float>> &x, int incx,
                        DeviceMemory<int> *result);
  Stream &ThenBlasIamin(uint64 elem_count,
                        const DeviceMemory<std::complex<double>> &x, int incx,
                        DeviceMemory<int> *result);

  // See BlasSupport::DoBlasGbmv.
  Stream &ThenBlasGbmv(blas::Transpose trans, uint64 m, uint64 n, uint64 kl,
                       uint64 ku, float alpha, const DeviceMemory<float> &a,
                       int lda, const DeviceMemory<float> &x, int incx,
                       float beta, DeviceMemory<float> *y, int incy);
  Stream &ThenBlasGbmv(blas::Transpose trans, uint64 m, uint64 n, uint64 kl,
                       uint64 ku, double alpha, const DeviceMemory<double> &a,
                       int lda, const DeviceMemory<double> &x, int incx,
                       double beta, DeviceMemory<double> *y, int incy);
  Stream &ThenBlasGbmv(blas::Transpose trans, uint64 m, uint64 n, uint64 kl,
                       uint64 ku, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasGbmv(blas::Transpose trans, uint64 m, uint64 n, uint64 kl,
                       uint64 ku, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *y, int incy);

  // See BlasSupport::DoBlasGemv.
  Stream &ThenBlasGemv(blas::Transpose trans, uint64 m, uint64 n, float alpha,
                       const DeviceMemory<float> &a, int lda,
                       const DeviceMemory<float> &x, int incx, float beta,
                       DeviceMemory<float> *y, int incy);
  Stream &ThenBlasGemv(blas::Transpose trans, uint64 m, uint64 n, double alpha,
                       const DeviceMemory<double> &a, int lda,
                       const DeviceMemory<double> &x, int incx, double beta,
                       DeviceMemory<double> *y, int incy);
  Stream &ThenBlasGemv(blas::Transpose trans, uint64 m, uint64 n,
                       std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasGemv(blas::Transpose trans, uint64 m, uint64 n,
                       std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *y, int incy);

  // See BlasSupport::DoBlasGer.
  Stream &ThenBlasGer(uint64 m, uint64 n, float alpha,
                      const DeviceMemory<float> &x, int incx,
                      const DeviceMemory<float> &y, int incy,
                      DeviceMemory<float> *a, int lda);
  Stream &ThenBlasGer(uint64 m, uint64 n, double alpha,
                      const DeviceMemory<double> &x, int incx,
                      const DeviceMemory<double> &y, int incy,
                      DeviceMemory<double> *a, int lda);

  // See BlasSupport::DoBlasGerc.
  Stream &ThenBlasGerc(uint64 m, uint64 n, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       const DeviceMemory<std::complex<float>> &y, int incy,
                       DeviceMemory<std::complex<float>> *a, int lda);
  Stream &ThenBlasGerc(uint64 m, uint64 n, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       const DeviceMemory<std::complex<double>> &y, int incy,
                       DeviceMemory<std::complex<double>> *a, int lda);

  // See BlasSupport::DoBlasGeru.
  Stream &ThenBlasGeru(uint64 m, uint64 n, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       const DeviceMemory<std::complex<float>> &y, int incy,
                       DeviceMemory<std::complex<float>> *a, int lda);
  Stream &ThenBlasGeru(uint64 m, uint64 n, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       const DeviceMemory<std::complex<double>> &y, int incy,
                       DeviceMemory<std::complex<double>> *a, int lda);

  // See BlasSupport::DoBlasHbmv.
  Stream &ThenBlasHbmv(blas::UpperLower uplo, uint64 n, uint64 k,
                       std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasHbmv(blas::UpperLower uplo, uint64 n, uint64 k,
                       std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *y, int incy);

  // See BlasSupport::DoBlasHemv.
  Stream &ThenBlasHemv(blas::UpperLower uplo, uint64 n,
                       std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasHemv(blas::UpperLower uplo, uint64 n,
                       std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *y, int incy);

  // See BlasSupport::DoBlasHer.
  Stream &ThenBlasHer(blas::UpperLower uplo, uint64 n, float alpha,
                      const DeviceMemory<std::complex<float>> &x, int incx,
                      DeviceMemory<std::complex<float>> *a, int lda);
  Stream &ThenBlasHer(blas::UpperLower uplo, uint64 n, double alpha,
                      const DeviceMemory<std::complex<double>> &x, int incx,
                      DeviceMemory<std::complex<double>> *a, int lda);

  // See BlasSupport::DoBlasHer2.
  Stream &ThenBlasHer2(blas::UpperLower uplo, uint64 n,
                       std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       const DeviceMemory<std::complex<float>> &y, int incy,
                       DeviceMemory<std::complex<float>> *a, int lda);
  Stream &ThenBlasHer2(blas::UpperLower uplo, uint64 n,
                       std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       const DeviceMemory<std::complex<double>> &y, int incy,
                       DeviceMemory<std::complex<double>> *a, int lda);

  // See BlasSupport::DoBlasHpmv.
  Stream &ThenBlasHpmv(blas::UpperLower uplo, uint64 n,
                       std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &ap,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *y, int incy);
  Stream &ThenBlasHpmv(blas::UpperLower uplo, uint64 n,
                       std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &ap,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *y, int incy);

  // See BlasSupport::DoBlasHpr.
  Stream &ThenBlasHpr(blas::UpperLower uplo, uint64 n, float alpha,
                      const DeviceMemory<std::complex<float>> &x, int incx,
                      DeviceMemory<std::complex<float>> *ap);
  Stream &ThenBlasHpr(blas::UpperLower uplo, uint64 n, double alpha,
                      const DeviceMemory<std::complex<double>> &x, int incx,
                      DeviceMemory<std::complex<double>> *ap);

  // See BlasSupport::DoBlasHpr2.
  Stream &ThenBlasHpr2(blas::UpperLower uplo, uint64 n,
                       std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &x, int incx,
                       const DeviceMemory<std::complex<float>> &y, int incy,
                       DeviceMemory<std::complex<float>> *ap);
  Stream &ThenBlasHpr2(blas::UpperLower uplo, uint64 n,
                       std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &x, int incx,
                       const DeviceMemory<std::complex<double>> &y, int incy,
                       DeviceMemory<std::complex<double>> *ap);

  // See BlasSupport::DoBlasSbmv.
  Stream &ThenBlasSbmv(blas::UpperLower uplo, uint64 n, uint64 k, float alpha,
                       const DeviceMemory<float> &a, int lda,
                       const DeviceMemory<float> &x, int incx, float beta,
                       DeviceMemory<float> *y, int incy);
  Stream &ThenBlasSbmv(blas::UpperLower uplo, uint64 n, uint64 k, double alpha,
                       const DeviceMemory<double> &a, int lda,
                       const DeviceMemory<double> &x, int incx, double beta,
                       DeviceMemory<double> *y, int incy);

  // See BlasSupport::DoBlasSpmv.
  Stream &ThenBlasSpmv(blas::UpperLower uplo, uint64 n, float alpha,
                       const DeviceMemory<float> &ap,
                       const DeviceMemory<float> &x, int incx, float beta,
                       DeviceMemory<float> *y, int incy);
  Stream &ThenBlasSpmv(blas::UpperLower uplo, uint64 n, double alpha,
                       const DeviceMemory<double> &ap,
                       const DeviceMemory<double> &x, int incx, double beta,
                       DeviceMemory<double> *y, int incy);

  // See BlasSupport::DoBlasSpr.
  Stream &ThenBlasSpr(blas::UpperLower uplo, uint64 n, float alpha,
                      const DeviceMemory<float> &x, int incx,
                      DeviceMemory<float> *ap);
  Stream &ThenBlasSpr(blas::UpperLower uplo, uint64 n, double alpha,
                      const DeviceMemory<double> &x, int incx,
                      DeviceMemory<double> *ap);

  // See BlasSupport::DoBlasSpr2.
  Stream &ThenBlasSpr2(blas::UpperLower uplo, uint64 n, float alpha,
                       const DeviceMemory<float> &x, int incx,
                       const DeviceMemory<float> &y, int incy,
                       DeviceMemory<float> *ap);
  Stream &ThenBlasSpr2(blas::UpperLower uplo, uint64 n, double alpha,
                       const DeviceMemory<double> &x, int incx,
                       const DeviceMemory<double> &y, int incy,
                       DeviceMemory<double> *ap);

  // See BlasSupport::DoBlasSymv.
  Stream &ThenBlasSymv(blas::UpperLower uplo, uint64 n, float alpha,
                       const DeviceMemory<float> &a, int lda,
                       const DeviceMemory<float> &x, int incx, float beta,
                       DeviceMemory<float> *y, int incy);
  Stream &ThenBlasSymv(blas::UpperLower uplo, uint64 n, double alpha,
                       const DeviceMemory<double> &a, int lda,
                       const DeviceMemory<double> &x, int incx, double beta,
                       DeviceMemory<double> *y, int incy);

  // See BlasSupport::DoBlasSyr.
  Stream &ThenBlasSyr(blas::UpperLower uplo, uint64 n, float alpha,
                      const DeviceMemory<float> &x, int incx,
                      DeviceMemory<float> *a, int lda);
  Stream &ThenBlasSyr(blas::UpperLower uplo, uint64 n, double alpha,
                      const DeviceMemory<double> &x, int incx,
                      DeviceMemory<double> *a, int lda);

  // See BlasSupport::DoBlasSyr2.
  Stream &ThenBlasSyr2(blas::UpperLower uplo, uint64 n, float alpha,
                       const DeviceMemory<float> &x, int incx,
                       const DeviceMemory<float> &y, int incy,
                       DeviceMemory<float> *a, int lda);
  Stream &ThenBlasSyr2(blas::UpperLower uplo, uint64 n, double alpha,
                       const DeviceMemory<double> &x, int incx,
                       const DeviceMemory<double> &y, int incy,
                       DeviceMemory<double> *a, int lda);

  // See BlasSupport::DoBlasTbmv.
  Stream &ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n, uint64 k,
                       const DeviceMemory<float> &a, int lda,
                       DeviceMemory<float> *x, int incx);
  Stream &ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n, uint64 k,
                       const DeviceMemory<double> &a, int lda,
                       DeviceMemory<double> *x, int incx);
  Stream &ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n, uint64 k,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasTbmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n, uint64 k,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       DeviceMemory<std::complex<double>> *x, int incx);

  // See BlasSupport::DoBlasTbsv.
  Stream &ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n, uint64 k,
                       const DeviceMemory<float> &a, int lda,
                       DeviceMemory<float> *x, int incx);
  Stream &ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n, uint64 k,
                       const DeviceMemory<double> &a, int lda,
                       DeviceMemory<double> *x, int incx);
  Stream &ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n, uint64 k,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasTbsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n, uint64 k,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       DeviceMemory<std::complex<double>> *x, int incx);

  // See BlasSupport::DoBlasTpmv.
  Stream &ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                       int incx);
  Stream &ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<double> &ap, DeviceMemory<double> *x,
                       int incx);
  Stream &ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<std::complex<float>> &ap,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasTpmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<std::complex<double>> &ap,
                       DeviceMemory<std::complex<double>> *x, int incx);

  // See BlasSupport::DoBlasTpsv.
  Stream &ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                       int incx);
  Stream &ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<double> &ap, DeviceMemory<double> *x,
                       int incx);
  Stream &ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<std::complex<float>> &ap,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasTpsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<std::complex<double>> &ap,
                       DeviceMemory<std::complex<double>> *x, int incx);

  // See BlasSupport::DoBlasTrmv.
  Stream &ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<float> &a, int lda,
                       DeviceMemory<float> *x, int incx);
  Stream &ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<double> &a, int lda,
                       DeviceMemory<double> *x, int incx);
  Stream &ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasTrmv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       DeviceMemory<std::complex<double>> *x, int incx);

  // See BlasSupport::DoBlasTrsv.
  Stream &ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<float> &a, int lda,
                       DeviceMemory<float> *x, int incx);
  Stream &ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<double> &a, int lda,
                       DeviceMemory<double> *x, int incx);
  Stream &ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       DeviceMemory<std::complex<float>> *x, int incx);
  Stream &ThenBlasTrsv(blas::UpperLower uplo, blas::Transpose trans,
                       blas::Diagonal diag, uint64 n,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       DeviceMemory<std::complex<double>> *x, int incx);

  // See BlasSupport::DoBlasGemm.
  Stream &ThenBlasGemm(blas::Transpose transa, blas::Transpose transb, uint64 m,
                       uint64 n, uint64 k, float alpha,
                       const DeviceMemory<Eigen::half> &a, int lda,
                       const DeviceMemory<Eigen::half> &b, int ldb, float beta,
                       DeviceMemory<Eigen::half> *c, int ldc);
  Stream &ThenBlasGemm(blas::Transpose transa, blas::Transpose transb, uint64 m,
                       uint64 n, uint64 k, float alpha,
                       const DeviceMemory<float> &a, int lda,
                       const DeviceMemory<float> &b, int ldb, float beta,
                       DeviceMemory<float> *c, int ldc);
  Stream &ThenBlasGemm(blas::Transpose transa, blas::Transpose transb, uint64 m,
                       uint64 n, uint64 k, double alpha,
                       const DeviceMemory<double> &a, int lda,
                       const DeviceMemory<double> &b, int ldb, double beta,
                       DeviceMemory<double> *c, int ldc);
  Stream &ThenBlasGemm(blas::Transpose transa, blas::Transpose transb, uint64 m,
                       uint64 n, uint64 k, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       const DeviceMemory<std::complex<float>> &b, int ldb,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *c, int ldc);
  Stream &ThenBlasGemm(blas::Transpose transa, blas::Transpose transb, uint64 m,
                       uint64 n, uint64 k, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       const DeviceMemory<std::complex<double>> &b, int ldb,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *c, int ldc);

  // See BlasSupport::DoBlasGemmBatched.
  Stream &ThenBlasGemmBatched(blas::Transpose transa, blas::Transpose transb,
                              uint64 m, uint64 n, uint64 k, float alpha,
                              const port::ArraySlice<DeviceMemory<float> *> &a,
                              int lda,
                              const port::ArraySlice<DeviceMemory<float> *> &b,
                              int ldb, float beta,
                              const port::ArraySlice<DeviceMemory<float> *> &c,
                              int ldc, int batch_count);
  Stream &ThenBlasGemmBatched(blas::Transpose transa, blas::Transpose transb,
                              uint64 m, uint64 n, uint64 k, double alpha,
                              const port::ArraySlice<DeviceMemory<double> *> &a,
                              int lda,
                              const port::ArraySlice<DeviceMemory<double> *> &b,
                              int ldb, double beta,
                              const port::ArraySlice<DeviceMemory<double> *> &c,
                              int ldc, int batch_count);
  Stream &ThenBlasGemmBatched(
      blas::Transpose transa, blas::Transpose transb, uint64 m, uint64 n,
      uint64 k, std::complex<float> alpha,
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a, int lda,
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b, int ldb,
      std::complex<float> beta,
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c, int ldc,
      int batch_count);
  Stream &ThenBlasGemmBatched(
      blas::Transpose transa, blas::Transpose transb, uint64 m, uint64 n,
      uint64 k, std::complex<double> alpha,
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a, int lda,
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b, int ldb,
      std::complex<double> beta,
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c, int ldc,
      int batch_count);
  Stream &ThenBlasGemmBatchedWithScratch(
      blas::Transpose transa, blas::Transpose transb, uint64 m, uint64 n,
      uint64 k, float alpha, const port::ArraySlice<DeviceMemory<float> *> &a,
      int lda, const port::ArraySlice<DeviceMemory<float> *> &b, int ldb,
      float beta, const port::ArraySlice<DeviceMemory<float> *> &c, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator);
  Stream &ThenBlasGemmBatchedWithScratch(
      blas::Transpose transa, blas::Transpose transb, uint64 m, uint64 n,
      uint64 k, double alpha, const port::ArraySlice<DeviceMemory<double> *> &a,
      int lda, const port::ArraySlice<DeviceMemory<double> *> &b, int ldb,
      double beta, const port::ArraySlice<DeviceMemory<double> *> &c, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator);
  Stream &ThenBlasGemmBatchedWithScratch(
      blas::Transpose transa, blas::Transpose transb, uint64 m, uint64 n,
      uint64 k, std::complex<float> alpha,
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a, int lda,
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b, int ldb,
      std::complex<float> beta,
      const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator);
  Stream &ThenBlasGemmBatchedWithScratch(
      blas::Transpose transa, blas::Transpose transb, uint64 m, uint64 n,
      uint64 k, std::complex<double> alpha,
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a, int lda,
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b, int ldb,
      std::complex<double> beta,
      const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c, int ldc,
      int batch_count, ScratchAllocator *scratch_allocator);

  // See BlasSupport::DoBlasHemm.
  Stream &ThenBlasHemm(blas::Side side, blas::UpperLower uplo, uint64 m,
                       uint64 n, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       const DeviceMemory<std::complex<float>> &b, int ldb,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *c, int ldc);
  Stream &ThenBlasHemm(blas::Side side, blas::UpperLower uplo, uint64 m,
                       uint64 n, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       const DeviceMemory<std::complex<double>> &b, int ldb,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *c, int ldc);

  // See BlasSupport::DoBlasHerk.
  Stream &ThenBlasHerk(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                       uint64 k, float alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       float beta, DeviceMemory<std::complex<float>> *c,
                       int ldc);
  Stream &ThenBlasHerk(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                       uint64 k, double alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       double beta, DeviceMemory<std::complex<double>> *c,
                       int ldc);

  // See BlasSupport::DoBlasHer2k.
  Stream &ThenBlasHer2k(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                        uint64 k, std::complex<float> alpha,
                        const DeviceMemory<std::complex<float>> &a, int lda,
                        const DeviceMemory<std::complex<float>> &b, int ldb,
                        float beta, DeviceMemory<std::complex<float>> *c,
                        int ldc);
  Stream &ThenBlasHer2k(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                        uint64 k, std::complex<double> alpha,
                        const DeviceMemory<std::complex<double>> &a, int lda,
                        const DeviceMemory<std::complex<double>> &b, int ldb,
                        double beta, DeviceMemory<std::complex<double>> *c,
                        int ldc);

  // See BlasSupport::DoBlasSymm.
  Stream &ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64 m,
                       uint64 n, float alpha, const DeviceMemory<float> &a,
                       int lda, const DeviceMemory<float> &b, int ldb,
                       float beta, DeviceMemory<float> *c, int ldc);
  Stream &ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64 m,
                       uint64 n, double alpha, const DeviceMemory<double> &a,
                       int lda, const DeviceMemory<double> &b, int ldb,
                       double beta, DeviceMemory<double> *c, int ldc);
  Stream &ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64 m,
                       uint64 n, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       const DeviceMemory<std::complex<float>> &b, int ldb,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *c, int ldc);
  Stream &ThenBlasSymm(blas::Side side, blas::UpperLower uplo, uint64 m,
                       uint64 n, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       const DeviceMemory<std::complex<double>> &b, int ldb,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *c, int ldc);

  // See BlasSupport::DoBlasSyrk.
  Stream &ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                       uint64 k, float alpha, const DeviceMemory<float> &a,
                       int lda, float beta, DeviceMemory<float> *c, int ldc);
  Stream &ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                       uint64 k, double alpha, const DeviceMemory<double> &a,
                       int lda, double beta, DeviceMemory<double> *c, int ldc);
  Stream &ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                       uint64 k, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       std::complex<float> beta,
                       DeviceMemory<std::complex<float>> *c, int ldc);
  Stream &ThenBlasSyrk(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                       uint64 k, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       std::complex<double> beta,
                       DeviceMemory<std::complex<double>> *c, int ldc);

  // See BlasSupport::DoBlasSyr2k.
  Stream &ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                        uint64 k, float alpha, const DeviceMemory<float> &a,
                        int lda, const DeviceMemory<float> &b, int ldb,
                        float beta, DeviceMemory<float> *c, int ldc);
  Stream &ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                        uint64 k, double alpha, const DeviceMemory<double> &a,
                        int lda, const DeviceMemory<double> &b, int ldb,
                        double beta, DeviceMemory<double> *c, int ldc);
  Stream &ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                        uint64 k, std::complex<float> alpha,
                        const DeviceMemory<std::complex<float>> &a, int lda,
                        const DeviceMemory<std::complex<float>> &b, int ldb,
                        std::complex<float> beta,
                        DeviceMemory<std::complex<float>> *c, int ldc);
  Stream &ThenBlasSyr2k(blas::UpperLower uplo, blas::Transpose trans, uint64 n,
                        uint64 k, std::complex<double> alpha,
                        const DeviceMemory<std::complex<double>> &a, int lda,
                        const DeviceMemory<std::complex<double>> &b, int ldb,
                        std::complex<double> beta,
                        DeviceMemory<std::complex<double>> *c, int ldc);

  // See BlasSupport::DoBlasTrmm.
  Stream &ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64 m,
                       uint64 n, float alpha, const DeviceMemory<float> &a,
                       int lda, DeviceMemory<float> *b, int ldb);
  Stream &ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64 m,
                       uint64 n, double alpha, const DeviceMemory<double> &a,
                       int lda, DeviceMemory<double> *b, int ldb);
  Stream &ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64 m,
                       uint64 n, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       DeviceMemory<std::complex<float>> *b, int ldb);
  Stream &ThenBlasTrmm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64 m,
                       uint64 n, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       DeviceMemory<std::complex<double>> *b, int ldb);

  // See BlasSupport::DoBlasTrsm.
  Stream &ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64 m,
                       uint64 n, float alpha, const DeviceMemory<float> &a,
                       int lda, DeviceMemory<float> *b, int ldb);
  Stream &ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64 m,
                       uint64 n, double alpha, const DeviceMemory<double> &a,
                       int lda, DeviceMemory<double> *b, int ldb);
  Stream &ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64 m,
                       uint64 n, std::complex<float> alpha,
                       const DeviceMemory<std::complex<float>> &a, int lda,
                       DeviceMemory<std::complex<float>> *b, int ldb);
  Stream &ThenBlasTrsm(blas::Side side, blas::UpperLower uplo,
                       blas::Transpose transa, blas::Diagonal diag, uint64 m,
                       uint64 n, std::complex<double> alpha,
                       const DeviceMemory<std::complex<double>> &a, int lda,
                       DeviceMemory<std::complex<double>> *b, int ldb);

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
  //   int64 bytes_read = f->Read(seed_data, bytes_to_read);
  //   < error checking >
  //   stream.ThenSetRngSeed(seed_data, bytes_read);
  //
  // To seed with reproducible data:
  //   uint64_t seed_data[2] = { <data> };
  //   stream.ThenSetRngSeed(seed_data, 16);
  Stream &ThenSetRngSeed(const uint8 *seed, uint64 seed_bytes);

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
                     uint64 size);

  // Entrain onto the stream: a memcpy to a GPU destination from a host source
  // of the given target size. host_src must be a pointer to host memory
  // allocated by StreamExecutor::HostMemoryAllocate or otherwise allocated and
  // then registered with StreamExecutor::HostMemoryRegister.
  Stream &ThenMemcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                     uint64 size);

  // Alternative interface for memcpying from device to host that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  Stream &ThenMemcpyD2H(const DeviceMemory<T> &gpu_src,
                        port::MutableArraySlice<T> host_dst) {
    auto host_size = host_dst.size() * sizeof(T);
    CHECK(gpu_src.size() == 0 || host_size >= gpu_src.size());
    return ThenMemcpy(host_dst.begin(), gpu_src, host_size);
  }

  // Alternative interface for memcpying from host to device that takes an
  // array slice. Checks that the destination size can accommodate the host
  // slice size.
  template <typename T>
  Stream &ThenMemcpyH2D(port::ArraySlice<T> host_src,
                        DeviceMemory<T> *gpu_dst) {
    auto host_size = host_src.size() * sizeof(T);
    CHECK(gpu_dst->size() == 0 || gpu_dst->size() >= host_size);
    return ThenMemcpy(gpu_dst, host_src.begin(), host_size);
  }

  // Entrain onto the stream: a memcpy to a GPU destination from a GPU source
  // of the given target size. gpu_src/dst must be pointers to GPU memory and
  // peer access must be enabled between their owning StreamExecutors.
  Stream &ThenMemcpy(DeviceMemoryBase *gpu_dst, const DeviceMemoryBase &gpu_src,
                     uint64 size);

  // Calls to the device-to-device copy overload of ThenMemcpy -- useful for
  // ensuring that the host pointer isn't getting confused accidentally with a
  // device pointer if you're not doing metaprogramming against the API.
  Stream &ThenMemcpyD2D(DeviceMemoryBase *gpu_dst,
                        const DeviceMemoryBase &gpu_src, uint64 size) {
    return ThenMemcpy(gpu_dst, gpu_src, size);
  }

  // Entrain onto the stream: a memset of zero at a GPU location of size
  // bytes.
  // The location must not be null.
  Stream &ThenMemZero(DeviceMemoryBase *location, uint64 size);

  // Entrain onto the stream: a memset of a 32-bit pattern at a GPU location
  // of
  // size bytes, where bytes must be evenly 32-bit sized (i.e. evenly
  // divisible
  // by 4). The location must not be null.
  Stream &ThenMemset32(DeviceMemoryBase *location, const uint32 &pattern,
                       uint64 size);

  // (Synchronously) block the host code waiting for the operations
  // entrained on the stream (enqueued to this point in program
  // execution) to complete.
  //
  // Returns true if the stream is ok().
  bool BlockHostUntilDone();

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
  // Host callbacks block/occupy the stream just as device functions
  // (execute one at a time, block later stream operations).
  // Behavior is undefined when synchronizing using OpenCL user events.
  // Behavior is undefined if host callbacks call device routines or insert
  // them into any stream.
  // On certain platforms, ThenDoHostCallback is expected to have significant
  // negative effects on performance.
  Stream &ThenDoHostCallback(std::function<void()> callback);

  // Identical to ThenDoHostCallback; only exposed for testing purposes.
  Stream &ThenDoHostCallbackForTest(std::function<void()> callback);

  // Returns the StreamExecutor (parent object) associated with this stream.
  StreamExecutor *parent() const {
    CHECK(parent_ != nullptr);
    return parent_;
  }

  // Returns the (internal usage) temporary-memory-allocation manager associated
  // with this stream.
  internal::TemporaryMemoryManager *temporary_memory_manager();

 private:
  friend class host::HostBlas;  // for parent_.
  friend class host::HostFft;   // for parent_.
  friend class host::HostRng;   // for parent_.
  template <typename... Args>
  friend struct ThenBlasImpl;  // for implementing ThenBlasXXX.
  friend class ocl::CLBlas;    // for parent_.

  bool InErrorState() const {
    shared_lock lock{mu_};
    return !ok_;
  }

  // Sets the error state if operation_retcode is false.
  // This is a useful shorthand for many stream routines.
  void CheckError(bool operation_retcode) {
    if (operation_retcode) {
      return;
    }
    mutex_lock lock{mu_};
    ok_ = false;
  }

  void SetError() { CheckError(false /* = operation_retcode */); }

  // The StreamExecutor that supports the operation of this stream.
  StreamExecutor *parent_;

  // The platform-dependent implementation that the StreamExecutor interface
  // delegates to.
  std::unique_ptr<internal::StreamInterface> implementation_;

  // mutex that guards the allocation / error state flags.
  // Mutable so that it can be obtained via const reader lock.
  mutable mutex mu_;

  // Whether Init() was successfully called to allocate this stream on the
  // underlying platform. It simply flips from 0 to 1 with a sanity check.
  // See StreamExecutor::AllocateStream.
  bool allocated_ GUARDED_BY(mu_);

  // Whether all operations have entrained successfully to the current program
  // point.
  bool ok_ GUARDED_BY(mu_);

  // Sub-streams that are generated from this stream. Each element has a pointer
  // to sub-stream and a boolean value indicating if this substream is ready to
  // be reused.
  std::vector<std::pair<std::unique_ptr<Stream>, bool>> sub_streams_
      GUARDED_BY(mu_);

  // Streams can allocate temporary memories to help with work they enqueue
  // (e.g. for scratch memory spaces). This member tracks those allocations and
  // notes when they can be reclaimed -- reclamation is attempted when
  // BlockHostUntilDone() is called.
  internal::TemporaryMemoryManager temporary_memory_manager_;

  // Implementation of ThenConvolveBackwardBias that is shared by all types.
  template <typename T>
  Stream &ThenConvolveBackwardBiasImpl(
      const dnn::BatchDescriptor &input_descriptor,
      const DeviceMemory<T> &input_data,
      const dnn::BatchDescriptor &bias_descriptor,
      DeviceMemory<T> *backward_bias_data);

  SE_DISALLOW_COPY_AND_ASSIGN(Stream);
};

////////////
// Inlines

template <typename T>
inline port::StatusOr<std::unique_ptr<TemporaryDeviceMemory<T>>>
Stream::AllocateTemporaryArray(uint64 element_count) {
  return temporary_memory_manager_.AllocateArray<T>(element_count);
}

inline internal::TemporaryMemoryManager *Stream::temporary_memory_manager() {
  return &temporary_memory_manager_;
}

template <>
struct Quantization<uint8> {
  static constexpr dnn::QuantizedActivationMode kModeId =
      dnn::QuantizedActivationMode::k8Bit;
};

template <>
struct Quantization<uint16> {
  static constexpr dnn::QuantizedActivationMode kModeId =
      dnn::QuantizedActivationMode::k16Bit;
};

template <>
struct Quantization<int32> {
  static constexpr dnn::QuantizedActivationMode kModeId =
      dnn::QuantizedActivationMode::k32Bit;
};

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_STREAM_H_
