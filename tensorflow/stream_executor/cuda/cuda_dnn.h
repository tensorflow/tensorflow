/* Copyright 2015 Google Inc. All Rights Reserved.

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

// The CUDA-specific DNN library support, implementing the general DnnSupport
// interface.

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DNN_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DNN_H_

#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"

namespace perftools {
namespace gputools {
namespace cuda {

class CUDAExecutor;

// Opaque and unique identifer for the cuDNN plugin.
extern const PluginId kCuDnnPlugin;

// cudnn-library based DNN support. For details on overridden interface
// functions, see dnn.h.
class CudnnSupport : public dnn::DnnSupport {
 public:
  explicit CudnnSupport(CUDAExecutor* parent);
  ~CudnnSupport() override;

  port::Status Init() override;

  bool DoConvolve(Stream* stream, const dnn::BatchDescriptor& input_descriptor,
                  const DeviceMemory<float>& input_data,
                  const dnn::FilterDescriptor& filter_descriptor,
                  const DeviceMemory<float>& filter_data,
                  const dnn::ConvolutionDescriptor& convolution_descriptor,
                  const dnn::BatchDescriptor& output_descriptor,
                  DeviceMemory<float>* output_data,
                  ScratchAllocator* scratch_allocator) override;

  bool DoConvolve(Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
                  const DeviceMemory<double>& input_data,
                  const dnn::FilterDescriptor& filter_descriptor,
                  const DeviceMemory<double>& filter_data,
                  const dnn::ConvolutionDescriptor& convolution_descriptor,
                  const dnn::BatchDescriptor& output_descriptor,
                  DeviceMemory<double>* output_data) override;

  bool DoSeparableConvolve(
      Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
      const DeviceMemory<float>& input_data,
      const dnn::FilterDescriptor& filter_descriptor, int depth_multiplier,
      const DeviceMemory<float>& first_weights,
      const DeviceMemory<float>& second_weights,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data) override {
    LOG(ERROR) << "separable convolution not supported by CUDNN";
    return false;
  }

  bool DoConvolveBackwardData(
      Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<float>& filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& input_descriptor,
      DeviceMemory<float>* backward_input_data,
      ScratchAllocator* scratch_allocator) override;

  bool DoConvolveBackwardFilter(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<float>& input_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemory<float>* backward_filter_data,
      ScratchAllocator* scratch_allocator) override;

  bool DoMatMul(Stream* stream, const DeviceMemory<float>& input_data,
                const DeviceMemory<float>& weights,
                const dnn::BatchDescriptor& input_dimensions,
                const dnn::BatchDescriptor& output_dimensions,
                DeviceMemory<float>* output_data) override;

  bool DoMatMulQuantized(Stream* stream, const DeviceMemory<float>& input_data,
                         const DeviceMemory<int8>& quantized_weights,
                         const DeviceMemory<float>& weight_scales,
                         const dnn::BatchDescriptor& input_dimensions,
                         const dnn::BatchDescriptor& output_dimensions,
                         DeviceMemory<float>* output_data) override {
    LOG(ERROR) << "DNN MatMulQuantized not supported by CUDNN";
    return false;
  }

  bool DoMatMulQuantized(Stream* stream, const DeviceMemory<float>& input_data,
                         const DeviceMemory<int16>& quantized_weights,
                         const DeviceMemory<float>& weight_scales,
                         const dnn::BatchDescriptor& input_dimensions,
                         const dnn::BatchDescriptor& output_dimensions,
                         DeviceMemory<float>* output_data) override {
    LOG(ERROR) << "DNN MatMulQuantized not supported by CUDNN";
    return false;
  }

  bool DoBiasAdd(Stream* stream, const DeviceMemory<float>& input_data,
                 const DeviceMemory<float>& biases,
                 const dnn::BatchDescriptor& dimensions,
                 DeviceMemory<float>* output_data) override;

  bool DoActivate(Stream* stream, dnn::ActivationMode activation_mode,
                  const dnn::BatchDescriptor& dimensions,
                  const DeviceMemory<float>& input_data,
                  DeviceMemory<float>* output_data) override;

  bool DoPoolForward(Stream* stream,
                     const dnn::PoolingDescriptor& pooling_dimensions,
                     const dnn::BatchDescriptor& input_dimensions,
                     const DeviceMemory<float>& input_data,
                     const dnn::BatchDescriptor& output_dimensions,
                     DeviceMemory<float>* output_data) override;

  bool DoPoolBackward(Stream* stream,
                      const dnn::PoolingDescriptor& pooling_dimensions,
                      const dnn::BatchDescriptor& input_dimensions,
                      const DeviceMemory<float>& input_data,
                      const dnn::BatchDescriptor& output_dimensions,
                      const DeviceMemory<float>& output_data,
                      const DeviceMemory<float>& input_diff_data,
                      DeviceMemory<float>* output_diff_data) override;

  bool DoNormalize(Stream* stream,
                   const dnn::NormalizeDescriptor& normalize_descriptor,
                   const DeviceMemory<float>& input_data,
                   DeviceMemory<float>* output_data) override;

  bool DoDepthConcatenate(
      Stream* stream, port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
      port::ArraySlice<const DeviceMemory<float>*> input_data,
      DeviceMemory<float>* output_data) override;

  bool DoElementwiseOperate(
      Stream* stream, dnn::ElementwiseOperation operation,
      port::ArraySlice<dnn::BatchDescriptor> input_dimensions,
      port::ArraySlice<const DeviceMemory<float>*> input_data,
      const dnn::BatchDescriptor& output_dimensions,
      DeviceMemory<float>* output_data) override;

  bool DoXYPad(Stream* stream, const dnn::BatchDescriptor &dimensions,
               const DeviceMemory<float> &input_data,
               int64 left_pad, int64 right_pad, int64 top_pad,
               int64 bottom_pad, DeviceMemory<float> *output_data) override;

  bool DoXYSlice(Stream* stream, const dnn::BatchDescriptor &dimensions,
                 const DeviceMemory<float> &input_data,
                 int64 left_trim, int64 right_trim, int64 top_trim,
                 int64 bottom_trim, DeviceMemory<float> *output_data) override;

  bool DoMemcpyD2HQuantized(Stream* stream,
                            const DeviceMemory<float>& device_unquantized_src,
                            dnn::QuantizedActivationMode mode, void* host_dst,
                            int64 size) override;

  bool DoMemcpyH2DQuantized(
      Stream* stream, const void* host_src, int64 size,
      dnn::QuantizedActivationMode mode,
      DeviceMemory<float>* device_unquantized_dst) override;

  // Derives an output batch descriptor from an input batch and convolution
  // descriptors.
  bool DeriveOutputBatchDescriptor(
      const dnn::BatchDescriptor& batch_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      dnn::BatchDescriptor* output_batch_descriptor);

 private:
  // Guards the enqueueing of DNN operations via the dnn_handle_ below.
  mutex dnn_handle_mutex_;

  CUDAExecutor* parent_;  // Parent executor object. Not owned.

  // cudnn library handle. cudnnHandle_t type is not present in this header to
  // prevent third-party library header inclusions from leaking outside the
  // single cuda_dnn translation unit.
  void* dnn_handle_ GUARDED_BY(dnn_handle_mutex_);

  // NOTE(keveman): Temporary data layout transformation until cuDNN supports
  // kBatchYXDepth for backward pass. This function allocates temporary memory,
  // lays out the source data into the temporary but in the kBatchDepthXY
  // layout, and returns the temporary memory. The caller is responsible for
  // deallocating the temporary. Since the allocation is done using Stream's
  // AllocateTemporaryMemory, a later BlockHostUntilDone could be used for
  // deallocation.
  //
  // transform_scratch is populated with a legitimate temporary allocation iff
  // the original output data needs to be transformed.
  DeviceMemory<float> MaybeTransformLayout(
      Stream* stream, dnn::BatchDescriptor* output_descriptor,
      DeviceMemory<float> backward_output_data,
      std::unique_ptr<TemporaryDeviceMemory<float>>* transform_scratch)
      EXCLUSIVE_LOCKS_REQUIRED(dnn_handle_mutex_);

  SE_DISALLOW_COPY_AND_ASSIGN(CudnnSupport);
};

}  // namespace cuda
}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DNN_H_
