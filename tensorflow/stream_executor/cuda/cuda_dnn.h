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

namespace stream_executor {
namespace cuda {

class CUDAExecutor;
class CudnnRnnDescriptor;
class CudnnRnnSequenceTensorDescriptor;
class CudnnRnnStateTensorDescriptor;

// Opaque and unique identifier for the cuDNN plugin.
extern const PluginId kCuDnnPlugin;

// cudnn-library based DNN support. For details on overridden interface
// functions, see dnn.h.
class CudnnSupport : public dnn::DnnSupport {
 public:
  explicit CudnnSupport(CUDAExecutor* parent);
  ~CudnnSupport() override;

  port::Status Init() override;
  port::StatusOr<perftools::gputools::dnn::VersionInfo> GetVersion() override;

  port::StatusOr<std::unique_ptr<dnn::RnnDescriptor>> createRnnDescriptor(
      int num_layers, int hidden_size, int input_size,
      dnn::RnnInputMode input_mode, dnn::RnnDirectionMode direction_mode,
      dnn::RnnMode rnn_mode, dnn::DataType data_type,
      const dnn::AlgorithmConfig& algorithm_config, float dropout, uint64 seed,
      ScratchAllocator* state_allocator) override;

  port::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  createRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                    int data_size,
                                    dnn::DataType data_type) override;

  port::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
  createRnnStateTensorDescriptor(int num_layer, int batch_size, int data_size,
                                 dnn::DataType data_type) override;

  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceMemory<Eigen::half>& input_data,
                    const dnn::RnnStateTensorDescriptor& input_h_desc,
                    const DeviceMemory<Eigen::half>& input_h_data,
                    const dnn::RnnStateTensorDescriptor& input_c_desc,
                    const DeviceMemory<Eigen::half>& input_c_data,
                    const DeviceMemory<Eigen::half>& params,
                    const dnn::RnnSequenceTensorDescriptor& output_desc,
                    DeviceMemory<Eigen::half>* output_data,
                    const dnn::RnnStateTensorDescriptor& output_h_desc,
                    DeviceMemory<Eigen::half>* output_h_data,
                    const dnn::RnnStateTensorDescriptor& output_c_desc,
                    DeviceMemory<Eigen::half>* output_c_data, bool is_training,
                    ScratchAllocator* reserve_space_allocator,
                    ScratchAllocator* workspace_allocator,
                    dnn::ProfileResult* output_profile_result) override;

  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceMemory<float>& input_data,
                    const dnn::RnnStateTensorDescriptor& input_h_desc,
                    const DeviceMemory<float>& input_h_data,
                    const dnn::RnnStateTensorDescriptor& input_c_desc,
                    const DeviceMemory<float>& input_c_data,
                    const DeviceMemory<float>& params,
                    const dnn::RnnSequenceTensorDescriptor& output_desc,
                    DeviceMemory<float>* output_data,
                    const dnn::RnnStateTensorDescriptor& output_h_desc,
                    DeviceMemory<float>* output_h_data,
                    const dnn::RnnStateTensorDescriptor& output_c_desc,
                    DeviceMemory<float>* output_c_data, bool is_training,
                    ScratchAllocator* reserve_space_allocator,
                    ScratchAllocator* workspace_allocator,
                    dnn::ProfileResult* output_profile_result) override;

  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceMemory<double>& input_data,
                    const dnn::RnnStateTensorDescriptor& input_h_desc,
                    const DeviceMemory<double>& input_h_data,
                    const dnn::RnnStateTensorDescriptor& input_c_desc,
                    const DeviceMemory<double>& input_c_data,
                    const DeviceMemory<double>& params,
                    const dnn::RnnSequenceTensorDescriptor& output_desc,
                    DeviceMemory<double>* output_data,
                    const dnn::RnnStateTensorDescriptor& output_h_desc,
                    DeviceMemory<double>* output_h_data,
                    const dnn::RnnStateTensorDescriptor& output_c_desc,
                    DeviceMemory<double>* output_c_data, bool is_training,
                    ScratchAllocator* reserve_space_allocator,
                    ScratchAllocator* workspace_allocator,
                    dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceMemory<Eigen::half>& input_data,
                     const dnn::RnnStateTensorDescriptor& input_h_desc,
                     const DeviceMemory<Eigen::half>& input_h_data,
                     const dnn::RnnStateTensorDescriptor& input_c_desc,
                     const DeviceMemory<Eigen::half>& input_c_data,
                     const DeviceMemory<Eigen::half>& params,
                     const dnn::RnnSequenceTensorDescriptor& output_desc,
                     const DeviceMemory<Eigen::half>& output_data,
                     const dnn::RnnStateTensorDescriptor& output_h_desc,
                     const DeviceMemory<Eigen::half>& output_h_data,
                     const dnn::RnnStateTensorDescriptor& output_c_desc,
                     const DeviceMemory<Eigen::half>& output_c_data,
                     const DeviceMemory<Eigen::half>& output_backprop_data,
                     const DeviceMemory<Eigen::half>& output_h_backprop_data,
                     const DeviceMemory<Eigen::half>& output_c_backprop_data,
                     DeviceMemory<Eigen::half>* input_backprop_data,
                     DeviceMemory<Eigen::half>* input_h_backprop_data,
                     DeviceMemory<Eigen::half>* input_c_backprop_data,
                     DeviceMemory<Eigen::half>* params_backprop_data,
                     DeviceMemory<uint8>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceMemory<float>& input_data,
                     const dnn::RnnStateTensorDescriptor& input_h_desc,
                     const DeviceMemory<float>& input_h_data,
                     const dnn::RnnStateTensorDescriptor& input_c_desc,
                     const DeviceMemory<float>& input_c_data,
                     const DeviceMemory<float>& params,
                     const dnn::RnnSequenceTensorDescriptor& output_desc,
                     const DeviceMemory<float>& output_data,
                     const dnn::RnnStateTensorDescriptor& output_h_desc,
                     const DeviceMemory<float>& output_h_data,
                     const dnn::RnnStateTensorDescriptor& output_c_desc,
                     const DeviceMemory<float>& output_c_data,
                     const DeviceMemory<float>& output_backprop_data,
                     const DeviceMemory<float>& output_h_backprop_data,
                     const DeviceMemory<float>& output_c_backprop_data,
                     DeviceMemory<float>* input_backprop_data,
                     DeviceMemory<float>* input_h_backprop_data,
                     DeviceMemory<float>* input_c_backprop_data,
                     DeviceMemory<float>* params_backprop_data,
                     DeviceMemory<uint8>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceMemory<double>& input_data,
                     const dnn::RnnStateTensorDescriptor& input_h_desc,
                     const DeviceMemory<double>& input_h_data,
                     const dnn::RnnStateTensorDescriptor& input_c_desc,
                     const DeviceMemory<double>& input_c_data,
                     const DeviceMemory<double>& params,
                     const dnn::RnnSequenceTensorDescriptor& output_desc,
                     const DeviceMemory<double>& output_data,
                     const dnn::RnnStateTensorDescriptor& output_h_desc,
                     const DeviceMemory<double>& output_h_data,
                     const dnn::RnnStateTensorDescriptor& output_c_desc,
                     const DeviceMemory<double>& output_c_data,
                     const DeviceMemory<double>& output_backprop_data,
                     const DeviceMemory<double>& output_h_backprop_data,
                     const DeviceMemory<double>& output_c_backprop_data,
                     DeviceMemory<double>* input_backprop_data,
                     DeviceMemory<double>* input_h_backprop_data,
                     DeviceMemory<double>* input_c_backprop_data,
                     DeviceMemory<double>* params_backprop_data,
                     DeviceMemory<uint8>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  bool GetConvolveAlgorithms(
      bool with_winograd_nonfused, int cc_major, int cc_minor,
      std::vector<dnn::AlgorithmDesc>* out_algorithms) override;

  bool GetRnnAlgorithms(
      std::vector<dnn::AlgorithmDesc>* out_algorithms) override;

  bool GetConvolveBackwardDataAlgorithms(
      bool with_winograd_nonfused, int cc_major, int cc_minor,
      std::vector<dnn::AlgorithmDesc>* out_algorithms) override;

  bool GetConvolveBackwardFilterAlgorithms(
      bool with_winograd_nonfused, int cc_major, int cc_minor,
      std::vector<dnn::AlgorithmDesc>* out_algorithms) override;

  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<float>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<float>* y, DeviceMemory<float>* batch_mean,
      DeviceMemory<float>* batch_var, DeviceMemory<float>* saved_mean,
      DeviceMemory<float>* saved_inv_var, bool is_training,
      std::function<const DeviceMemory<float>&()> var_to_inv_var,
      std::function<void()> inv_var_to_var) override;

  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<Eigen::half>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<Eigen::half>* y, DeviceMemory<float>* batch_mean,
      DeviceMemory<float>* batch_var, DeviceMemory<float>* saved_mean,
      DeviceMemory<float>* saved_inv_var, bool is_training,
      std::function<const DeviceMemory<float>&()> var_to_inv_var,
      std::function<void()> inv_var_to_var) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceMemory<float>& y_backprop,
      const DeviceMemory<float>& x, const DeviceMemory<float>& scale,
      const DeviceMemory<float>& mean, const DeviceMemory<float>& inv_var,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<float>* x_backprop, DeviceMemory<float>* scale_backprop,
      DeviceMemory<float>* offset_backprop) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceMemory<Eigen::half>& y_backprop,
      const DeviceMemory<Eigen::half>& x, const DeviceMemory<float>& scale,
      const DeviceMemory<float>& mean, const DeviceMemory<float>& inv_var,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<Eigen::half>* x_backprop,
      DeviceMemory<float>* scale_backprop,
      DeviceMemory<float>* offset_backprop) override;

  bool DoConvolve(Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
                  const DeviceMemory<float>& input_data,
                  const dnn::FilterDescriptor& filter_descriptor,
                  const DeviceMemory<float>& filter_data,
                  const dnn::ConvolutionDescriptor& convolution_descriptor,
                  const dnn::BatchDescriptor& output_descriptor,
                  DeviceMemory<float>* output_data,
                  ScratchAllocator* scratch_allocator,
                  const dnn::AlgorithmConfig& algorithm_config,
                  dnn::ProfileResult* output_profile_result) override;

  bool DoConvolve(Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
                  const DeviceMemory<double>& input_data,
                  const dnn::FilterDescriptor& filter_descriptor,
                  const DeviceMemory<double>& filter_data,
                  const dnn::ConvolutionDescriptor& convolution_descriptor,
                  const dnn::BatchDescriptor& output_descriptor,
                  DeviceMemory<double>* output_data,
                  ScratchAllocator* scratch_allocator,
                  const dnn::AlgorithmConfig& algorithm_config,
                  dnn::ProfileResult* output_profile_result) override;

  bool DoConvolve(Stream* stream, const dnn::BatchDescriptor& batch_descriptor,
                  const DeviceMemory<Eigen::half>& input_data,
                  const dnn::FilterDescriptor& filter_descriptor,
                  const DeviceMemory<Eigen::half>& filter_data,
                  const dnn::ConvolutionDescriptor& convolution_descriptor,
                  const dnn::BatchDescriptor& output_descriptor,
                  DeviceMemory<Eigen::half>* output_data,
                  ScratchAllocator* scratch_allocator,
                  const dnn::AlgorithmConfig& algorithm_config,
                  dnn::ProfileResult* output_profile_result) override;

  bool DoFusedConvolve(
      Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
      const DeviceMemory<double>& conv_input_data, double conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<double>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const DeviceMemory<double>& side_input_data, double side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor,
      const DeviceMemory<double>& biases, dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<double>* output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  bool DoFusedConvolve(
      Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
      const DeviceMemory<float>& conv_input_data, float conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<float>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const DeviceMemory<float>& side_input_data, float side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor,
      const DeviceMemory<float>& biases, dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  bool DoFusedConvolve(Stream* stream,
                       const dnn::BatchDescriptor& conv_input_descriptor,
                       const DeviceMemory<Eigen::half>& conv_input_data,
                       float conv_input_scale,
                       const dnn::FilterDescriptor& filter_descriptor,
                       const DeviceMemory<Eigen::half>& filter_data,
                       const dnn::ConvolutionDescriptor& convolution_descriptor,
                       const DeviceMemory<Eigen::half>& side_input_data,
                       float side_input_scale,
                       const dnn::BatchDescriptor& bias_descriptor,
                       const DeviceMemory<Eigen::half>& biases,
                       dnn::ActivationMode activation_mode,
                       const dnn::BatchDescriptor& output_descriptor,
                       DeviceMemory<Eigen::half>* output_data,
                       ScratchAllocator* scratch_allocator,
                       const dnn::AlgorithmConfig& algorithm_config,
                       dnn::ProfileResult* output_profile_result) override;

  bool DoFusedConvolve(
      Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
      const DeviceMemory<int8>& conv_input_data, float conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<int8>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const DeviceMemory<int8>& side_input_data, float side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor,
      const DeviceMemory<float>& biases, dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<int8>* output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  bool DoConvolveQuantized(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<float>& input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<int8>& filter_coefficients,
      const DeviceMemory<float>& coefficient_scales,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data) override {
    LOG(ERROR) << "DoConvolveQuantized not supported by cuDNN";
    return false;
  }

  bool DoConvolveQuantized(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<float>& input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<int16>& filter_coefficients,
      const DeviceMemory<float>& coefficient_scales,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float>* output_data) override {
    LOG(ERROR) << "DoConvolveQuantized not supported by cuDNN";
    return false;
  }

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
      const DeviceMemory<double>& filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<double> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& input_descriptor,
      DeviceMemory<double>* backward_input_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  bool DoConvolveBackwardData(
      Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<float>& filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& input_descriptor,
      DeviceMemory<float>* backward_input_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  bool DoConvolveBackwardData(
      Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<Eigen::half>& filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<Eigen::half> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& input_descriptor,
      DeviceMemory<Eigen::half>* backward_input_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  bool DoConvolveBackwardFilter(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<double>& input_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<double> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemory<double>* backward_filter_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  bool DoConvolveBackwardFilter(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<float>& input_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<float> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemory<float>* backward_filter_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  bool DoConvolveBackwardFilter(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<Eigen::half>& input_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<Eigen::half> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemory<Eigen::half>* backward_filter_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  bool DoConvolveBackwardBias(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<double>& input_data,
      const dnn::BatchDescriptor& bias_descriptor,
      DeviceMemory<double>* backward_bias_data) override;

  bool DoConvolveBackwardBias(Stream* stream,
                              const dnn::BatchDescriptor& input_descriptor,
                              const DeviceMemory<float>& input_data,
                              const dnn::BatchDescriptor& bias_descriptor,
                              DeviceMemory<float>* backward_bias_data) override;

  bool DoConvolveBackwardBias(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<Eigen::half>& input_data,
      const dnn::BatchDescriptor& bias_descriptor,
      DeviceMemory<Eigen::half>* backward_bias_data) override;

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
                  DeviceMemory<float>* output_data, uint64 options) override;

  bool DoPoolForward(Stream* stream,
                     const dnn::PoolingDescriptor& pooling_dimensions,
                     const dnn::BatchDescriptor& input_dimensions,
                     const DeviceMemory<double>& input_data,
                     const dnn::BatchDescriptor& output_dimensions,
                     DeviceMemory<double>* output_data) override;

  bool DoPoolForward(Stream* stream,
                     const dnn::PoolingDescriptor& pooling_dimensions,
                     const dnn::BatchDescriptor& input_dimensions,
                     const DeviceMemory<float>& input_data,
                     const dnn::BatchDescriptor& output_dimensions,
                     DeviceMemory<float>* output_data) override;

  bool DoPoolForward(Stream* stream,
                     const dnn::PoolingDescriptor& pooling_dimensions,
                     const dnn::BatchDescriptor& input_dimensions,
                     const DeviceMemory<Eigen::half>& input_data,
                     const dnn::BatchDescriptor& output_dimensions,
                     DeviceMemory<Eigen::half>* output_data) override;

  bool DoPoolBackward(Stream* stream,
                      const dnn::PoolingDescriptor& pooling_dimensions,
                      const dnn::BatchDescriptor& input_dimensions,
                      const DeviceMemory<double>& input_data,
                      const dnn::BatchDescriptor& output_dimensions,
                      const DeviceMemory<double>& output_data,
                      const DeviceMemory<double>& input_diff_data,
                      DeviceMemory<double>* output_diff_data) override;

  bool DoPoolBackward(Stream* stream,
                      const dnn::PoolingDescriptor& pooling_dimensions,
                      const dnn::BatchDescriptor& input_dimensions,
                      const DeviceMemory<float>& input_data,
                      const dnn::BatchDescriptor& output_dimensions,
                      const DeviceMemory<float>& output_data,
                      const DeviceMemory<float>& input_diff_data,
                      DeviceMemory<float>* output_diff_data) override;

  bool DoPoolBackward(Stream* stream,
                      const dnn::PoolingDescriptor& pooling_dimensions,
                      const dnn::BatchDescriptor& input_dimensions,
                      const DeviceMemory<Eigen::half>& input_data,
                      const dnn::BatchDescriptor& output_dimensions,
                      const DeviceMemory<Eigen::half>& output_data,
                      const DeviceMemory<Eigen::half>& input_diff_data,
                      DeviceMemory<Eigen::half>* output_diff_data) override;

  bool DoNormalize(Stream* stream,
                   const dnn::NormalizeDescriptor& normalize_descriptor,
                   const DeviceMemory<float>& input_data,
                   DeviceMemory<float>* output_data) override;

  bool DoNormalizeWithDimensions(
      Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
      const dnn::BatchDescriptor& dimensions,
      const DeviceMemory<float>& input_data,
      DeviceMemory<float>* output_data) override;

  bool DoNormalizeBackwardWithDimensions(
      Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
      const dnn::BatchDescriptor& dimensions,
      const DeviceMemory<float>& raw_data,
      const DeviceMemory<float>& normalized_data,
      const DeviceMemory<float>& normalized_variable_gradient,
      DeviceMemory<float>* raw_variable_gradient) override;

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

  bool DoTransformTensor(Stream* stream, const dnn::BatchDescriptor& input_desc,
                         dnn::DataType input_type,
                         const DeviceMemoryBase& input_data,
                         const dnn::BatchDescriptor& output_desc,
                         dnn::DataType output_type, float scale,
                         DeviceMemoryBase* output_data) override;

  const Stream* GetCurrentDnnStream() const
      SHARED_LOCKS_REQUIRED(dnn_handle_mutex_) {
    return current_dnn_stream_;
  }

  void SetCurrentDnnStream(Stream* stream)
      EXCLUSIVE_LOCKS_REQUIRED(dnn_handle_mutex_) {
    current_dnn_stream_ = stream;
  }

  CUDAExecutor* GetParentExecutor() { return parent_; }

  // Guards the enqueueing of DNN operations via the dnn_handle_ below, and
  // access to current_dnn_stream_.
  //
  // This is a public member because we need to add thread safty annotations in
  // the cudnn wrapper functions in the cc file, which need to access this
  // mutex (the annotations require C++ permission checks).
  mutex dnn_handle_mutex_;

 private:
  CUDAExecutor* parent_;  // Parent executor object. Not owned.

  // cudnn library handle. cudnnHandle_t type is not present in this header to
  // prevent third-party library header inclusions from leaking outside the
  // single cuda_dnn translation unit.
  void* dnn_handle_ GUARDED_BY(dnn_handle_mutex_);

  // The current cudnn stream that is set by cudnnSetStream().
  Stream* current_dnn_stream_ GUARDED_BY(dnn_handle_mutex_);

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
  template<class T>
  DeviceMemory<T> MaybeTransformLayout(
      Stream* stream,
      dnn::BatchDescriptor* output_descriptor,
      DeviceMemory<T> backward_output_data,
      std::unique_ptr<TemporaryDeviceMemory<T>>* transform_scratch)
      EXCLUSIVE_LOCKS_REQUIRED(dnn_handle_mutex_);

  template <class T, class U>
  bool DoBatchNormalizationForwardImpl(
      Stream* stream, dnn::DataType input_data_type,
      dnn::DataType scale_data_type, const DeviceMemory<T>& x,
      const DeviceMemory<U>& scale, const DeviceMemory<U>& offset,
      const DeviceMemory<U>& estimated_mean,
      const DeviceMemory<U>& estimated_variance,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<T>* y, DeviceMemory<U>* batch_mean,
      DeviceMemory<U>* batch_var, DeviceMemory<U>* saved_mean,
      DeviceMemory<U>* saved_inv_var, bool is_training,
      std::function<const DeviceMemory<U>&()> var_to_inv_var,
      std::function<void()> inv_var_to_var);

  template <class T, class U>
  bool DoBatchNormalizationBackwardImpl(
      Stream* stream, int cudnn_input_type, int cudnn_scale_type,
      const DeviceMemory<T>& y_backprop, const DeviceMemory<T>& x,
      const DeviceMemory<U>& scale, const DeviceMemory<U>& mean,
      const DeviceMemory<U>& inv_var, const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<T>* x_backprop, DeviceMemory<U>* scale_backprop,
      DeviceMemory<U>* offset_backprop);

  template <class T>
  bool DoConvolveImpl(Stream* stream,
                      const dnn::BatchDescriptor& batch_descriptor,
                      const DeviceMemory<T>& input_data,
                      const dnn::FilterDescriptor& filter_descriptor,
                      const DeviceMemory<T>& filter_data,
                      const dnn::ConvolutionDescriptor& convolution_descriptor,
                      const dnn::BatchDescriptor& output_descriptor,
                      DeviceMemory<T>* output_data,
                      ScratchAllocator* scratch_allocator,
                      const dnn::AlgorithmConfig& algorithm_config,
                      dnn::ProfileResult* output_profile_result);

  template <typename Type, typename BiasType, typename ScaleType,
            int cudnn_data_type, int cudnn_compute_type>
  bool DoFusedConvolveImpl(
      Stream* stream, const dnn::BatchDescriptor& conv_input_descriptor,
      const DeviceMemory<Type>& conv_input_data, ScaleType conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<Type>& filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const DeviceMemory<Type>& side_input_data, ScaleType side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor,
      const DeviceMemory<BiasType>& biases, dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<Type>* output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result);

  template <class T>
  bool DoConvolveBackwardDataImpl(
      Stream* stream,
      const dnn::FilterDescriptor& filter_descriptor,
      const DeviceMemory<T>& filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemory<T> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::BatchDescriptor& input_descriptor,
      DeviceMemory<T>* backward_input_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result);

  template <class T>
  bool DoConvolveBackwardFilterImpl(
      Stream* stream, const dnn::BatchDescriptor& input_descriptor,
      const DeviceMemory<T>& input_data,
      const dnn::BatchDescriptor& output_descriptor_in,
      DeviceMemory<T> backward_output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemory<T>* backward_filter_data,
      ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result);

  template <class T>
  bool DoConvolveBackwardBiasImpl(Stream* stream,
                                  const dnn::BatchDescriptor& input_descriptor,
                                  const DeviceMemory<T>& input_data,
                                  const dnn::BatchDescriptor& bias_descriptor,
                                  DeviceMemory<T>* backward_bias_data);

  template <class T>
  bool DoRnnForwardImpl(Stream* stream, const CudnnRnnDescriptor& rnn_desc,
                        const CudnnRnnSequenceTensorDescriptor& input_desc,
                        const DeviceMemory<T>& input_data,
                        const CudnnRnnStateTensorDescriptor& input_h_desc,
                        const DeviceMemory<T>& input_h_data,
                        const CudnnRnnStateTensorDescriptor& input_c_desc,
                        const DeviceMemory<T>& input_c_data,
                        const DeviceMemory<T>& params,
                        const CudnnRnnSequenceTensorDescriptor& output_desc,
                        DeviceMemory<T>* output_data,
                        const CudnnRnnStateTensorDescriptor& output_h_desc,
                        DeviceMemory<T>* output_h_data,
                        const CudnnRnnStateTensorDescriptor& output_c_desc,
                        DeviceMemory<T>* output_c_data, bool is_training,
                        ScratchAllocator* reserve_space_allocator,
                        ScratchAllocator* workspace_allocator,
                        dnn::ProfileResult* output_profile_result);

  template <class T>
  bool DoRnnBackwardImpl(Stream* stream, const CudnnRnnDescriptor& rnn_desc,
                         const CudnnRnnSequenceTensorDescriptor& input_desc,
                         const DeviceMemory<T>& input_data,
                         const CudnnRnnStateTensorDescriptor& input_h_desc,
                         const DeviceMemory<T>& input_h_data,
                         const CudnnRnnStateTensorDescriptor& input_c_desc,
                         const DeviceMemory<T>& input_c_data,
                         const DeviceMemory<T>& params,
                         const CudnnRnnSequenceTensorDescriptor& output_desc,
                         const DeviceMemory<T>& output_data,
                         const CudnnRnnStateTensorDescriptor& output_h_desc,
                         const DeviceMemory<T>& output_h_data,
                         const CudnnRnnStateTensorDescriptor& output_c_desc,
                         const DeviceMemory<T>& output_c_data,
                         const DeviceMemory<T>& output_backprop_data,
                         const DeviceMemory<T>& output_h_backprop_data,
                         const DeviceMemory<T>& output_c_backprop_data,
                         DeviceMemory<T>* input_backprop_data,
                         DeviceMemory<T>* input_h_backprop_data,
                         DeviceMemory<T>* input_c_backprop_data,
                         DeviceMemory<T>* params_backprop_data,
                         DeviceMemory<uint8>* reserve_space_data,
                         ScratchAllocator* workspace_allocator,
                         dnn::ProfileResult* output_profile_result);

  SE_DISALLOW_COPY_AND_ASSIGN(CudnnSupport);
};

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_DNN_H_
