/* Copyright 2015 The OpenXLA Authors.

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

// The ROCM-specific DNN library support, implementing the general DnnSupport
// interface.

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_DNN_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_DNN_H_

#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "rocm/include/miopen/miopen.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/plugin_registry.h"

namespace stream_executor {
namespace gpu {

class MIOpenRnnDescriptor;
class MIOpenRnnSequenceTensorDescriptor;
class MIOpenRnnStateTensorDescriptor;
class MIOpenCTCLossDescriptor;

struct PoolingWorkspaceDescriptor {
  std::vector<int64_t> input_dims;
  std::vector<int64_t> output_dims;
  dnn::PoolingDescriptor op;
  int dtype;
  uint64_t timestamp;
  ScopedDeviceMemory<uint8_t> workspace;
  size_t workspace_size;
  bool IsSame(const dnn::BatchDescriptor& input_dimensions,
              const dnn::BatchDescriptor& output_dimensions,
              const dnn::PoolingDescriptor& pooling_dimensions, int _type);
};

struct PoolingWorkspaceCache {
  std::map<const void*, PoolingWorkspaceDescriptor> cache;
  const int trim_size = 1000;
  const uint64_t memory_budget = 2e7;
  uint64_t timestamp = 0;
  uint64_t memory_used = 0;
  bool find(const void* p, const dnn::BatchDescriptor& input_dimensions,
            const dnn::BatchDescriptor& output_dimensions,
            const dnn::PoolingDescriptor& pooling_dimensions, int _type,
            PoolingWorkspaceDescriptor*& pdesc);
  void insert(const void* p, const dnn::BatchDescriptor& input_dimensions,
              const dnn::BatchDescriptor& output_dimensions,
              const dnn::PoolingDescriptor& pooling_dimensions, int _type,
              ScopedDeviceMemory<uint8_t>& workspace, size_t wsp_size,
              hipStream_t hip_stream);

 private:
  void trim(hipStream_t hip_stream);
};

// miopen-library based DNN support. For details on overridden interface
// functions, see dnn.h.
class MIOpenSupport : public dnn::DnnSupport {
 public:
  explicit MIOpenSupport(StreamExecutor* parent);

  absl::Status Init() override;
  absl::StatusOr<stream_executor::dnn::VersionInfo> GetVersion() override;

  absl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>> CreateRnnDescriptor(
      int num_layers, int hidden_size, int input_size, int cell_size,
      int batch_size, dnn::RnnInputMode input_mode,
      dnn::RnnDirectionMode direction_mode, dnn::RnnMode rnn_mode,
      dnn::DataType data_type, const dnn::AlgorithmConfig& algorithm_config,
      const NumericOptions& numeric_options, float dropout, uint64_t seed,
      ScratchAllocator* state_allocator, bool use_padded_io) override;

  absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  CreateRnnSequenceTensorDescriptor(int seq_length, int batch_size,
                                    int data_size,
                                    dnn::DataType data_type) override;

  absl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
  CreateRnnStateTensorDescriptor(int num_layer, int batch_size, int data_size,
                                 dnn::DataType data_type) override;

  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceMemory<Eigen::half>& input_data,
                    const DeviceMemory<int>& seq_lengths_data,
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
                    const DeviceMemory<int>& seq_lengths_data,
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
                    const DeviceMemory<int>& seq_lengths_data,
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
                     const DeviceMemory<int>& seq_lengths_data,
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
                     DeviceMemory<uint8_t>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceMemory<float>& input_data,
                     const DeviceMemory<int>& seq_lengths_data,
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
                     DeviceMemory<uint8_t>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceMemory<double>& input_data,
                     const DeviceMemory<int>& seq_lengths_data,
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
                     DeviceMemory<uint8_t>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  absl::Status GetConvolveRunners(
      bool use_cudnn_frontend, dnn::ConvolutionKind kind,
      dnn::DataType input_type, dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, ScratchAllocator* scratch_allocator,
      const NumericOptions& numeric_options,
      std::vector<std::unique_ptr<const dnn::ConvRunner>>* out_runners)
      override;

  absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> ConvolveRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor) override;

  absl::StatusOr<std::unique_ptr<const dnn::FusedConvRunner>>
  FusedConvolveRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType bias_type, dnn::DataType output_type, double conv_scale,
      double side_input_scale, double leakyrelu_alpha,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& bias_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      dnn::ActivationMode activation_mode) override;

  bool GetMIOpenConvolveAlgorithms(
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      ScratchAllocator* scratch_allocator,
      std::vector<dnn::ProfileResult>* out_algorithms);

  absl::Status GetMIOpenConvolveAlgorithmsImmediateMode(
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      ScratchAllocator* scratch_allocator,
      std::vector<dnn::ProfileResult>* out_algorithms);

  absl::Status GetMIOpenConvolveAlgorithmsFindMode(
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      ScratchAllocator* scratch_allocator,
      std::vector<dnn::ProfileResult>* out_algorithms);

  bool GetRnnAlgorithms(
      std::vector<dnn::AlgorithmDesc>* out_algorithms) override;

  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<float>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const DeviceMemory<float>& side_input, const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      const double exponential_average_factor,
      dnn::ActivationMode activation_mode, DeviceMemory<float>* y,
      DeviceMemory<float>* batch_mean, DeviceMemory<float>* batch_var,
      DeviceMemory<float>* saved_mean, DeviceMemory<float>* saved_inv_var,
      bool is_training, ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<Eigen::half>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const DeviceMemory<Eigen::half>& side_input,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      const double exponential_average_factor,
      dnn::ActivationMode activation_mode, DeviceMemory<Eigen::half>* y,
      DeviceMemory<float>* batch_mean, DeviceMemory<float>* batch_var,
      DeviceMemory<float>* saved_mean, DeviceMemory<float>* saved_inv_var,
      bool is_training, ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceMemory<Eigen::bfloat16>& x,
      const DeviceMemory<float>& scale, const DeviceMemory<float>& offset,
      const DeviceMemory<float>& estimated_mean,
      const DeviceMemory<float>& estimated_variance,
      const DeviceMemory<Eigen::bfloat16>& side_input,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      const double exponential_average_factor,
      dnn::ActivationMode activation_mode, DeviceMemory<Eigen::bfloat16>* y,
      DeviceMemory<float>* batch_mean, DeviceMemory<float>* batch_var,
      DeviceMemory<float>* saved_mean, DeviceMemory<float>* saved_inv_var,
      bool is_training, ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceMemory<float>& y_backprop,
      const DeviceMemory<float>& x, const DeviceMemory<float>& scale,
      const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
      const DeviceMemory<float>& variance, const DeviceMemory<float>& y,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      dnn::ActivationMode activation_mode, DeviceMemory<float>* x_backprop,
      DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
      DeviceMemory<float>* side_input_backprop,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceMemory<Eigen::half>& y_backprop,
      const DeviceMemory<Eigen::half>& x, const DeviceMemory<float>& scale,
      const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
      const DeviceMemory<float>& inv_var, const DeviceMemory<Eigen::half>& y,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      dnn::ActivationMode activation_mode,
      DeviceMemory<Eigen::half>* x_backprop,
      DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
      DeviceMemory<Eigen::half>* side_input_backprop,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceMemory<Eigen::bfloat16>& y_backprop,
      const DeviceMemory<Eigen::bfloat16>& x, const DeviceMemory<float>& scale,
      const DeviceMemory<float>& offset, const DeviceMemory<float>& mean,
      const DeviceMemory<float>& inv_var,
      const DeviceMemory<Eigen::bfloat16>& y,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      dnn::ActivationMode activation_mode,
      DeviceMemory<Eigen::bfloat16>* x_backprop,
      DeviceMemory<float>* scale_backprop, DeviceMemory<float>* offset_backprop,
      DeviceMemory<Eigen::bfloat16>* side_input_backprop,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) override;

  absl::Status DoConvolve(
      dnn::ConvolutionKind kind, dnn::DataType element_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      dnn::AlgorithmDesc algorithm_desc, DeviceMemory<uint8_t> scratch_memory,
      dnn::ProfileResult* output_profile_result) override;

  absl::Status DoFusedConvolve(
      Stream* stream, dnn::DataType input_type, dnn::DataType side_input_type,
      dnn::DataType bias_type, dnn::DataType output_type,
      const dnn::BatchDescriptor& conv_input_descriptor,
      DeviceMemoryBase conv_input_data, double conv_input_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      DeviceMemoryBase side_input_data, double side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor, DeviceMemoryBase biases,
      dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  absl::Status GetFusedMatmulRunners(
      bool use_cudnn_frontend, dnn::DataType input_type,
      dnn::DataType bias_type, dnn::DataType output_type, Stream* stream,
      bool trans_a, bool trans_b, uint64_t m, uint64_t n, uint64_t k,
      int64_t lda, int64_t ldb, int64_t ldc,
      dnn::ActivationMode activation_mode, bool use_fallback,
      const NumericOptions& numeric_options,
      std::vector<std::unique_ptr<const dnn::FusedMatmulRunner>>*
          out_exec_plans) override;

  absl::Status GetFusedConvolveRunners(
      bool use_cudnn_frontend, dnn::ConvolutionKind kind,
      dnn::DataType input_type, dnn::DataType bias_type,
      dnn::DataType output_type, double conv_scale, double side_input_scale,
      double leakyrelu_alpha, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& bias_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, dnn::ActivationMode activation_mode,
      const NumericOptions& numeric_options,
      std::vector<std::unique_ptr<const dnn::FusedConvRunner>>* out_exec_plans)
      override;

  absl::Status DoPoolForward(dnn::DataType element_type, Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             DeviceMemoryBase input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceMemoryBase output_data,
                             ScratchAllocator* workspace_allocator) override;

  absl::Status DoPoolBackward(dnn::DataType element_type, Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              DeviceMemoryBase input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              DeviceMemoryBase output_data,
                              DeviceMemoryBase input_diff_data,
                              DeviceMemoryBase output_diff_data,
                              ScratchAllocator* workspace_allocator) override;

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
      DeviceMemory<float>* raw_variable_gradient,
      ScratchAllocator* workspace_allocator = nullptr) override;

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

  StreamExecutor* GetParentExecutor() { return parent_; }

  absl::Status DoCtcLoss(Stream* stream, dnn::DataType element_type,
                         const dnn::RnnStateTensorDescriptor& probs_desc,
                         const DeviceMemoryBase probs_data,
                         absl::Span<const int> labels_data,
                         absl::Span<const int> labels_lengths_data,
                         absl::Span<const int> input_lengths_data,
                         DeviceMemoryBase costs_data,
                         const dnn::RnnStateTensorDescriptor& grads_desc,
                         DeviceMemoryBase grads_data,
                         DeviceMemory<uint8_t> scratch_memory,
                         int ctc_loss_algo_id) override;

 private:
  StreamExecutor* parent_;  // Parent executor object. Not owned.

  // Flag to indicate whether Get*Algorithm routines should only return
  // the best algorithm (as opposed to a list of all applicable ones)
  bool return_best_algo_only_;

  // Flag to indicate whether to use Immediate (or Find) mode for Convolutions
  bool use_immediate_mode_;

  // Provide access to the MIOpen handle.
  std::unique_ptr<class MIOpenAccess> miopen_;

  PoolingWorkspaceCache m_pooling_cache;
  bool m_pooling_cache_allowed = false;
  bool m_pooling_cache_enabled = false;

  template <class T, class U>
  absl::Status DoBatchNormalizationForwardImpl(
      Stream* stream, dnn::DataType input_data_type,
      dnn::DataType scale_data_type, const DeviceMemory<T>& x,
      const DeviceMemory<U>& scale, const DeviceMemory<U>& offset,
      const DeviceMemory<U>& estimated_mean,
      const DeviceMemory<U>& estimated_variance,
      const DeviceMemory<T>& side_input, const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      const double exponential_average_factor,
      dnn::ActivationMode activation_mode, DeviceMemory<T>* y,
      DeviceMemory<U>* batch_mean, DeviceMemory<U>* batch_var,
      DeviceMemory<U>* saved_mean, DeviceMemory<U>* saved_inv_var,
      bool is_training);

  template <class T, class U>
  absl::Status DoBatchNormalizationBackwardImpl(
      Stream* stream, int miopen_input_type, int miopen_scale_type,
      const DeviceMemory<T>& y_backprop, const DeviceMemory<T>& x,
      const DeviceMemory<U>& scale, const DeviceMemory<U>& mean,
      const DeviceMemory<U>& variance, const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, const double epsilon,
      DeviceMemory<T>* x_backprop, DeviceMemory<U>* scale_backprop,
      DeviceMemory<U>* offset_backprop);

  template <class T>
  absl::Status DoRnnForwardImpl(
      Stream* stream, const MIOpenRnnDescriptor& rnn_desc,
      const MIOpenRnnSequenceTensorDescriptor& input_desc,
      const DeviceMemory<T>& input_data,
      const MIOpenRnnStateTensorDescriptor& input_h_desc,
      const DeviceMemory<T>& input_h_data,
      const MIOpenRnnStateTensorDescriptor& input_c_desc,
      const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
      const MIOpenRnnSequenceTensorDescriptor& output_desc,
      DeviceMemory<T>* output_data,
      const MIOpenRnnStateTensorDescriptor& output_h_desc,
      DeviceMemory<T>* output_h_data,
      const MIOpenRnnStateTensorDescriptor& output_c_desc,
      DeviceMemory<T>* output_c_data, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);
  template <class T>
  absl::Status DoRnnBackwardImpl(
      Stream* stream, const MIOpenRnnDescriptor& rnn_desc,
      const MIOpenRnnSequenceTensorDescriptor& input_desc,
      const DeviceMemory<T>& input_data,
      const MIOpenRnnStateTensorDescriptor& input_h_desc,
      const DeviceMemory<T>& input_h_data,
      const MIOpenRnnStateTensorDescriptor& input_c_desc,
      const DeviceMemory<T>& input_c_data, const DeviceMemory<T>& params,
      const MIOpenRnnSequenceTensorDescriptor& output_desc,
      const DeviceMemory<T>& output_data,
      const MIOpenRnnStateTensorDescriptor& output_h_desc,
      const DeviceMemory<T>& output_h_data,
      const MIOpenRnnStateTensorDescriptor& output_c_desc,
      const DeviceMemory<T>& output_c_data,
      const DeviceMemory<T>& output_backprop_data,
      const DeviceMemory<T>& output_h_backprop_data,
      const DeviceMemory<T>& output_c_backprop_data,
      DeviceMemory<T>* input_backprop_data,
      DeviceMemory<T>* input_h_backprop_data,
      DeviceMemory<T>* input_c_backprop_data,
      DeviceMemory<T>* params_backprop_data,
      DeviceMemory<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoPrepareForConvolution(
      dnn::ConvolutionKind kind, dnn::DataType element_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor, DeviceMemoryBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceMemoryBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceMemoryBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      const dnn::AlgorithmConfig& algorithm_config,
      ScratchAllocator* scratch_allocator, dnn::AlgorithmDesc* algorithm_desc,
      DeviceMemory<uint8_t>* scratch_memory) override;

  absl::Status DoCtcLossImpl(
      Stream* stream, const MIOpenRnnStateTensorDescriptor& probs_desc,
      const DeviceMemoryBase probs_data, absl::Span<const int> labels_data,
      absl::Span<const int> labels_lengths_data,
      absl::Span<const int> input_lengths_data, DeviceMemoryBase costs_data,
      const MIOpenRnnStateTensorDescriptor& grads_desc,
      DeviceMemoryBase grads_data, const MIOpenCTCLossDescriptor& ctc_loss_desc,
      DeviceMemory<uint8_t> scratch_memory, int ctc_loss_algo_id);

  absl::Status DoPrepareForCtcLoss(
      Stream* stream, dnn::DataType element_type,
      const dnn::RnnStateTensorDescriptor& probs_desc,
      const dnn::RnnStateTensorDescriptor& grads_desc,
      absl::Span<const int> labels_data,
      absl::Span<const int> labels_lengths_data,
      absl::Span<const int> input_lengths_data,
      const NumericOptions& numeric_options,
      ScratchAllocator* scratch_allocator,
      DeviceMemory<uint8_t>* scratch_memory, int* ctc_loss_algo_id) override;

  MIOpenSupport(const MIOpenSupport&) = delete;
  void operator=(const MIOpenSupport&) = delete;
};

// A helper function for the front frameworks.
// e.g., TF(tensorflow/core/kernels/conv_ops.cc, fused_batch_norm_op.cc
// and tensorflow/core/grappler/optimizers/generic_layout_optimizer.cc)
// This will decide whether to use NHWC in Convolution/Batchnorm.
// This mode can be faster in in FP16 workloads on gfx908 and beyond.
// Requires ROCm 5.0+.
// TODO (ROCm): Use autotune to choose between this mode and NCHW
// when MIOpen has more optimized kernels.
bool UseNhwcLayoutForRocm();

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_DNN_H_
