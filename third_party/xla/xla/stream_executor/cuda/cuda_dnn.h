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

// The CUDA-specific DNN library support, implementing the general DnnSupport
// interface.

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_DNN_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_DNN_H_

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/cudnn_frontend/include/cudnn_frontend.h"
#include "xla/stream_executor/cuda/cudnn_sdpa_score_mod.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/protobuf/dnn.pb.h"

namespace stream_executor {
namespace gpu {

class CudnnRnnDescriptor;
class CudnnRnnSequenceTensorDescriptor;
class CudnnRnnStateTensorDescriptor;
class CudnnCtcLossDescriptor;

using BatchDescriptorSlice = absl::Span<const dnn::BatchDescriptor>;

template <typename T>
using DeviceAddressSlice = absl::Span<const DeviceAddress<T>* const>;

class CudnnGraph : public dnn::DnnGraph {
 public:
  explicit CudnnGraph(cudnn_frontend::graph::Graph&& graph)
      : graph_(std::move(graph)) {}
  // Prepares a graph and checks whether it is generally supported.
  absl::Status Prepare(dnn::DnnSupport&, const EngineOptions&) override;
  // Builds single plan of the graph with given ID.
  absl::Status Build(dnn::DnnSupport&, std::optional<int64_t> plan_id) override;
  // Builds all the plans
  absl::Status Execute(Stream& stream, absl::Span<DeviceAddressBase> operands,
                       int64_t local_device_ordinal) const override;
  const cudnn_frontend::graph::Graph& Graph() const { return graph_; }
  void InitDropoutState(int64_t local_device_count, int64_t seed,
                        int64_t increment) override {
    dropout_rng_seed_ = seed;
    current_dropout_rng_offset_ = std::vector<int64_t>(local_device_count, 0);
    dropout_rng_offset_increment_ = increment;
  }
  void UpdateDropoutState(int64_t local_device_ordinal) const {
    current_dropout_rng_offset_[local_device_ordinal] +=
        dropout_rng_offset_increment_;
  }
  absl::StatusOr<bool> SupportsExplicitCommandBufferConstruction()
      const override;
  absl::Status PopulateOrUpdateRawCommandBuffer(
      Stream&, absl::Span<DeviceAddressBase> operands, RawCommandBufferHandle,
      bool do_update) override;

 private:
  cudnn_frontend::graph::Graph graph_;
  int64_t dropout_rng_seed_;
  mutable std::vector<int64_t> current_dropout_rng_offset_;
  int64_t dropout_rng_offset_increment_ = 0;
  using VariantPack = std::unordered_map<int64_t, void*>;
  VariantPack PackOperands(
      absl::Span<DeviceAddressBase> operands, DeviceAddressBase& workspace,
      std::optional<int64_t> local_device_ordinal = std::nullopt) const;
};

// cudnn-library based DNN support. For details on overridden interface
// functions, see dnn.h.
class CudnnSupport : public dnn::DnnSupport {
 public:
  explicit CudnnSupport(StreamExecutor* parent);

  absl::Status Init() override;
  absl::StatusOr<stream_executor::dnn::VersionInfo> GetVersion() override;

  absl::StatusOr<std::unique_ptr<dnn::RnnDescriptor>> CreateRnnDescriptor(
      int num_layers, int hidden_size, int input_size, int cell_size,
      int batch_size, dnn::RnnInputMode input_mode,
      dnn::RnnDirectionMode direction_mode, dnn::RnnMode rnn_mode,
      dnn::DataType data_type, const dnn::AlgorithmConfig& algorithm_config,
      const EngineOptions& engine_options, float dropout, uint64_t seed,
      ScratchAllocator* state_allocator, bool use_padded_io) override;

  absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  CreateRnnSequenceTensorDescriptor(int max_seq_length, int batch_size,
                                    int data_size,
                                    dnn::DataType data_type) override;

  absl::StatusOr<std::unique_ptr<dnn::RnnSequenceTensorDescriptor>>
  CreateRnnSequenceTensorDescriptor(int max_seq_length, int batch_size,
                                    int data_size,
                                    const absl::Span<const int>& seq_lengths,
                                    bool time_major,
                                    dnn::DataType data_type) override;

  absl::StatusOr<std::unique_ptr<dnn::RnnStateTensorDescriptor>>
  CreateRnnStateTensorDescriptor(int num_layer, int batch_size, int data_size,
                                 dnn::DataType data_type) override;

  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceAddress<Eigen::half>& input_data,
                    const DeviceAddress<int>& seq_lengths_data,
                    const dnn::RnnStateTensorDescriptor& input_h_desc,
                    const DeviceAddress<Eigen::half>& input_h_data,
                    const dnn::RnnStateTensorDescriptor& input_c_desc,
                    const DeviceAddress<Eigen::half>& input_c_data,
                    const DeviceAddress<Eigen::half>& params,
                    const dnn::RnnSequenceTensorDescriptor& output_desc,
                    DeviceAddress<Eigen::half>* output_data,
                    const dnn::RnnStateTensorDescriptor& output_h_desc,
                    DeviceAddress<Eigen::half>* output_h_data,
                    const dnn::RnnStateTensorDescriptor& output_c_desc,
                    DeviceAddress<Eigen::half>* output_c_data, bool is_training,
                    ScratchAllocator* reserve_space_allocator,
                    ScratchAllocator* workspace_allocator,
                    dnn::ProfileResult* output_profile_result) override;

  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceAddress<float>& input_data,
                    const DeviceAddress<int>& seq_lengths_data,
                    const dnn::RnnStateTensorDescriptor& input_h_desc,
                    const DeviceAddress<float>& input_h_data,
                    const dnn::RnnStateTensorDescriptor& input_c_desc,
                    const DeviceAddress<float>& input_c_data,
                    const DeviceAddress<float>& params,
                    const dnn::RnnSequenceTensorDescriptor& output_desc,
                    DeviceAddress<float>* output_data,
                    const dnn::RnnStateTensorDescriptor& output_h_desc,
                    DeviceAddress<float>* output_h_data,
                    const dnn::RnnStateTensorDescriptor& output_c_desc,
                    DeviceAddress<float>* output_c_data, bool is_training,
                    ScratchAllocator* reserve_space_allocator,
                    ScratchAllocator* workspace_allocator,
                    dnn::ProfileResult* output_profile_result) override;

  bool DoRnnForward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                    const dnn::RnnSequenceTensorDescriptor& input_desc,
                    const DeviceAddress<double>& input_data,
                    const DeviceAddress<int>& seq_lengths_data,
                    const dnn::RnnStateTensorDescriptor& input_h_desc,
                    const DeviceAddress<double>& input_h_data,
                    const dnn::RnnStateTensorDescriptor& input_c_desc,
                    const DeviceAddress<double>& input_c_data,
                    const DeviceAddress<double>& params,
                    const dnn::RnnSequenceTensorDescriptor& output_desc,
                    DeviceAddress<double>* output_data,
                    const dnn::RnnStateTensorDescriptor& output_h_desc,
                    DeviceAddress<double>* output_h_data,
                    const dnn::RnnStateTensorDescriptor& output_c_desc,
                    DeviceAddress<double>* output_c_data, bool is_training,
                    ScratchAllocator* reserve_space_allocator,
                    ScratchAllocator* workspace_allocator,
                    dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceAddress<Eigen::half>& input_data,
                     const DeviceAddress<int>& seq_lengths_data,
                     const dnn::RnnStateTensorDescriptor& input_h_desc,
                     const DeviceAddress<Eigen::half>& input_h_data,
                     const dnn::RnnStateTensorDescriptor& input_c_desc,
                     const DeviceAddress<Eigen::half>& input_c_data,
                     const DeviceAddress<Eigen::half>& params,
                     const dnn::RnnSequenceTensorDescriptor& output_desc,
                     const DeviceAddress<Eigen::half>& output_data,
                     const dnn::RnnStateTensorDescriptor& output_h_desc,
                     const DeviceAddress<Eigen::half>& output_h_data,
                     const dnn::RnnStateTensorDescriptor& output_c_desc,
                     const DeviceAddress<Eigen::half>& output_c_data,
                     const DeviceAddress<Eigen::half>& output_backprop_data,
                     const DeviceAddress<Eigen::half>& output_h_backprop_data,
                     const DeviceAddress<Eigen::half>& output_c_backprop_data,
                     DeviceAddress<Eigen::half>* input_backprop_data,
                     DeviceAddress<Eigen::half>* input_h_backprop_data,
                     DeviceAddress<Eigen::half>* input_c_backprop_data,
                     DeviceAddress<Eigen::half>* params_backprop_data,
                     DeviceAddress<uint8_t>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceAddress<float>& input_data,
                     const DeviceAddress<int>& seq_lengths_data,
                     const dnn::RnnStateTensorDescriptor& input_h_desc,
                     const DeviceAddress<float>& input_h_data,
                     const dnn::RnnStateTensorDescriptor& input_c_desc,
                     const DeviceAddress<float>& input_c_data,
                     const DeviceAddress<float>& params,
                     const dnn::RnnSequenceTensorDescriptor& output_desc,
                     const DeviceAddress<float>& output_data,
                     const dnn::RnnStateTensorDescriptor& output_h_desc,
                     const DeviceAddress<float>& output_h_data,
                     const dnn::RnnStateTensorDescriptor& output_c_desc,
                     const DeviceAddress<float>& output_c_data,
                     const DeviceAddress<float>& output_backprop_data,
                     const DeviceAddress<float>& output_h_backprop_data,
                     const DeviceAddress<float>& output_c_backprop_data,
                     DeviceAddress<float>* input_backprop_data,
                     DeviceAddress<float>* input_h_backprop_data,
                     DeviceAddress<float>* input_c_backprop_data,
                     DeviceAddress<float>* params_backprop_data,
                     DeviceAddress<uint8_t>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  bool DoRnnBackward(Stream* stream, const dnn::RnnDescriptor& rnn_desc,
                     const dnn::RnnSequenceTensorDescriptor& input_desc,
                     const DeviceAddress<double>& input_data,
                     const DeviceAddress<int>& seq_lengths_data,
                     const dnn::RnnStateTensorDescriptor& input_h_desc,
                     const DeviceAddress<double>& input_h_data,
                     const dnn::RnnStateTensorDescriptor& input_c_desc,
                     const DeviceAddress<double>& input_c_data,
                     const DeviceAddress<double>& params,
                     const dnn::RnnSequenceTensorDescriptor& output_desc,
                     const DeviceAddress<double>& output_data,
                     const dnn::RnnStateTensorDescriptor& output_h_desc,
                     const DeviceAddress<double>& output_h_data,
                     const dnn::RnnStateTensorDescriptor& output_c_desc,
                     const DeviceAddress<double>& output_c_data,
                     const DeviceAddress<double>& output_backprop_data,
                     const DeviceAddress<double>& output_h_backprop_data,
                     const DeviceAddress<double>& output_c_backprop_data,
                     DeviceAddress<double>* input_backprop_data,
                     DeviceAddress<double>* input_h_backprop_data,
                     DeviceAddress<double>* input_c_backprop_data,
                     DeviceAddress<double>* params_backprop_data,
                     DeviceAddress<uint8_t>* reserve_space_data,
                     ScratchAllocator* workspace_allocator,
                     dnn::ProfileResult* output_profile_result) override;

  absl::Status GetConvolveRunners(
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor,
      DeviceAddressBase input_data,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceAddressBase filter_data,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceAddressBase output_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, ScratchAllocator* scratch_allocator,
      const EngineOptions& engine_options,
      std::vector<std::unique_ptr<const dnn::ConvRunner>>* out_exec_plans)
      override;

  absl::StatusOr<std::unique_ptr<const dnn::ConvRunner>> ConvolveRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor) override;

  absl::Status GetGraphConvolveRunners(
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, const EngineOptions& engine_options,
      std::vector<std::unique_ptr<const dnn::GraphConvRunner>>* out_exec_plans,
      std::string serialized_graph) override;

  absl::StatusOr<std::unique_ptr<const dnn::GraphConvRunner>>
  GraphConvolveRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType output_type, const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      std::string serialized_graph) override;

  absl::Status GetFusedConvolveRunners(
      dnn::ConvolutionKind kind, dnn::DataType input_type,
      dnn::DataType bias_type, dnn::DataType output_type, double conv_scale,
      double side_input_scale, double leakyrelu_alpha, Stream* stream,
      const dnn::BatchDescriptor& input_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::BatchDescriptor& bias_descriptor,
      const dnn::BatchDescriptor& output_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      bool use_fallback, dnn::ActivationMode activation_mode,
      const EngineOptions& engine_options,
      std::vector<std::unique_ptr<const dnn::FusedConvRunner>>* out_exec_plans)
      override;

  absl::Status GetFusedMatmulRunners(
      dnn::DataType input_type, dnn::DataType bias_type,
      dnn::DataType output_type, Stream* stream, bool trans_a, bool trans_b,
      uint64_t m, uint64_t n, uint64_t k, int64_t lda, int64_t ldb, int64_t ldc,
      dnn::ActivationMode activation_mode, bool use_fallback,
      const EngineOptions& engine_options,
      std::vector<std::unique_ptr<const dnn::FusedMatmulRunner>>*
          out_exec_plans) override;

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

  absl::StatusOr<std::unique_ptr<const dnn::NormRunner>> NormRunnerFromDesc(
      Stream* stream, const dnn::AlgorithmDesc& algorithm_desc,
      dnn::NormKind kind, double epsilon,
      const dnn::TensorDescriptor& x_descriptor,
      const dnn::TensorDescriptor& scale_descriptor,
      const dnn::TensorDescriptor& y_or_dx_descriptor,
      std::optional<dnn::TensorDescriptor> bias_descriptor,
      std::optional<dnn::TensorDescriptor> dy_descriptor,
      std::optional<dnn::TensorDescriptor> expectation_descriptor,
      std::optional<dnn::TensorDescriptor> norm_factor_descriptor,
      std::optional<dnn::TensorDescriptor> dscale_descriptor,
      std::optional<dnn::TensorDescriptor> dbias_descriptor) override;

  bool GetRnnAlgorithms(
      std::vector<dnn::AlgorithmDesc>* out_algorithms) override;

  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceAddress<float>& x,
      const DeviceAddress<float>& scale, const DeviceAddress<float>& offset,
      const DeviceAddress<float>& estimated_mean,
      const DeviceAddress<float>& estimated_var_iance,
      const DeviceAddress<float>& side_input,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, double epsilon,
      double exponential_average_factor, dnn::ActivationMode activation_mode,
      DeviceAddress<float>* y, DeviceAddress<float>* batch_mean,
      DeviceAddress<float>* batch_var, DeviceAddress<float>* saved_mean,
      DeviceAddress<float>* saved_inv_var, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceAddress<Eigen::half>& x,
      const DeviceAddress<float>& scale, const DeviceAddress<float>& offset,
      const DeviceAddress<float>& estimated_mean,
      const DeviceAddress<float>& estimated_variance,
      const DeviceAddress<Eigen::half>& side_input,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, double epsilon,
      double exponential_average_factor, dnn::ActivationMode activation_mode,
      DeviceAddress<Eigen::half>* y, DeviceAddress<float>* batch_mean,
      DeviceAddress<float>* batch_var, DeviceAddress<float>* saved_mean,
      DeviceAddress<float>* saved_inv_var, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationForward(
      Stream* stream, const DeviceAddress<Eigen::bfloat16>& x,
      const DeviceAddress<float>& scale, const DeviceAddress<float>& offset,
      const DeviceAddress<float>& estimated_mean,
      const DeviceAddress<float>& estimated_variance,
      const DeviceAddress<Eigen::bfloat16>& side_input,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, double epsilon,
      double exponential_average_factor, dnn::ActivationMode activation_mode,
      DeviceAddress<Eigen::bfloat16>* y, DeviceAddress<float>* batch_mean,
      DeviceAddress<float>* batch_var, DeviceAddress<float>* saved_mean,
      DeviceAddress<float>* saved_inv_var, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceAddress<float>& y_backprop,
      const DeviceAddress<float>& x, const DeviceAddress<float>& scale,
      const DeviceAddress<float>& offset, const DeviceAddress<float>& mean,
      const DeviceAddress<float>& inv_var, const DeviceAddress<float>& y,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, double epsilon,
      dnn::ActivationMode activation_mode, DeviceAddress<float>* x_backprop,
      DeviceAddress<float>* scale_backprop,
      DeviceAddress<float>* offset_backprop,
      DeviceAddress<float>* side_input_backprop,
      DeviceAddress<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceAddress<Eigen::half>& y_backprop,
      const DeviceAddress<Eigen::half>& x, const DeviceAddress<float>& scale,
      const DeviceAddress<float>& offset, const DeviceAddress<float>& mean,
      const DeviceAddress<float>& inv_var, const DeviceAddress<Eigen::half>& y,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, double epsilon,
      dnn::ActivationMode activation_mode,
      DeviceAddress<Eigen::half>* x_backprop,
      DeviceAddress<float>* scale_backprop,
      DeviceAddress<float>* offset_backprop,
      DeviceAddress<Eigen::half>* side_input_backprop,
      DeviceAddress<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) override;

  bool DoBatchNormalizationBackward(
      Stream* stream, const DeviceAddress<Eigen::bfloat16>& y_backprop,
      const DeviceAddress<Eigen::bfloat16>& x,
      const DeviceAddress<float>& scale, const DeviceAddress<float>& offset,
      const DeviceAddress<float>& mean, const DeviceAddress<float>& inv_var,
      const DeviceAddress<Eigen::bfloat16>& y,
      const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, double epsilon,
      dnn::ActivationMode activation_mode,
      DeviceAddress<Eigen::bfloat16>* x_backprop,
      DeviceAddress<float>* scale_backprop,
      DeviceAddress<float>* offset_backprop,
      DeviceAddress<Eigen::bfloat16>* side_input_backprop,
      DeviceAddress<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator) override;

  absl::Status DoFusedConvolve(
      Stream* stream, dnn::DataType input_type, dnn::DataType side_input_type,
      dnn::DataType bias_type, dnn::DataType output_type,
      const dnn::BatchDescriptor& conv_input_descriptor,
      DeviceAddressBase conv_input_data, double conv_scale,
      const dnn::FilterDescriptor& filter_descriptor,
      DeviceAddressBase filter_data,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      DeviceAddressBase side_input_data, double side_input_scale,
      const dnn::BatchDescriptor& bias_descriptor, DeviceAddressBase biases,
      dnn::ActivationMode activation_mode,
      const dnn::BatchDescriptor& output_descriptor,
      DeviceAddressBase output_data, ScratchAllocator* scratch_allocator,
      const dnn::AlgorithmConfig& algorithm_config,
      dnn::ProfileResult* output_profile_result) override;

  absl::Status CudnnReorderConvolutionFilterAndBias(
      Stream* stream, const dnn::FilterDescriptor& filter_descriptor,
      const DeviceAddress<int8_t>& filter_input,
      DeviceAddress<int8_t>* filter_output,
      std::optional<const DeviceAddress<float>> bias_input,
      std::optional<DeviceAddress<float>> bias_output) override;

  absl::Status DoPoolForward(dnn::DataType element_type, Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const dnn::BatchDescriptor& input_dimensions,
                             DeviceAddressBase input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceAddressBase output_data,
                             ScratchAllocator* workspace_allocator) override;

  absl::Status DoPoolForward(dnn::DataType element_type, Stream* stream,
                             const dnn::PoolingDescriptor& pooling_dimensions,
                             const EngineOptions& engine_options,
                             const dnn::BatchDescriptor& input_dimensions,
                             DeviceAddressBase input_data,
                             const dnn::BatchDescriptor& output_dimensions,
                             DeviceAddressBase output_data,
                             ScratchAllocator* workspace_allocator) override;

  absl::Status DoPoolBackward(dnn::DataType element_type, Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const dnn::BatchDescriptor& input_dimensions,
                              DeviceAddressBase input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              DeviceAddressBase output_data,
                              DeviceAddressBase input_diff_data,
                              DeviceAddressBase output_diff_data,
                              ScratchAllocator* workspace_allocator) override;

  absl::Status DoPoolBackward(dnn::DataType element_type, Stream* stream,
                              const dnn::PoolingDescriptor& pooling_dimensions,
                              const EngineOptions& engine_options,
                              const dnn::BatchDescriptor& input_dimensions,
                              DeviceAddressBase input_data,
                              const dnn::BatchDescriptor& output_dimensions,
                              DeviceAddressBase output_data,
                              DeviceAddressBase input_diff_data,
                              DeviceAddressBase output_diff_data,
                              ScratchAllocator* workspace_allocator) override;

  bool DoNormalizeWithDimensions(
      Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
      const dnn::BatchDescriptor& dimensions,
      const DeviceAddress<float>& input_data,
      DeviceAddress<float>* output_data) override;

  bool DoNormalizeBackwardWithDimensions(
      Stream* stream, const dnn::NormalizeDescriptor& normalize_descriptor,
      const dnn::BatchDescriptor& dimensions,
      const DeviceAddress<float>& raw_data,
      const DeviceAddress<float>& normalized_data,
      const DeviceAddress<float>& normalized_variable_gradient,
      DeviceAddress<float>* raw_variable_gradient,
      ScratchAllocator* workspace_allocator) override;

  // Derives an output batch descriptor from an input batch and convolution
  // descriptors.
  bool DeriveOutputBatchDescriptor(
      const dnn::BatchDescriptor& batch_descriptor,
      const dnn::FilterDescriptor& filter_descriptor,
      const dnn::ConvolutionDescriptor& convolution_descriptor,
      dnn::BatchDescriptor* output_batch_descriptor);

  absl::Status DoCtcLoss(Stream* stream, dnn::DataType element_type,
                         const dnn::RnnStateTensorDescriptor& probs_desc,
                         DeviceAddressBase probs_data,
                         absl::Span<const int> labels_data,
                         absl::Span<const int> labels_lengths_data,
                         absl::Span<const int> input_lengths_data,
                         DeviceAddressBase costs_data,
                         const dnn::RnnStateTensorDescriptor& grads_desc,
                         DeviceAddressBase grads_data,
                         DeviceAddress<uint8_t> scratch_memory,
                         int ctc_loss_algo_id) override;

  bool DoTransformTensor(Stream* stream, const dnn::BatchDescriptor& input_desc,
                         dnn::DataType input_type,
                         const DeviceAddressBase& input_data,
                         const dnn::BatchDescriptor& output_desc,
                         dnn::DataType output_type, float scale,
                         DeviceAddressBase* output_data) override;

  void NotifyStreamDestroyed(Stream* stream) override;

  // Loads complete graph from its serialized representation.
  absl::StatusOr<std::unique_ptr<dnn::DnnGraph>> DeserializeGraph(
      Stream& stream, absl::string_view serialized_data) const override;

 private:
  // Uses cuDNN handle for execution.
  friend class CudnnGraph;

  StreamExecutor* parent_;  // Parent executor object. Not owned.

  // Provides access to the cuDNN handle.
  std::unique_ptr<class CudnnAccess> cudnn_;

  bool GetConvolveAlgorithms(CudaComputeCapability cuda_compute_capability,
                             dnn::DataType input_type,
                             const EngineOptions& engine_options,
                             std::vector<dnn::AlgorithmDesc>* out_algorithms);

  bool GetConvolveBackwardDataAlgorithms(
      CudaComputeCapability cuda_compute_capability, dnn::DataType input_type,
      const EngineOptions& engine_options,
      std::vector<dnn::AlgorithmDesc>* out_algorithms);

  bool GetConvolveBackwardFilterAlgorithms(
      CudaComputeCapability cuda_compute_capability, dnn::DataType input_type,
      const EngineOptions& engine_options,
      std::vector<dnn::AlgorithmDesc>* out_algorithms);

  template <class T, class U>
  absl::Status DoBatchNormalizationForwardImpl(
      Stream* stream, dnn::DataType input_data_type,
      dnn::DataType scale_data_type, const DeviceAddress<T>& x,
      const DeviceAddress<U>& scale, const DeviceAddress<U>& offset,
      const DeviceAddress<U>& estimated_mean,
      const DeviceAddress<U>& estimated_variance,
      const DeviceAddress<T>& side_input, const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, double epsilon,
      double exponential_average_factor, dnn::ActivationMode activation_mode,
      DeviceAddress<T>* y, DeviceAddress<U>* batch_mean,
      DeviceAddress<U>* batch_var, DeviceAddress<U>* saved_mean,
      DeviceAddress<U>* saved_inv_var, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator);

  template <class T, class U>
  absl::Status DoBatchNormalizationBackwardImpl(
      Stream* stream, int cudnn_input_type, int cudnn_scale_type,
      const DeviceAddress<T>& y_backprop, const DeviceAddress<T>& x,
      const DeviceAddress<U>& scale, const DeviceAddress<U>& offset,
      const DeviceAddress<U>& mean, const DeviceAddress<U>& inv_var,
      const DeviceAddress<T>& y, const dnn::BatchDescriptor& x_desc,
      const dnn::BatchDescriptor& scale_offset_desc, double epsilon,
      dnn::ActivationMode activation_mode, DeviceAddress<T>* x_backprop,
      DeviceAddress<U>* scale_backprop, DeviceAddress<U>* offset_backprop,
      DeviceAddress<T>* side_input_backprop,
      DeviceAddress<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator);

  template <class T>
  absl::Status DoRnnForwardImpl(
      Stream* stream, const CudnnRnnDescriptor& rnn_desc,
      const CudnnRnnSequenceTensorDescriptor& input_desc,
      const DeviceAddress<T>& input_data,
      const DeviceAddress<int>& seq_lengths_data,
      const CudnnRnnStateTensorDescriptor& input_h_desc,
      const DeviceAddress<T>& input_h_data,
      const CudnnRnnStateTensorDescriptor& input_c_desc,
      const DeviceAddress<T>& input_c_data, const DeviceAddress<T>& params,
      const CudnnRnnSequenceTensorDescriptor& output_desc,
      DeviceAddress<T>* output_data,
      const CudnnRnnStateTensorDescriptor& output_h_desc,
      DeviceAddress<T>* output_h_data,
      const CudnnRnnStateTensorDescriptor& output_c_desc,
      DeviceAddress<T>* output_c_data, bool is_training,
      ScratchAllocator* reserve_space_allocator,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  template <class T>
  absl::Status DoRnnBackwardImpl(
      Stream* stream, const CudnnRnnDescriptor& rnn_desc,
      const CudnnRnnSequenceTensorDescriptor& input_desc,
      const DeviceAddress<T>& input_data,
      const DeviceAddress<int>& seq_lengths_data,
      const CudnnRnnStateTensorDescriptor& input_h_desc,
      const DeviceAddress<T>& input_h_data,
      const CudnnRnnStateTensorDescriptor& input_c_desc,
      const DeviceAddress<T>& input_c_data, const DeviceAddress<T>& params,
      const CudnnRnnSequenceTensorDescriptor& output_desc,
      const DeviceAddress<T>& output_data,
      const CudnnRnnStateTensorDescriptor& output_h_desc,
      const DeviceAddress<T>& output_h_data,
      const CudnnRnnStateTensorDescriptor& output_c_desc,
      const DeviceAddress<T>& output_c_data,
      const DeviceAddress<T>& output_backprop_data,
      const DeviceAddress<T>& output_h_backprop_data,
      const DeviceAddress<T>& output_c_backprop_data,
      DeviceAddress<T>* input_backprop_data,
      DeviceAddress<T>* input_h_backprop_data,
      DeviceAddress<T>* input_c_backprop_data,
      DeviceAddress<T>* params_backprop_data,
      DeviceAddress<uint8_t>* reserve_space_data,
      ScratchAllocator* workspace_allocator,
      dnn::ProfileResult* output_profile_result);

  absl::Status DoCtcLossImpl(
      Stream* stream, const CudnnRnnStateTensorDescriptor& probs_desc,
      DeviceAddressBase probs_data, absl::Span<const int> labels_data,
      absl::Span<const int> labels_lengths_data,
      absl::Span<const int> input_lengths_data, DeviceAddressBase costs_data,
      const CudnnRnnStateTensorDescriptor& grads_desc,
      DeviceAddressBase grads_data, const CudnnCtcLossDescriptor& ctc_loss_desc,
      DeviceAddress<uint8_t> scratch_memory, int ctc_loss_algo_id);

 private:
  absl::Status DoPrepareForCtcLoss(
      Stream* stream, dnn::DataType element_type,
      const dnn::RnnStateTensorDescriptor& probs_desc,
      const dnn::RnnStateTensorDescriptor& grads_desc,
      absl::Span<const int> labels_data,
      absl::Span<const int> labels_lengths_data,
      absl::Span<const int> input_lengths_data,
      const EngineOptions& engine_options, ScratchAllocator* scratch_allocator,
      DeviceAddress<uint8_t>* scratch_memory, int* ctc_loss_algo_id) override;

  CudnnSupport(const CudnnSupport&) = delete;
  void operator=(const CudnnSupport&) = delete;
};

absl::StatusOr<CudnnGraph> GetCudnnFlashAttentionOperationGraph(
    dnn::DnnSupport& dnn_support,
    const dnn::MatmulTensorDescriptor& q_descriptor,
    const dnn::MatmulTensorDescriptor& k_descriptor,
    const dnn::MatmulTensorDescriptor& v_descriptor,
    const dnn::TensorDescriptor& o_descriptor,
    std::optional<dnn::TensorDescriptor> bias_descriptor,
    std::optional<dnn::TensorDescriptor> stats_descriptor,
    std::optional<dnn::TensorDescriptor> page_table_k_descriptor,
    std::optional<dnn::TensorDescriptor> page_table_v_descriptor, double scale,
    bool use_dropout, std::optional<double> dropout_rate,
    dnn::FMHAMaskKind mask_type, int sliding_window_length,
    int max_seg_per_batch, ScoreModFunc* score_mod);

absl::StatusOr<CudnnGraph> GetCudnnFlashAttentionF8OperationGraph(
    dnn::DnnSupport& dnn_support,
    const dnn::MatmulTensorDescriptor& q_descriptor,
    const dnn::MatmulTensorDescriptor& k_descriptor,
    const dnn::MatmulTensorDescriptor& v_descriptor,
    const dnn::TensorDescriptor& o_descriptor,
    const std::optional<dnn::TensorDescriptor>& stats_descriptor, double scale,
    dnn::FMHAMaskKind mask_type);

absl::StatusOr<CudnnGraph> GetCudnnFlashAttentionBackwardOperationGraph(
    dnn::DnnSupport& dnn_support, const dnn::MatmulTensorDescriptor& q_desc,
    const dnn::MatmulTensorDescriptor& k_desc,
    const dnn::MatmulTensorDescriptor& p_desc,
    const dnn::MatmulTensorDescriptor& v_desc,
    const dnn::MatmulTensorDescriptor& do_desc,
    const dnn::TensorDescriptor& dq_desc, const dnn::TensorDescriptor& dk_desc,
    const dnn::TensorDescriptor& dv_desc,
    const std::optional<dnn::TensorDescriptor> bias_descriptor,
    const std::optional<dnn::TensorDescriptor> dbias_descriptor,
    std::optional<double> dropout_rate, std::optional<int64_t> seed,
    double scale, bool use_dropout, bool use_bias,
    const dnn::FMHAMaskKind mask_type, bool force_deterministic,
    const int sliding_window_length, const int max_seg_per_batch,
    ScoreModFunc* score_mod);

absl::StatusOr<CudnnGraph> GetCudnnFlashAttentionBackwardF8OperationGraph(
    dnn::DnnSupport& dnn_support, const dnn::MatmulTensorDescriptor& q_desc,
    const dnn::MatmulTensorDescriptor& k_desc,
    const dnn::MatmulTensorDescriptor& p_desc,
    const dnn::MatmulTensorDescriptor& v_desc,
    const dnn::MatmulTensorDescriptor& do_desc,
    const dnn::TensorDescriptor& dq_desc, const dnn::TensorDescriptor& dk_desc,
    const dnn::TensorDescriptor& dv_desc, double scale,
    dnn::FMHAMaskKind mask_type);

absl::StatusOr<CudnnGraph> GetCudnnBlockScaledDotOperationGraph(
    dnn::DnnSupport& dnn_support, const dnn::TensorDescriptor& lhs_data,
    const dnn::TensorDescriptor& lhs_scale,
    const dnn::TensorDescriptor& rhs_data,
    const dnn::TensorDescriptor& rhs_scale, dnn::DataType result_type,
    int block_size, bool has_global_scale);

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_DNN_H_
