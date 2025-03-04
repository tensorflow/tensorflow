/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------*/

#include "tensorflow/core/kernels/conv_ops_gpu.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "tensorflow/core/kernels/autotune_conv_impl.h"
#include "tensorflow/core/kernels/numeric_options_utils.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

bool ComputeInNhwcEnabled(DataType data_type, se::Stream* stream,
                          bool use_4d_tensor) {
#if GOOGLE_CUDA
  // Tensor Core supports efficient convolution with fp16 for NVIDIA Volta+
  // GPUs and bf16/tf32 for Ampere+ GPUs in NHWC data layout. In all other
  // configurations it's more efficient to run computation in NCHW data format.
  bool use_nhwc_tf32 = data_type == DT_FLOAT &&
                       stream->GetCudaComputeCapability().IsAtLeast(
                           se::CudaComputeCapability::kAmpere) &&
                       tensorflow::tensor_float_32_execution_enabled();
  bool use_nhwc_fp16 =
      data_type == DT_HALF && stream->GetCudaComputeCapability().IsAtLeast(
                                  se::CudaComputeCapability::kVolta);
  bool use_nhwc_bf16 =
      data_type == DT_BFLOAT16 && stream->GetCudaComputeCapability().IsAtLeast(
                                      se::CudaComputeCapability::kAmpere);
  if (use_4d_tensor) {
    return use_nhwc_fp16 || use_nhwc_tf32 || use_nhwc_bf16;
  }
  return CUDNN_VERSION >= 8000 &&
         (use_nhwc_fp16 || use_nhwc_tf32 || use_nhwc_bf16);
#else
  // fast NHWC implementation is a CUDA only feature
  return false;
#endif  // GOOGLE_CUDA
}

// Finds the best convolution algorithm for the given ConvLaunch (cuda
// convolution on the stream) and parameters, by running all possible
// algorithms and measuring execution time.
template <typename T>
StatusOr<AutotuneEntry<se::dnn::FusedConvOp>> AutotuneFusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::FusedConvOp>>*
        autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, double leakyrelu_alpha,
    se::DeviceMemory<T> input_ptr, se::DeviceMemory<T> filter_ptr,
    se::DeviceMemory<T> output_ptr, se::DeviceMemory<T> bias_ptr,
    se::DeviceMemory<T> side_input_ptr, int64_t scratch_size_limit) {
#if GOOGLE_CUDA
  AutotuneEntry<se::dnn::FusedConvOp> autotune_entry;
  auto* stream = ctx->op_device_context()->stream();

  if (!autotune_map->Find(params, &autotune_entry)) {
    profiler::ScopedAnnotation trace("cudnn_autotuning");

    se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                                stream);
    se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter);
    se::DeviceMemory<T> output_ptr_rz(
        WrapRedzoneBestEffort(&rz_allocator, output_ptr));

    std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>> runners;
    auto element_type = se::dnn::ToDataType<T>::value;
    auto dnn = stream->parent()->AsDnn();
    if (dnn == nullptr) {
      return absl::InvalidArgumentError("No DNN in stream executor.");
    }
    TF_RETURN_IF_ERROR(dnn->GetFusedConvolveRunners(
        CudnnUseFrontend(), se::dnn::ConvolutionKind::FORWARD, element_type,
        element_type, element_type, conv_scale, side_input_scale,
        leakyrelu_alpha, stream, input_desc, filter_desc, bias_desc,
        output_desc, conv_desc, /*use_fallback=*/false, activation_mode,
        GetNumericOptionsForCuDnn(), &runners));

    auto launch_func =
        [&](se::ScratchAllocator* allocator_used,
            const std::unique_ptr<const se::dnn::FusedConvRunner>& runner,
            se::dnn::ProfileResult* profile_result) -> Status {
      TF_ASSIGN_OR_RETURN(auto scratch, allocator_used->AllocateBytes(
                                            runner->GetWorkspaceSize()));
      return (*runner)(stream, profile_result, scratch, input_ptr, filter_ptr,
                       side_input_ptr, bias_ptr, output_ptr_rz);
    };

    TF_ASSIGN_OR_RETURN(auto results,
                        internal::AutotuneConvImpl(
                            ctx, runners, cudnn_use_autotune, launch_func,
                            scratch_size_limit, rz_allocator));
    // Only log on an AutotuneConv cache miss.
    LogFusedConvForwardAutotuneResults(
        se::dnn::ToDataType<T>::value, input_ptr, filter_ptr, output_ptr,
        bias_ptr, side_input_ptr, input_desc, filter_desc, output_desc,
        conv_desc, conv_scale, side_input_scale, activation_mode,
        stream->parent(), results);

    // Two-level autotuning: Cudnn frontend supports two engine lists:
    // heuristics and fallback. Heuristics engines are normally faster.
    // To reduce autotuning time, we evaluate the fallback engines only when
    // none of the heuristics engines work.
    bool found_working_engine = false;
    for (auto& result : results) {
      if (!result.has_failure()) {
        found_working_engine = true;
        break;
      }
    }

    if (!CudnnUseFrontend() || found_working_engine) {
      TF_ASSIGN_OR_RETURN(autotune_entry,
                          BestCudnnConvAlgorithm<se::dnn::FusedConvOp>(
                              results, std::move(runners)));
    } else {
      LOG(WARNING)
          << "None of the algorithms provided by cuDNN frontend heuristics "
             "worked; trying fallback algorithms.  Conv: "
          << params.ToString();
      std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>>
          fallback_runners;
      TF_RETURN_IF_ERROR(dnn->GetFusedConvolveRunners(
          CudnnUseFrontend(), se::dnn::ConvolutionKind::FORWARD, element_type,
          element_type, element_type, conv_scale, side_input_scale,
          leakyrelu_alpha, stream, input_desc, filter_desc, bias_desc,
          output_desc, conv_desc, /*use_fallback=*/true, activation_mode,
          GetNumericOptionsForCuDnn(), &fallback_runners));

      TF_ASSIGN_OR_RETURN(auto fallback_results,
                          internal::AutotuneConvImpl(
                              ctx, fallback_runners, cudnn_use_autotune,
                              launch_func, scratch_size_limit, rz_allocator));

      LogFusedConvForwardAutotuneResults(
          se::dnn::ToDataType<T>::value, input_ptr, filter_ptr, output_ptr,
          bias_ptr, side_input_ptr, input_desc, filter_desc, output_desc,
          conv_desc, conv_scale, side_input_scale, activation_mode,
          stream->parent(), fallback_results);

      TF_ASSIGN_OR_RETURN(autotune_entry,
                          BestCudnnConvAlgorithm<se::dnn::FusedConvOp>(
                              fallback_results, std::move(fallback_runners)));
    }

    autotune_map->Insert(params, autotune_entry);
  }
  return autotune_entry;
#else
  return errors::Unimplemented(
      "Fused conv not implemented on non-CUDA platforms.");
#endif
}

template StatusOr<AutotuneEntry<se::dnn::FusedConvOp>>
AutotuneFusedConv<double>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::FusedConvOp>>*
        autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, double leakyrelu_alpha,
    se::DeviceMemory<double> input_ptr, se::DeviceMemory<double> filter_ptr,
    se::DeviceMemory<double> output_ptr, se::DeviceMemory<double> bias_ptr,
    se::DeviceMemory<double> side_input_ptr, int64_t scratch_size_limit);

template StatusOr<AutotuneEntry<se::dnn::FusedConvOp>> AutotuneFusedConv<float>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::FusedConvOp>>*
        autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, double leakyrelu_alpha,
    se::DeviceMemory<float> input_ptr, se::DeviceMemory<float> filter_ptr,
    se::DeviceMemory<float> output_ptr, se::DeviceMemory<float> bias_ptr,
    se::DeviceMemory<float> side_input_ptr, int64_t scratch_size_limit);

template StatusOr<AutotuneEntry<se::dnn::FusedConvOp>>
AutotuneFusedConv<Eigen::half>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::FusedConvOp>>*
        autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, double leakyrelu_alpha,
    se::DeviceMemory<Eigen::half> input_ptr,
    se::DeviceMemory<Eigen::half> filter_ptr,
    se::DeviceMemory<Eigen::half> output_ptr,
    se::DeviceMemory<Eigen::half> bias_ptr,
    se::DeviceMemory<Eigen::half> side_input_ptr, int64_t scratch_size_limit);

template <typename T>
StatusOr<AutotuneEntry<se::dnn::ConvOp>> AutotuneUnfusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<T> input_ptr, const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<T> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc, se::DeviceMemory<T> output_ptr,
    int64_t scratch_size_limit) {
  AutotuneEntry<se::dnn::ConvOp> autotune_entry;

  auto* stream = ctx->op_device_context()->stream();

  if (!autotune_map->Find(conv_parameters, &autotune_entry)) {
    profiler::ScopedAnnotation annotation("cudnn_autotuning");

#if GOOGLE_CUDA
    se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                                stream);
    se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter);

    // TODO(awpr): second-guess whether it's okay that this profiles
    // convolutions on uninitialized memory.
    switch (kind) {
      case se::dnn::ConvolutionKind::FORWARD:
      case se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION:
      case se::dnn::ConvolutionKind::FORWARD_GRAPH:
        output_ptr = se::DeviceMemory<T>(
            WrapRedzoneBestEffort(&rz_allocator, output_ptr));
        break;
      case se::dnn::ConvolutionKind::BACKWARD_DATA:
        input_ptr = se::DeviceMemory<T>(
            WrapRedzoneBestEffort(&rz_allocator, input_ptr));
        break;
      case se::dnn::ConvolutionKind::BACKWARD_FILTER:
        filter_ptr = se::DeviceMemory<T>(
            WrapRedzoneBestEffort(&rz_allocator, filter_ptr));
        break;
      default:
        return errors::InvalidArgument(
            absl::StrFormat("Unknown ConvolutionKind %d", kind));
    }

    const auto element_type = se::dnn::ToDataType<T>::value;
    std::vector<std::unique_ptr<const se::dnn::ConvRunner>> runners;
    auto dnn = stream->parent()->AsDnn();
    if (dnn == nullptr) {
      return absl::InvalidArgumentError("No DNN in stream executor.");
    }
    TF_RETURN_IF_ERROR(dnn->GetConvolveRunners(
        CudnnUseFrontend(), kind, element_type, element_type, stream,
        input_desc, input_ptr, filter_desc, filter_ptr, output_desc, output_ptr,
        conv_desc, /*use_fallback=*/false, &rz_allocator,
        GetNumericOptionsForCuDnn(), &runners));
    auto launch_func =
        [&](se::ScratchAllocator* allocator_used,
            const std::unique_ptr<const se::dnn::ConvRunner>& runner,
            se::dnn::ProfileResult* profile_result) -> Status {
      TF_ASSIGN_OR_RETURN(auto scratch, allocator_used->AllocateBytes(
                                            runner->GetWorkspaceSize()));
      return (*runner)(stream, profile_result, scratch, input_ptr, filter_ptr,
                       output_ptr);
    };
    TF_ASSIGN_OR_RETURN(auto results,
                        internal::AutotuneConvImpl(
                            ctx, runners, cudnn_use_autotune, launch_func,
                            scratch_size_limit, rz_allocator));

    LogConvAutotuneResults(kind, se::dnn::ToDataType<T>::value, input_ptr,
                           filter_ptr, output_ptr, input_desc, filter_desc,
                           output_desc, conv_desc, stream->parent(), results);

    // Two-level autotuning: Cudnn frontend supports two engine lists:
    // heuristics and fallback. Heuristics engines are normally faster.
    // To reduce autotuning time, we evaluate the fallback engines only when
    // none of the heuristics engines work.
    bool found_working_engine = false;
    for (auto& result : results) {
      if (!result.has_failure()) {
        found_working_engine = true;
        break;
      }
    }

    if (!CudnnUseFrontend() || found_working_engine) {
      TF_ASSIGN_OR_RETURN(
          autotune_entry,
          BestCudnnConvAlgorithm<se::dnn::ConvOp>(results, std::move(runners)));
    } else {
      LOG(WARNING)
          << "None of the algorithms provided by cuDNN frontend heuristics "
             "worked; trying fallback algorithms.  Conv: "
          << conv_parameters.ToString();
      std::vector<std::unique_ptr<const se::dnn::ConvRunner>> fallback_runners;
      TF_RETURN_IF_ERROR(dnn->GetConvolveRunners(
          CudnnUseFrontend(), kind, element_type, element_type, stream,
          input_desc, input_ptr, filter_desc, filter_ptr, output_desc,
          output_ptr, conv_desc, /*use_fallback=*/true, &rz_allocator,
          GetNumericOptionsForCuDnn(), &fallback_runners));

      TF_ASSIGN_OR_RETURN(auto fallback_results,
                          internal::AutotuneConvImpl(
                              ctx, fallback_runners, cudnn_use_autotune,
                              launch_func, scratch_size_limit, rz_allocator));

      LogConvAutotuneResults(kind, se::dnn::ToDataType<T>::value, input_ptr,
                             filter_ptr, output_ptr, input_desc, filter_desc,
                             output_desc, conv_desc, stream->parent(),
                             fallback_results);

      TF_ASSIGN_OR_RETURN(autotune_entry,
                          BestCudnnConvAlgorithm<se::dnn::ConvOp>(
                              fallback_results, std::move(fallback_runners)));
    }

#elif TENSORFLOW_USE_ROCM
    DnnScratchAllocator scratch_allocator(scratch_size_limit, ctx);

    std::vector<se::dnn::ProfileResult> algorithms;
    auto dnn = stream->parent()->AsDnn();
    if (dnn == nullptr) {
      return absl::InvalidArgumentError("No DNN in stream executor.");
    }
    if (!dnn->GetMIOpenConvolveAlgorithms(
            kind, se::dnn::ToDataType<T>::value, stream, input_desc, input_ptr,
            filter_desc, filter_ptr, output_desc, output_ptr, conv_desc,
            &scratch_allocator, &algorithms)) {
      return errors::Unknown(
          "Failed to get convolution algorithm. This is probably "
          "because MIOpen failed to initialize, so try looking to "
          "see if a warning log message was printed above.");
    }

    std::vector<xla::AutotuneResult> results;
    if (algorithms.size() == 1) {
      auto profile_result = algorithms[0];
      results.emplace_back();
      auto& result = results.back();
      *result.mutable_algorithm() = profile_result.algorithm().ToProto();

      result.set_scratch_bytes(profile_result.scratch_size());
      *result.mutable_run_time() = proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));
    } else {
      for (auto miopen_algorithm : algorithms) {
        auto profile_algorithm = miopen_algorithm.algorithm();
        se::dnn::ProfileResult profile_result;
        auto miopen_launch_status = dnn->ConvolveWithAlgorithm(
            stream, kind, input_desc, input_ptr, filter_desc, filter_ptr,
            output_desc, output_ptr, conv_desc, &scratch_allocator,
            se::dnn::AlgorithmConfig(profile_algorithm,
                                     miopen_algorithm.scratch_size()),
            &profile_result);
        if (miopen_launch_status.ok() && profile_result.is_valid()) {
          results.emplace_back();
          auto& result = results.back();
          *result.mutable_algorithm() = profile_algorithm.ToProto();

          result.set_scratch_bytes(scratch_allocator.TotalByteSize());
          *result.mutable_run_time() = proto_utils::ToDurationProto(
              absl::Milliseconds(profile_result.elapsed_time_in_ms()));
        }
      }
    }
    LogConvAutotuneResults(kind, se::dnn::ToDataType<T>::value, input_ptr,
                           filter_ptr, output_ptr, input_desc, filter_desc,
                           output_desc, conv_desc, stream->parent(), results);

    TF_ASSIGN_OR_RETURN(auto algo_desc, BestCudnnConvAlgorithm(results));
    autotune_entry = AutotuneEntry<se::dnn::ConvOp>(algo_desc);
#endif

    autotune_map->Insert(conv_parameters, autotune_entry);
  }

  return autotune_entry;
}

#define DECLARE_GPU_SPEC(T)                                                 \
  template StatusOr<AutotuneEntry<se::dnn::ConvOp>> AutotuneUnfusedConv<T>( \
      bool cudnn_use_autotune,                                              \
      AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>*          \
          autotune_map,                                                     \
      const ConvParameters& conv_parameters, OpKernelContext* ctx,          \
      se::dnn::ConvolutionKind kind,                                        \
      const se::dnn::BatchDescriptor& input_desc,                           \
      se::DeviceMemory<T> input_ptr,                                        \
      const se::dnn::FilterDescriptor& filter_desc,                         \
      se::DeviceMemory<T> filter_ptr,                                       \
      const se::dnn::ConvolutionDescriptor& conv_desc,                      \
      const se::dnn::BatchDescriptor& output_desc,                          \
      se::DeviceMemory<T> output_ptr, int64_t scratch_size_limit);

DECLARE_GPU_SPEC(double);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(Eigen::bfloat16);

#undef DECLARE_GPU_SPEC

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
