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
#include "tensorflow/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

#if GOOGLE_CUDA
template <typename LaunchFunc>
StatusOr<std::vector<tensorflow::AutotuneResult>> AutotuneConvImpl(
    OpKernelContext* ctx, const std::vector<se::dnn::AlgorithmConfig>& configs,
    const LaunchFunc& launch_func, size_t scratch_size_limit,
    const se::RedzoneAllocator& rz_allocator) {
  auto* stream = ctx->op_device_context()->stream();

  se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                              stream);

  std::vector<tensorflow::AutotuneResult> results;
  // TODO(reedwm): Warn if determinism is enabled after autotune is run
  for (const auto& profile_config : configs) {
    // TODO(zhengxq): profile each algorithm multiple times to better
    // accuracy.
    se::RedzoneAllocator rz_scratch_allocator(
        stream, &tf_allocator_adapter, se::GpuAsmOpts(),
        /*memory_limit=*/scratch_size_limit);
    DnnScratchAllocator scratch_allocator(scratch_size_limit, ctx);
    se::ScratchAllocator* allocator_used =
        !RedzoneCheckDisabled()
            ? static_cast<se::ScratchAllocator*>(&rz_scratch_allocator)
            : static_cast<se::ScratchAllocator*>(&scratch_allocator);

    se::dnn::ProfileResult profile_result;
    Status cudnn_launch_status =
        launch_func(allocator_used, profile_config, &profile_result);

    if (cudnn_launch_status.ok() && profile_result.is_valid()) {
      results.emplace_back();
      auto& result = results.back();
      if (CudnnUseFrontend()) {
        result.mutable_cuda_conv_plan()->set_exec_plan_id(
            profile_config.algorithm()->exec_plan_id());
      } else {
        result.mutable_conv()->set_algorithm(
            profile_config.algorithm()->algo_id());
        result.mutable_conv()->set_tensor_ops_enabled(
            profile_config.algorithm()->tensor_ops_enabled());
      }

      result.set_scratch_bytes(
          !RedzoneCheckDisabled()
              ? rz_scratch_allocator.TotalAllocatedBytesExcludingRedzones()
              : scratch_allocator.TotalByteSize());
      *result.mutable_run_time() = proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));

      CheckRedzones(rz_scratch_allocator, &result);
      CheckRedzones(rz_allocator, &result);
    } else if (CudnnUseFrontend()) {
      // When CuDNN frontend APIs are used, we need to make sure the profiling
      // results are one-to-one mapping of the "plans". So, we insert dummy
      // results when the execution fails.
      results.emplace_back();
      auto& result = results.back();
      result.mutable_failure()->set_kind(AutotuneResult::UNKNOWN);
      result.mutable_failure()->set_msg(
          absl::StrCat("Profiling failure on CUDNN engine: ",
                       profile_config.algorithm()->exec_plan_id()));
    }
  }

  return results;
}
#endif  // GOOGLE_CUDA

// Finds the best convolution algorithm for the given ConvLaunch (cuda
// convolution on the stream) and parameters, by running all possible
// algorithms and measuring execution time.
// TODO(ezhulenev): Move it to conv_ops_gpu.h and share with conv_ops.cc.
template <typename T>
StatusOr<se::dnn::AlgorithmConfig> AutotuneFusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, se::dnn::AlgorithmConfig>* autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, se::DeviceMemory<T> input_ptr,
    se::DeviceMemory<T> filter_ptr, se::DeviceMemory<T> output_ptr,
    se::DeviceMemory<T> bias_ptr, se::DeviceMemory<T> side_input_ptr,
    int64_t scratch_size) {
#if GOOGLE_CUDA
  se::dnn::AlgorithmConfig algorithm_config;

  if (cudnn_use_autotune) {
    // Check if we already have an algorithm selected for the given parameters.
    if (autotune_map->Find(params, &algorithm_config)) {
      return algorithm_config;
    }
    profiler::ScopedAnnotation trace("cudnn_autotuning");

    auto* stream = ctx->op_device_context()->stream();

    // Find all candidate algorithms or execution plans (for CuDNN frontend
    // APIs).
    std::vector<std::unique_ptr<se::dnn::ConvolveExecutionPlan>> plans;
    std::vector<se::dnn::AlgorithmDesc> algorithms;
    std::vector<se::dnn::AlgorithmConfig> configs;
    if (CudnnUseFrontend()) {
      if (!stream->parent()
               ->GetFusedConvolveExecutionPlans(
                   se::dnn::ConvolutionKind::FORWARD,
                   se::dnn::ToDataType<T>::value, conv_scale, side_input_scale,
                   stream, input_desc, filter_desc, bias_desc, output_desc,
                   conv_desc, activation_mode, &plans)
               .ok()) {
        return errors::Unknown(
            "Failed to get convolution plans. This is probably because cuDNN "
            "failed to initialize, so try looking to see if a warning log "
            "message was printed above.");
      }
      for (const auto& plan : plans) {
        configs.push_back(se::dnn::AlgorithmConfig(
            se::dnn::AlgorithmDesc{plan->getTag(), plan->get_raw_desc()},
            plan->getWorkspaceSize()));
      }
    } else {
      if (!stream->parent()->GetConvolveAlgorithms(
              se::dnn::ConvolutionKind::FORWARD, &algorithms)) {
        return errors::Unknown(
            "Failed to get convolution algorithm. This is probably because "
            "cuDNN failed to initialize, so try looking to see if a warning "
            "log message was printed above.");
      }
      for (const auto& algorithm : algorithms) {
        configs.push_back(se::dnn::AlgorithmConfig(algorithm));
      }
    }

    se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                                stream);
    se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter,
                                      se::GpuAsmOpts());
    se::DeviceMemory<T> output_ptr_rz(
        WrapRedzoneBestEffort(&rz_allocator, output_ptr));

    auto launch_func = [&](se::ScratchAllocator* allocator_used,
                           se::dnn::AlgorithmConfig profile_config,
                           se::dnn::ProfileResult* profile_result) -> Status {
      if (CudnnUseFrontend()) {
        return stream->FusedConvolveWithExecutionPlan(
            input_desc, input_ptr,             // input
            conv_scale,                        // input_scale
            filter_desc, filter_ptr,           // filter
            conv_desc,                         // conv
            side_input_ptr, side_input_scale,  // side_input
            bias_desc, bias_ptr,               // bias
            activation_mode,                   // activation
            output_desc, &output_ptr_rz,       // output
            allocator_used, profile_config, profile_result);
      } else {
        return stream->FusedConvolveWithAlgorithm(
            input_desc, input_ptr,             // input
            conv_scale,                        // input_scale
            filter_desc, filter_ptr,           // filter
            conv_desc,                         // conv
            side_input_ptr, side_input_scale,  // side_input
            bias_desc, bias_ptr,               // bias
            activation_mode,                   // activation
            output_desc, &output_ptr_rz,       // output
            allocator_used, profile_config, profile_result);
      }
    };

    SE_ASSIGN_OR_RETURN(
        auto results, AutotuneConvImpl(ctx, configs, launch_func, scratch_size,
                                       rz_allocator));

    // Only log on an AutotuneConv cache miss.
    LogFusedConvForwardAutotuneResults(
        se::dnn::ToDataType<T>::value, input_ptr, filter_ptr, output_ptr,
        bias_ptr, side_input_ptr, input_desc, filter_desc, output_desc,
        conv_desc, conv_scale, side_input_scale, activation_mode,
        stream->parent(), results);
    TF_RETURN_IF_ERROR(
        BestCudnnConvAlgorithm(results, &plans, &algorithm_config));
    autotune_map->Insert(params, algorithm_config);
  }
  return algorithm_config;
#else
  return errors::Unimplemented(
      "Fused conv not implemented on non-CUDA platforms.");
#endif
}

template StatusOr<se::dnn::AlgorithmConfig> AutotuneFusedConv<double>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, se::dnn::AlgorithmConfig>* autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, se::DeviceMemory<double> input_ptr,
    se::DeviceMemory<double> filter_ptr, se::DeviceMemory<double> output_ptr,
    se::DeviceMemory<double> bias_ptr, se::DeviceMemory<double> side_input_ptr,
    int64_t scratch_size);

template StatusOr<se::dnn::AlgorithmConfig> AutotuneFusedConv<float>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, se::dnn::AlgorithmConfig>* autotune_map,
    const ConvParameters& params, OpKernelContext* ctx,
    const se::dnn::BatchDescriptor& input_desc,
    const se::dnn::FilterDescriptor& filter_desc,
    const se::dnn::BatchDescriptor& bias_desc,
    const se::dnn::BatchDescriptor& output_desc,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::ActivationMode activation_mode, double conv_scale,
    double side_input_scale, se::DeviceMemory<float> input_ptr,
    se::DeviceMemory<float> filter_ptr, se::DeviceMemory<float> output_ptr,
    se::DeviceMemory<float> bias_ptr, se::DeviceMemory<float> side_input_ptr,
    int64_t scratch_size);

template <typename T>
StatusOr<se::dnn::AlgorithmConfig> AutotuneUnfusedConv(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, se::dnn::AlgorithmConfig>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<T> input_ptr, const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<T> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc, se::DeviceMemory<T> output_ptr,
    int64_t scratch_size_limit) {
  se::dnn::AlgorithmConfig algorithm_config;

  // cudnn_use_autotune is applicable only the CUDA flow
  // for ROCm/MIOpen, we need to call GetMIOpenConvolveAlgorithms explicitly
  // if we do not have a cached algorithm_config for this conv_parameters
  bool do_autotune =
#if TENSORFLOW_USE_ROCM
      true ||
#endif
      cudnn_use_autotune;
  if (do_autotune && !autotune_map->Find(conv_parameters, &algorithm_config)) {
    profiler::ScopedAnnotation annotation("cudnn_autotuning");

    auto* stream = ctx->op_device_context()->stream();

    std::vector<std::unique_ptr<se::dnn::ConvolveExecutionPlan>> plans;
#if GOOGLE_CUDA
    std::vector<se::dnn::AlgorithmDesc> algorithms;
    std::vector<se::dnn::AlgorithmConfig> configs;
    const auto get_algo_failed_error = errors::Unknown(
        "Failed to get convolution algorithm. This is probably because cuDNN "
        "failed to initialize, so try looking to see if a warning log message "
        "was printed above.");

    if (CudnnUseFrontend()) {
      if (!stream->parent()->GetConvolveExecutionPlans(
              kind, se::dnn::ToDataType<T>::value, stream, input_desc,
              filter_desc, output_desc, conv_desc, &plans)) {
        return get_algo_failed_error;
      }
      for (const auto& plan : plans) {
        configs.push_back(se::dnn::AlgorithmConfig(
            se::dnn::AlgorithmDesc{plan->getTag(), plan->get_raw_desc()},
            plan->getWorkspaceSize()));
      }
    } else {
      if (!stream->parent()->GetConvolveAlgorithms(kind, &algorithms)) {
        return get_algo_failed_error;
      }
      for (const auto& algorithm : algorithms) {
        configs.push_back(se::dnn::AlgorithmConfig(algorithm));
      }
    }

    se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                                stream);
    se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter,
                                      se::GpuAsmOpts());

    // TODO(awpr): second-guess whether it's okay that this profiles
    // convolutions on uninitialized memory.
    switch (kind) {
      case se::dnn::ConvolutionKind::FORWARD:
      case se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION:
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

    auto launch_func = [&](se::ScratchAllocator* allocator_used,
                           se::dnn::AlgorithmConfig profile_config,
                           se::dnn::ProfileResult* profile_result) -> Status {
      if (CudnnUseFrontend()) {
        return stream->ConvolveWithExecutionPlan(
            kind, input_desc, input_ptr, filter_desc, filter_ptr, output_desc,
            output_ptr, conv_desc, allocator_used, profile_config,
            profile_result);
      } else {
        return stream->ConvolveWithAlgorithm(
            kind, input_desc, input_ptr, filter_desc, filter_ptr, output_desc,
            output_ptr, conv_desc, allocator_used, profile_config,
            profile_result);
      }
    };

    SE_ASSIGN_OR_RETURN(auto results,
                        AutotuneConvImpl(ctx, configs, launch_func,
                                         scratch_size_limit, rz_allocator));
#elif TENSORFLOW_USE_ROCM
    DnnScratchAllocator scratch_allocator(scratch_size_limit, ctx);

    std::vector<se::dnn::ProfileResult> algorithms;
    if (!stream->parent()->GetMIOpenConvolveAlgorithms(
            kind, se::dnn::ToDataType<T>::value, stream, input_desc, input_ptr,
            filter_desc, filter_ptr, output_desc, output_ptr, conv_desc,
            &scratch_allocator, &algorithms)) {
      return errors::Unknown(
          "Failed to get convolution algorithm. This is probably "
          "because MIOpen failed to initialize, so try looking to "
          "see if a warning log message was printed above.");
    }

    std::vector<tensorflow::AutotuneResult> results;
    if (algorithms.size() == 1) {
      auto profile_result = algorithms[0];
      results.emplace_back();
      auto& result = results.back();
      result.mutable_conv()->set_algorithm(
          profile_result.algorithm().algo_id());
      result.mutable_conv()->set_tensor_ops_enabled(
          profile_result.algorithm().tensor_ops_enabled());

      result.set_scratch_bytes(profile_result.scratch_size());
      *result.mutable_run_time() = proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));
    } else {
      for (auto miopen_algorithm : algorithms) {
        auto profile_algorithm = miopen_algorithm.algorithm();
        se::dnn::ProfileResult profile_result;
        auto miopen_launch_status = stream->ConvolveWithAlgorithm(
            kind, input_desc, input_ptr, filter_desc, filter_ptr, output_desc,
            output_ptr, conv_desc, &scratch_allocator,
            se::dnn::AlgorithmConfig(profile_algorithm,
                                     miopen_algorithm.scratch_size()),
            &profile_result);
        if (miopen_launch_status.ok() && profile_result.is_valid()) {
          results.emplace_back();
          auto& result = results.back();
          result.mutable_conv()->set_algorithm(profile_algorithm.algo_id());
          result.mutable_conv()->set_tensor_ops_enabled(
              profile_algorithm.tensor_ops_enabled());

          result.set_scratch_bytes(scratch_allocator.TotalByteSize());
          *result.mutable_run_time() = proto_utils::ToDurationProto(
              absl::Milliseconds(profile_result.elapsed_time_in_ms()));
        }
      }
    }
#endif
    LogConvAutotuneResults(kind, se::dnn::ToDataType<T>::value, input_ptr,
                           filter_ptr, output_ptr, input_desc, filter_desc,
                           output_desc, conv_desc, stream->parent(), results);

    SE_RETURN_IF_ERROR(
        BestCudnnConvAlgorithm(results, &plans, &algorithm_config));

    autotune_map->Insert(conv_parameters, algorithm_config);
  }
  return algorithm_config;
}

template StatusOr<se::dnn::AlgorithmConfig> AutotuneUnfusedConv<double>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, se::dnn::AlgorithmConfig>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<double> input_ptr,
    const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<double> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc,
    se::DeviceMemory<double> output_ptr, int64_t scratch_size_limit);

template StatusOr<se::dnn::AlgorithmConfig> AutotuneUnfusedConv<float>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, se::dnn::AlgorithmConfig>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<float> input_ptr,
    const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<float> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc,
    se::DeviceMemory<float> output_ptr, int64_t scratch_size_limit);

template StatusOr<se::dnn::AlgorithmConfig> AutotuneUnfusedConv<Eigen::half>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, se::dnn::AlgorithmConfig>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<Eigen::half> input_ptr,
    const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<Eigen::half> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc,
    se::DeviceMemory<Eigen::half> output_ptr, int64_t scratch_size_limit);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
