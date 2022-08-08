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
namespace {

template <typename LaunchFunc, typename Sig>
StatusOr<std::vector<tensorflow::AutotuneResult>> AutotuneConvImpl(
    OpKernelContext* ctx,
    std::vector<std::unique_ptr<const se::dnn::OpRunner<Sig>>>& runners,
    bool actually_do_autotune, const LaunchFunc& launch_func,
    size_t scratch_size_limit, const se::RedzoneAllocator& rz_allocator) {
  auto* stream = ctx->op_device_context()->stream();

  se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                              stream);

  std::vector<tensorflow::AutotuneResult> results;
  // TODO(reedwm): Warn if determinism is enabled after autotune is run
  for (auto& runner : runners) {
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

    TF_ASSIGN_OR_RETURN(auto desc, runner->ToAlgorithmDesc());
    se::dnn::ProfileResult profile_result;
    Status cudnn_launch_status =
        actually_do_autotune
            ? launch_func(allocator_used, runner, &profile_result)
            : OkStatus();
    if (!actually_do_autotune) {
      // Make the result valid according to `is_valid`.
      profile_result.set_algorithm(desc);
      profile_result.set_elapsed_time_in_ms(0);
    }

    // We need to make sure the profiling results are one-to-one with the
    // "runners". So, we insert dummy results when the execution fails.
    results.emplace_back();
    auto& result = results.back();
    *result.mutable_algorithm() = desc.ToProto();
    if (cudnn_launch_status.ok() && profile_result.is_valid()) {
      result.set_scratch_bytes(
          !RedzoneCheckDisabled()
              ? rz_scratch_allocator.TotalAllocatedBytesExcludingRedzones()
              : scratch_allocator.TotalByteSize());
      *result.mutable_run_time() = proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));

      CheckRedzones(rz_scratch_allocator, &result);
      CheckRedzones(rz_allocator, &result);
    } else {
      result.mutable_failure()->set_kind(AutotuneResult::UNKNOWN);
      result.mutable_failure()->set_msg(
          absl::StrCat("Profiling failure on CUDNN engine ", desc.ToString(),
                       ": ", cudnn_launch_status.ToString()));
    }
  }

  return results;
}
}  // namespace
#endif  // GOOGLE_CUDA

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
    double side_input_scale, se::DeviceMemory<T> input_ptr,
    se::DeviceMemory<T> filter_ptr, se::DeviceMemory<T> output_ptr,
    se::DeviceMemory<T> bias_ptr, se::DeviceMemory<T> side_input_ptr,
    int64_t scratch_size_limit) {
#if GOOGLE_CUDA
  AutotuneEntry<se::dnn::FusedConvOp> autotune_entry;
  auto* stream = ctx->op_device_context()->stream();

  if (!autotune_map->Find(params, &autotune_entry)) {
    profiler::ScopedAnnotation trace("cudnn_autotuning");

    se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                                stream);
    se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter,
                                      se::GpuAsmOpts());
    se::DeviceMemory<T> output_ptr_rz(
        WrapRedzoneBestEffort(&rz_allocator, output_ptr));

    std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>> runners;
    auto element_type = se::dnn::ToDataType<T>::value;
    TF_RETURN_IF_ERROR(stream->parent()->GetFusedConvolveRunners(
        CudnnUseFrontend(), se::dnn::ConvolutionKind::FORWARD, element_type,
        element_type, element_type, conv_scale, side_input_scale, stream,
        input_desc, filter_desc, bias_desc, output_desc, conv_desc,
        /*use_fallback=*/false, activation_mode, &runners));

    auto launch_func =
        [&](se::ScratchAllocator* allocator_used,
            const std::unique_ptr<const se::dnn::FusedConvRunner>& runner,
            se::dnn::ProfileResult* profile_result) -> Status {
      TF_ASSIGN_OR_RETURN(auto scratch, allocator_used->AllocateBytes(
                                            runner->GetWorkspaceSize()));
      return (*runner)(stream, profile_result, scratch, input_ptr, filter_ptr,
                       side_input_ptr, bias_ptr, output_ptr_rz);
    };

    TF_ASSIGN_OR_RETURN(
        auto results,
        AutotuneConvImpl(ctx, runners, cudnn_use_autotune, launch_func,
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
      TF_RETURN_IF_ERROR(stream->parent()->GetFusedConvolveRunners(
          CudnnUseFrontend(), se::dnn::ConvolutionKind::FORWARD, element_type,
          element_type, element_type, conv_scale, side_input_scale, stream,
          input_desc, filter_desc, bias_desc, output_desc, conv_desc,
          /*use_fallback=*/true, activation_mode, &fallback_runners));

      TF_ASSIGN_OR_RETURN(
          auto fallback_results,
          AutotuneConvImpl(ctx, fallback_runners, cudnn_use_autotune,
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
    double side_input_scale, se::DeviceMemory<double> input_ptr,
    se::DeviceMemory<double> filter_ptr, se::DeviceMemory<double> output_ptr,
    se::DeviceMemory<double> bias_ptr, se::DeviceMemory<double> side_input_ptr,
    int64_t scratch_size_limit);

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
    double side_input_scale, se::DeviceMemory<float> input_ptr,
    se::DeviceMemory<float> filter_ptr, se::DeviceMemory<float> output_ptr,
    se::DeviceMemory<float> bias_ptr, se::DeviceMemory<float> side_input_ptr,
    int64_t scratch_size_limit);

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
    double side_input_scale, se::DeviceMemory<Eigen::half> input_ptr,
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

    const auto element_type = se::dnn::ToDataType<T>::value;
    std::vector<std::unique_ptr<const se::dnn::ConvRunner>> runners;
    TF_RETURN_IF_ERROR(stream->parent()->GetConvolveRunners(
        CudnnUseFrontend(), kind, element_type, element_type, stream,
        input_desc, input_ptr, filter_desc, filter_ptr, output_desc, output_ptr,
        conv_desc, /*use_fallback=*/false, &rz_allocator, &runners));
    auto launch_func =
        [&](se::ScratchAllocator* allocator_used,
            const std::unique_ptr<const se::dnn::ConvRunner>& runner,
            se::dnn::ProfileResult* profile_result) -> Status {
      TF_ASSIGN_OR_RETURN(auto scratch, allocator_used->AllocateBytes(
                                            runner->GetWorkspaceSize()));
      return (*runner)(stream, profile_result, scratch, input_ptr, filter_ptr,
                       output_ptr);
    };
    TF_ASSIGN_OR_RETURN(
        auto results,
        AutotuneConvImpl(ctx, runners, cudnn_use_autotune, launch_func,
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
      TF_RETURN_IF_ERROR(stream->parent()->GetConvolveRunners(
          CudnnUseFrontend(), kind, element_type, element_type, stream,
          input_desc, input_ptr, filter_desc, filter_ptr, output_desc,
          output_ptr, conv_desc, /*use_fallback=*/true, &rz_allocator,
          &fallback_runners));

      TF_ASSIGN_OR_RETURN(
          auto fallback_results,
          AutotuneConvImpl(ctx, fallback_runners, cudnn_use_autotune,
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
      *result.mutable_algorithm() = profile_result.algorithm().ToProto();

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

template StatusOr<AutotuneEntry<se::dnn::ConvOp>> AutotuneUnfusedConv<double>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<double> input_ptr,
    const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<double> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc,
    se::DeviceMemory<double> output_ptr, int64_t scratch_size_limit);

template StatusOr<AutotuneEntry<se::dnn::ConvOp>> AutotuneUnfusedConv<float>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
    const ConvParameters& conv_parameters, OpKernelContext* ctx,
    se::dnn::ConvolutionKind kind, const se::dnn::BatchDescriptor& input_desc,
    se::DeviceMemory<float> input_ptr,
    const se::dnn::FilterDescriptor& filter_desc,
    se::DeviceMemory<float> filter_ptr,
    const se::dnn::ConvolutionDescriptor& conv_desc,
    const se::dnn::BatchDescriptor& output_desc,
    se::DeviceMemory<float> output_ptr, int64_t scratch_size_limit);

template StatusOr<AutotuneEntry<se::dnn::ConvOp>>
AutotuneUnfusedConv<Eigen::half>(
    bool cudnn_use_autotune,
    AutotuneMap<ConvParameters, AutotuneEntry<se::dnn::ConvOp>>* autotune_map,
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
