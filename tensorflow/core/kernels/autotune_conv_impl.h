/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_AUTOTUNE_CONV_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_AUTOTUNE_CONV_IMPL_H_

#if GOOGLE_CUDA
#define EIGEN_USE_THREADS

#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/util/proto/proto_utils.h"

namespace tensorflow::internal {

template <typename LaunchFunc, typename Sig>
StatusOr<std::vector<xla::AutotuneResult>> AutotuneConvImpl(
    OpKernelContext* ctx,
    std::vector<std::unique_ptr<const se::dnn::OpRunner<Sig>>>& runners,
    bool actually_do_autotune, const LaunchFunc& launch_func,
    size_t scratch_size_limit, const se::RedzoneAllocator& rz_allocator) {
  auto* stream = ctx->op_device_context()->stream();

  se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                              stream);

  std::vector<xla::AutotuneResult> results;
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
      result.mutable_failure()->set_kind(xla::AutotuneResult::UNKNOWN);
      result.mutable_failure()->set_msg(
          absl::StrCat("Profiling failure on CUDNN engine ", desc.ToString(),
                       ": ", cudnn_launch_status.ToString()));
    }
  }

  return results;
}

}  // namespace tensorflow::internal

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_AUTOTUNE_CONV_IMPL_H_
