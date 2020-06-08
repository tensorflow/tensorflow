/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/collective_nccl_gatherer.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/nccl/nccl_manager.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

void NcclGatherer::Run(StatusCallback done) {
  auto* compute_stream = col_ctx_->op_ctx->op_device_context()->stream();
  auto* gpu_info = col_ctx_->op_ctx->device()->tensorflow_gpu_device_info();
  const int num_global_devices = col_params_->group.group_size;
  const int num_local_devices = col_params_->instance.num_devices_per_task.at(
      col_params_->instance.task_names[col_params_->default_rank]);
  string nccl_collective_key =
      NcclCollectiveKey(col_ctx_->exec_key, col_ctx_->step_id);
  auto participant = absl::make_unique<NcclManager::Participant>(
      compute_stream->parent(), compute_stream, gpu_info, col_ctx_->input,
      col_ctx_->output, col_params_->default_rank, std::move(done));
  VLOG(1) << "NcclGatherer calling NcclManager::AddToAllGather num_tasks "
          << col_params_->group.num_tasks << " current task "
          << col_params_->instance.task_names[col_params_->default_rank]
          << " num local devices " << num_local_devices
          << " num global devices " << num_global_devices << " rank "
          << col_params_->default_rank << " device " << col_ctx_->device_name
          << " instance " << col_params_->instance.instance_key;
  NcclManager::instance()->AddToAllGather(
      std::move(participant),
      {std::move(nccl_collective_key), num_local_devices, num_global_devices,
       col_params_->group.runtime_details.communicator_key,
       /*source_rank=*/-1});
  {
    // `WaitForDependencies` may block if the collective instances on which this
    // op depends have not yet launched.  When this function returns, this op is
    // ready to go.
    profiler::TraceMe activity("WaitForDependencies",
                               profiler::TraceMeLevel::kInfo);
    col_ctx_->col_exec->WaitForDependencies(*col_params_);
    NcclManager::instance()->SignalMultiNodeReady(nccl_collective_key);
  }
  {
    // When all devices at this worker have called `SignalMultiNodeReady`, the
    // `NcclManager` will enqueue the NCCL kernel on the NCCL stream.  Thus the
    // implementation of `UnblockDependencies` keeps track of the number of
    // devices that have launched.
    profiler::TraceMe activity("Schedule", profiler::TraceMeLevel::kInfo);
    col_ctx_->col_exec->UnblockDependencies(*col_params_);
  }
}

REGISTER_COLLECTIVE(NcclGather, NcclGatherer);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
