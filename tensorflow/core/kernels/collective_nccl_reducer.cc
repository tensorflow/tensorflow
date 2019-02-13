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
#include "tensorflow/core/kernels/collective_nccl_reducer.h"

#ifdef GOOGLE_CUDA

#include "tensorflow/core/common_runtime/collective_util.h"
#include "tensorflow/core/nccl/nccl_manager.h"

namespace tensorflow {
namespace {
string NcclCollectiveKey(const string& exec_key, int step_id) {
  return strings::StrCat(exec_key, ":", step_id);
}
}  // namespace

NcclReducer::NcclReducer() : col_ctx_(nullptr), col_params_(nullptr) {}

Status NcclReducer::InitializeCollectiveParams(CollectiveParams* col_params) {
  if (col_params->instance.type != REDUCTION_COLLECTIVE ||
      col_params->instance.impl_details.collective_name != "NcclReduce") {
    return errors::Internal("Unexpected collective type ",
                            col_params->instance.type, " expected ",
                            REDUCTION_COLLECTIVE, "; or collective name ",
                            col_params->instance.impl_details.collective_name,
                            " expected NcclReduce");
  } else {
    return Status::OK();
  }
}

Status NcclReducer::InitializeCollectiveContext(CollectiveContext* col_ctx) {
  col_ctx_ = col_ctx;
  col_params_ = &col_ctx->col_params;
  return collective_util::InitializeDeviceAndLocality(
      col_ctx->dev_mgr, col_ctx->device_name, &col_ctx->device,
      &col_ctx->device_locality);
}

Status NcclReducer::InitializeInstanceBeforeGroupDiscovery(
    CollectiveParams* col_params) {
  if (col_params->default_rank == 0 && col_params->group.num_tasks > 1) {
    col_params->instance.communicator_key =
        NcclManager::instance()->GenerateCommunicatorKey();
  }
  return Status::OK();
}

Status ReductionOp(const string& merge_op, ncclRedOp_t* reduction_op) {
  if (merge_op == "Add") {
    *reduction_op = ncclSum;
    return Status::OK();
  } else if (merge_op == "Mul") {
    *reduction_op = ncclProd;
    return Status::OK();
  } else {
    return errors::Internal("Expected merge_op to be either Add or Mul, found ",
                            merge_op);
  }
}

void NcclReducer::Run(StatusCallback done) {
  ncclRedOp_t reduction_op;
  Status s = ReductionOp(col_params_->merge_op->type_string(), &reduction_op);
  if (!s.ok()) {
    done(s);
    return;
  }

  Tensor group_size;
  Notification group_size_ready;
  Status group_size_status;
  if (col_params_->final_op) {
    // Create an on-device scalar value from group_size_.
    // TODO(ayushd, tucker): avoid this copy by either reusing across
    // invocations or providing the scalar to the kernel in host memory.
    Tensor group_size_val(col_ctx_->output->dtype(), TensorShape({}));
    switch (col_ctx_->output->dtype()) {
      case DT_FLOAT:
        group_size_val.scalar<float>()() = col_params_->group.group_size;
        break;
      case DT_DOUBLE:
        group_size_val.scalar<double>()() = col_params_->group.group_size;
        break;
      case DT_INT32:
        group_size_val.scalar<int32>()() = col_params_->group.group_size;
        break;
      case DT_INT64:
        group_size_val.scalar<int64>()() = col_params_->group.group_size;
        break;
      default:
        done(errors::Internal("Unsupported type ", col_ctx_->output->dtype()));
        return;
    }
    group_size = Tensor(
        col_ctx_->device->GetAllocator(col_ctx_->op_ctx->input_alloc_attr(0)),
        col_ctx_->output->dtype(), TensorShape({}));
    DeviceContext* op_dev_ctx = col_ctx_->op_ctx->op_device_context();
    // Enqueue copy on gpu stream.
    op_dev_ctx->CopyCPUTensorToDevice(
        &group_size_val, col_ctx_->device, &group_size,
        [&group_size_ready, &group_size_status](const Status& s) {
          group_size_status = s;
          group_size_ready.Notify();
        });
  } else {
    group_size_ready.Notify();
  }

  Notification nccl_done;
  Status nccl_status;
  auto* compute_stream = col_ctx_->op_ctx->op_device_context()->stream();
  auto* gpu_info = col_ctx_->op_ctx->device()->tensorflow_gpu_device_info();
  // `AddToAllReduce` performs consistency checks for the NCCL call and enqueues
  // the `Participant` struct locally.  When all local participants with this
  // `nccl_collective_key` have called `AddToAllReduce` and
  // `SignalMultiNodeReady`, all devices at this worker are ready to process
  // this NCCL op.
  //
  // The `NcclManager` uses a dedicated CUDA stream for NCCL kernels.  At this
  // point, it synchronizes the NCCL stream with the compute stream, and then
  // enqueues the NCCL kernel on the NCCL stream.
  const int num_global_devices = col_params_->group.group_size;
  const int num_local_devices = col_params_->instance.num_devices_per_task.at(
      col_params_->instance.task_names[col_params_->default_rank]);
  const string nccl_collective_key =
      NcclCollectiveKey(col_ctx_->exec_key, col_ctx_->step_id);
  auto done_callback = [&nccl_done, &nccl_status](const Status& s) {
    nccl_status = s;
    nccl_done.Notify();
  };
  auto participant = absl::make_unique<NcclManager::Participant>(
      compute_stream->parent(), compute_stream, gpu_info->event_mgr,
      gpu_info->gpu_id, col_ctx_->input, col_ctx_->output,
      col_params_->default_rank, std::move(done_callback));
  VLOG(1) << "NcclReducer calling NcclManager::AddToAllReduce num_tasks "
          << col_params_->group.num_tasks << " current task "
          << col_params_->instance.task_names[col_params_->default_rank]
          << " num local devices " << num_local_devices
          << " num global devices " << num_global_devices << " device "
          << col_ctx_->device_name << " instance "
          << col_params_->instance.instance_key;
  NcclManager::instance()->AddToAllReduce(
      std::move(participant),
      {nccl_collective_key, num_local_devices, num_global_devices,
       col_params_->instance.communicator_key},
      reduction_op);

  // NOTE(ayushd): We need to synchronize NCCL launches across nodes to prevent
  // deadlocks.  In the current implementation, we define a deterministic
  // sequential launch order between potentially concurrent collective instances
  // by introducing control information during static graph analysis in
  // graph/collective_order.cc.  This can be either in the form of explicit
  // control edges or via `wait_for` attribute on the collective op.
  //
  // The other end of the design spectrum would have a distinguished node
  // dynamically signal the next collective to launch to all other participants.
  // This has higher degree of runtime coordination, but it may be able to
  // achieve better performance if the (arbitrary) static execution order
  // assigned in the first approach turns out to not be good from a scheduling
  // perspective.  e.g. consider a graph in which c1, c2, and c3 are three
  // concurrent collective instances, and the static ordering assigns c1 -> c2
  // -> c3.  In practice, it could turn out that c3 is always ready to execute
  // before c1 or c2.
  //
  // `WaitForDependencies` may block if the collective instances on which this
  // op depends have not yet launched.  When this function returns, this op is
  // ready to go.
  col_ctx_->col_exec->WaitForDependencies(*col_params_);
  NcclManager::instance()->SignalMultiNodeReady(nccl_collective_key);
  // When all devices at this worker have called `SignalMultiNodeReady`, the
  // `NcclManager` will enqueue the NCCL kernel on the NCCL stream.  Thus the
  // implementation of `Launched` keeps track of the number of devices that have
  // launched.
  col_ctx_->col_exec->Launched(*col_params_);

  // Wait for nccl op and group_size copy to succeed, then do final_op.
  group_size_ready.WaitForNotification();
  nccl_done.WaitForNotification();
  Status final_status =
      group_size_status.ok() ? nccl_status : group_size_status;
  if (final_status.ok() && col_params_->final_op) {
    final_status = collective_util::ComputeBinOp(
        col_ctx_->op_ctx, col_ctx_->op_params, col_ctx_->device,
        col_params_->final_op.get(), col_ctx_->output, &group_size);
  }
  done(final_status);
}

REGISTER_COLLECTIVE(NcclReduce, NcclReducer);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
