/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/nccl/collective_communicator.h"

#include "tensorflow/core/framework/cancellation.h"

#if TENSORFLOW_USE_NCCL && (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)

#include "absl/memory/memory.h"
#include "tensorflow/core/nccl/nccl_manager.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

class NcclCommunicator : public NcclCommunicatorInterface {
 public:
  string GenerateCommunicatorKey() override {
    return nccl_manager_.GenerateCommunicatorKey();
  }

  void Enqueue(std::shared_ptr<CollectiveContext> col_ctx,
               StatusCallback done) override;

  void StartAbort(const Status& s) override;

 private:
  NcclManager nccl_manager_;
};

namespace {
Status ReductionOp(const string& merge_op, ncclRedOp_t* reduction_op) {
  if (merge_op == "Add") {
    *reduction_op = ncclSum;
    return Status::OK();
  } else if (merge_op == "Mul") {
    *reduction_op = ncclProd;
    return Status::OK();
  } else if (merge_op == "Maximum") {
    *reduction_op = ncclMax;
    return Status::OK();
  } else if (merge_op == "Minimum") {
    *reduction_op = ncclMin;
    return Status::OK();
  } else {
    return errors::Internal(
        "Expected merge_op to be in [Add, Mul, Maximum, Minimum], found ",
        merge_op);
  }
}

string NcclCollectiveKey(const string& exec_key, int step_id) {
  return strings::StrCat(exec_key, ":", step_id);
}
}  // namespace

std::unique_ptr<NcclCommunicatorInterface> MaybeCreateNcclCommunicator(
    const ConfigProto& config) {
  // Skip creating a NcclCommunicator if there are 0 GPUs configured.
  const auto& device_count = config.device_count();
  auto item = device_count.find("GPU");
  if (item != device_count.end() && item->second == 0) {
    return nullptr;
  }
  return absl::make_unique<NcclCommunicator>();
}

void NcclCommunicator::Enqueue(std::shared_ptr<CollectiveContext> col_ctx,
                               StatusCallback done) {
  const CollectiveParams* col_params = col_ctx->col_params.get();
  const int num_global_devices = col_params->group.group_size;
  const int num_local_devices = col_params->group.num_devices_per_task.at(
      col_params->group.members[col_params->default_rank].task);
  const string nccl_collective_key =
      NcclCollectiveKey(col_ctx->exec_key, col_ctx->step_id);
  auto* compute_stream = col_ctx->op_ctx->op_device_context()->stream();
  auto* gpu_info = col_ctx->op_ctx->device()->tensorflow_gpu_device_info();
  auto participant = absl::make_unique<NcclManager::Participant>(
      compute_stream->parent(), compute_stream, gpu_info, col_ctx->input,
      col_ctx->output, col_ctx->col_params->default_rank,
      /*done_callback=*/nullptr);
  CancellationManager* cancel_mgr = col_ctx->op_ctx->cancellation_manager();
  if (cancel_mgr == nullptr) {
    participant->done_callback = std::move(done);
  } else {
    CancellationToken cancel_token = cancel_mgr->get_cancellation_token();
    bool already_cancelled =
        !cancel_mgr->RegisterCallback(cancel_token, [this]() {
          nccl_manager_.StartAbort(errors::Cancelled("op cancelled"));
          nccl_manager_.Reset();
        });
    if (already_cancelled) {
      done(errors::Cancelled("op cancelled"));
      return;
    }
    participant->done_callback = [cancel_mgr, cancel_token,
                                  done = std::move(done)](const Status& s) {
      // Do not block on deregistration since this can be invoked by
      // NcclManager::StartAbort() in the cancellation callback.
      cancel_mgr->TryDeregisterCallback(cancel_token);
      done(s);
    };
  }
  NcclManager::Context context(
      nccl_collective_key, num_local_devices, num_global_devices,
      col_params->group.runtime_details.communicator_key,
      col_params->source_rank);
  VLOG(1) << "NcclCommunicator::Enqueue type " << col_params->instance.type
          << " num_tasks " << col_params->group.num_tasks << " current task "
          << col_params->group.members[col_params->default_rank].task
          << " num local devices " << num_local_devices
          << " num global devices " << num_global_devices << " device "
          << col_ctx->device_name << " instance "
          << col_params->instance.instance_key;
  // `AddTo*` performs consistency checks for the NCCL call and enqueues the
  // `Participant` struct locally.  When all local participants with this
  // `nccl_collective_key` have called `AddToAllReduce` and
  // `SignalMultiNodeReady`, all devices at this worker are ready to process
  // this NCCL op.
  //
  // The `NcclManager` uses a dedicated CUDA stream for NCCL kernels.  At this
  // point, it synchronizes the NCCL stream with the compute stream, and then
  // enqueues the NCCL kernel on the NCCL stream.
  switch (col_params->instance.type) {
    case REDUCTION_COLLECTIVE: {
      ncclRedOp_t reduction_op;
      Status s =
          ReductionOp(col_params->merge_op->type_string(), &reduction_op);
      if (!s.ok()) {
        participant->done_callback(s);
        return;
      }
      nccl_manager_.AddToAllReduce(std::move(participant), context,
                                   reduction_op);
      break;
    }
    case GATHER_COLLECTIVE: {
      nccl_manager_.AddToAllGather(std::move(participant), context);
      break;
    }
    case BROADCAST_COLLECTIVE: {
      if (col_params->is_source) {
        nccl_manager_.AddBroadcastSend(std::move(participant), context);
      } else {
        nccl_manager_.AddBroadcastRecv(std::move(participant), context);
      }
      break;
    }
    default: {
      participant->done_callback(errors::Internal("Unexpected CollectiveType ",
                                                  col_params->instance.type));
      return;
    }
  }
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
  {
    // `WaitForDependencies` may block if the collective instances on which this
    // op depends have not yet launched.  When this function returns, this op is
    // ready to go.
    profiler::TraceMe activity("WaitForDependencies",
                               profiler::TraceMeLevel::kInfo);
    col_ctx->col_exec->WaitForDependencies(*col_params);
    nccl_manager_.SignalMultiNodeReady(nccl_collective_key);
  }
  {
    // When all devices at this worker have called `SignalMultiNodeReady`, the
    // `NcclManager` will enqueue the NCCL kernel on the NCCL stream.  Thus the
    // implementation of `UnblockDependencies` keeps track of the number of
    // devices that have launched.
    profiler::TraceMe activity("Schedule", profiler::TraceMeLevel::kInfo);
    col_ctx->col_exec->UnblockDependencies(*col_params);
  }
}

void NcclCommunicator::StartAbort(const Status& s) {
  nccl_manager_.StartAbort(s);
}

}  // namespace tensorflow

#else
namespace tensorflow {
std::unique_ptr<NcclCommunicatorInterface> MaybeCreateNcclCommunicator(
    const ConfigProto& config) {
  return nullptr;
}
}  // namespace tensorflow
#endif  // TENSORFLOW_USE_NCCL && (GOOGLE_CUDA || TENSORFLOW_USE_ROCM)
