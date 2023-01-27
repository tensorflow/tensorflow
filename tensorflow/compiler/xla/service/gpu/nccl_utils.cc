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

#include "tensorflow/compiler/xla/service/gpu/nccl_utils.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/rendezvous.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/tsl/platform/env.h"

namespace xla {
namespace gpu {

bool IsGlobalNcclConfig() {
  static const bool global_nccl_config = std::getenv("NCCL_COMM_ID") != nullptr;
  return global_nccl_config;
}

bool IsNcclLaunchModeParallel() {
  static const bool is_launch_mode_parallel = []() {
    const char* launch_mode = std::getenv("NCCL_LAUNCH_MODE");
    return launch_mode && std::string_view(launch_mode) == "PARALLEL";
  }();
  return is_launch_mode_parallel;
}

Status ToStatus(ncclResult_t s, const char* file, int64_t line,
                const char* expr) {
  if (s == ncclSuccess) {
    return OkStatus();
  }
  return tsl::errors::Internal(
      absl::StrFormat("%s:%d: NCCL operation %s failed: %s", file, line, expr,
                      ncclGetErrorString(s)));
}

ncclRedOp_t ToNcclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ncclSum;
    case ReductionKind::PRODUCT:
      return ncclProd;
    case ReductionKind::MIN:
      return ncclMin;
    case ReductionKind::MAX:
      return ncclMax;
  }
}

namespace {

StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType element_type,
                                        Thunk::Kind reduction_op) {
  switch (element_type) {
    case S8:
      return ncclInt8;
    case PRED:
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
    case C64:
      return ncclFloat32;
    case F64:
    case C128:
      return ncclFloat64;
    case S16:
    case U16:
      // For all-reduce and reduce-scatter, we expect 16 bit integer types to be
      // promoted to 32-bit.
      if (reduction_op == Thunk::kNcclAllReduce ||
          reduction_op == Thunk::kNcclAllReduceStart ||
          reduction_op == Thunk::kNcclReduceScatter) {
        return tsl::errors::InvalidArgument(absl::StrFormat(
            "Unsupported data type: %s", PrimitiveType_Name(element_type)));
      }
      // For collectives that just move data around, we can use ncclFloat16 for
      // 16-bit integer data types.
      return ncclFloat16;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case BF16:
      return ncclBfloat16;
#endif
    default:
      return tsl::errors::InvalidArgument(absl::StrFormat(
          "Unsupported data type: %s", PrimitiveType_Name(element_type)));
  }
}

StatusOr<ncclUniqueId> ToNcclUniqueId(const std::string& id_str) {
  static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES,
                "NCCL_UNIQUE_ID_BYTES");

  TF_RET_CHECK(id_str.size() == NCCL_UNIQUE_ID_BYTES);
  ncclUniqueId id;
  absl::c_copy(id_str, id.internal);
  return id;
}

StatusOr<std::string> LocalNcclUniqueIdCallback(const NcclCliqueKey&) {
  ncclUniqueId id;
  XLA_CUDA_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
}

struct NcclCliqueState {
  ncclUniqueId unique_id;
  int64_t run_id = -1;

  // `mu` guards `communicators` and `status` during initialization.
  // Once `ready` has been notified, the communicators may be accessed without
  // synchronization.
  absl::Mutex mu;
  absl::Notification ready;
  Status status;
  absl::flat_hash_map<int, std::unique_ptr<NcclComm>> communicators;
};

using NcclClique = Lockable<NcclCliqueState>;

std::shared_ptr<StatusOr<NcclClique::Lock>> AcquireNcclClique(
    RunId run_id, OpId op_id, NcclCliqueKey clique_key,
    const NcclUniqueIdCallback& unique_id_callback,
    size_t num_local_participants) {
  static auto& cliques = *new ThreadSafeMap<NcclCliqueKey, NcclClique>;

  auto rendezvous_key = std::make_tuple(run_id, op_id, std::move(clique_key));

  int64_t terminate_timeout = xla::GetDebugOptionsFromFlags()
                                  .xla_gpu_nccl_termination_timeout_seconds();

  return RendezvousSingle<StatusOr<NcclClique::Lock>>(
      rendezvous_key, num_local_participants,
      [&]() -> StatusOr<NcclClique::Lock> {
        const NcclCliqueKey& clique_key = std::get<2>(rendezvous_key);
        NcclClique::Lock clique = cliques[clique_key].Acquire();
        if (clique->run_id < 0) {
          TF_ASSIGN_OR_RETURN(std::string id, unique_id_callback(clique_key));
          TF_ASSIGN_OR_RETURN(clique->unique_id, ToNcclUniqueId(id));
        }
        // If multiple executable are running simultaneously while using
        // multiple hosts, it is possible that different executables could
        // acquire the same clique on different hosts. We protect against this
        // by checking that the run ID increases monotonically.
        bool is_local = clique_key.devices().size() == num_local_participants;
        TF_RET_CHECK(is_local || (run_id.ToInt() >= clique->run_id));
        clique->run_id = run_id.ToInt();
        return clique;
      },
      /*warn_stuck_timeout=*/absl::Seconds(10),
      (terminate_timeout >= 0) ? absl::Seconds(terminate_timeout)
                               : absl::InfiniteDuration());
}

void CheckNcclAsyncError(NcclComm& lockable_comm) {
  ncclComm_t comm = *lockable_comm.Acquire();
  if (comm == nullptr) return;

  Status status = [comm] {
    ncclResult_t async_err;
    XLA_CUDA_RETURN_IF_ERROR(ncclCommGetAsyncError(comm, &async_err));
    if (async_err != ncclSuccess) {
      LOG(ERROR) << "Aborting communicator: " << comm
                 << " due to async NCCL error: "
                 << ncclGetErrorString(async_err);
      XLA_CUDA_RETURN_IF_ERROR(ncclCommAbort(comm));
    }
    return XLA_CUDA_STATUS(async_err);
  }();

  if (!status.ok()) LOG(ERROR) << status.ToString();
}

}  // namespace

StatusOr<std::pair<ncclDataType_t, int>> ToNcclDataTypeAndCountMultiplier(
    PrimitiveType element_type, Thunk::Kind reduction_op) {
  TF_ASSIGN_OR_RETURN(ncclDataType_t dtype,
                      ToNcclDataType(element_type, reduction_op));
  bool is_complex = primitive_util::IsComplexType(element_type);
  return std::make_pair(dtype, is_complex ? 2 : 1);
}

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices) {
  if (local_devices == nullptr) return participants.size();

  return absl::c_count_if(participants, [&](const GlobalDeviceId& device_id) {
    return absl::c_linear_search(*local_devices, device_id);
  });
}

StatusOr<const NcclUniqueIdCallback*> GetNcclUniqueIdCallback(
    const NcclUniqueIdCallback* unique_id_callback, bool is_local) {
  if (unique_id_callback != nullptr) return unique_id_callback;

  TF_RET_CHECK(is_local || IsGlobalNcclConfig())
      << "If non-local devices are taking part of a collective API on "
         "GPU, the nccl_unique_id_callback must be provided by the client.";

  static auto* local_callback =
      new NcclUniqueIdCallback(LocalNcclUniqueIdCallback);
  return local_callback;
}

StatusOr<NcclComm::Lock> AcquireNcclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    size_t num_local_participants,
    const NcclUniqueIdCallback& unique_id_callback, int rank) {
  // Ensure that this group of threads have exclusive access to the clique to
  // prevent threads from different groups locking communicators in the clique.
  NcclCliqueKey clique_key(std::move(participants));
  std::shared_ptr<StatusOr<NcclClique::Lock>> clique = AcquireNcclClique(
      run_id, op_id, clique_key, unique_id_callback, num_local_participants);

  if (!clique->ok()) return clique->status();

  struct AllCommunicators {
    absl::Mutex mu;
    std::vector<NcclComm*> communicators ABSL_GUARDED_BY(mu);
  };
  static auto& all_communicators = *new AllCommunicators;

  // Launch a thread that periodically checks all NCCL communicators for
  // asynchronous errors. If an asynchronous error is observed, the communicator
  // is aborted and an error message logged.
  static auto check_async_error_thread = tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "nccl_async_error_thread", [&] {
        while (true) {
          absl::SleepFor(absl::Seconds(30));
          absl::MutexLock lock(&all_communicators.mu);
          for (NcclComm* comm : all_communicators.communicators) {
            CheckNcclAsyncError(*comm);
          }
        }
      });
  (void)check_async_error_thread;  // Silence unused variable warning.

  NcclCliqueState& state = ***clique;
  if (!state.ready.HasBeenNotified()) {
    int nranks = clique_key.devices().size();
    const ncclUniqueId& id = state.unique_id;

    ncclComm_t comm = nullptr;
    Status status = XLA_CUDA_STATUS(ncclCommInitRank(&comm, nranks, id, rank));

    size_t num_initialized = [&] {
      absl::MutexLock lock(&state.mu);
      state.status.Update(status);
      state.communicators[rank] = std::make_unique<NcclComm>(comm);
      return state.communicators.size();
    }();

    // Wait for all communicators to initialize before allowing any progress.
    // Otherwise we may get deadlocks, because ncclCommInitRank may allocate,
    // which may block on the completion of device activity on a peer device,
    // which may depend on the completion of this collective if we do not have a
    // barrier to prevent it.
    if (num_initialized == num_local_participants) {
      state.ready.Notify();
    } else {
      TF_RETURN_IF_ERROR(status);
      state.ready.WaitForNotification();
    }

    absl::MutexLock lock(&all_communicators.mu);
    all_communicators.communicators.push_back(state.communicators[rank].get());
  }

  TF_RETURN_IF_ERROR(state.status);
  return state.communicators[rank]->Acquire();
}
}  // namespace gpu
}  // namespace xla
