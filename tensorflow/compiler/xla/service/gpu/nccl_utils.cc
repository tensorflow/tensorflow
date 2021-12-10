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

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/service/global_device_id.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace gpu {

bool IsGlobalNcclConfig() {
  static const bool global_nccl_config = std::getenv("NCCL_COMM_ID") != nullptr;
  return global_nccl_config;
}

bool IsNcclLaunchModeParallel() {
  static const bool is_launch_mode_parallel =
      absl::string_view(std::getenv("NCCL_LAUNCH_MODE")) == "PARALLEL";
  return is_launch_mode_parallel;
}

Status ToStatus(ncclResult_t s, const char* file, int64_t line,
                const char* expr) {
  if (s == ncclSuccess) {
    return Status::OK();
  }
  return tensorflow::errors::Internal(
      absl::StrFormat("%s:%d: NCCL operation %s failed: %s", file, line, expr,
                      ncclGetErrorString(s)));
}

Status ToStatus(cudaError_t s, const char* file, int64_t line,
                const char* expr) {
  if (s == cudaSuccess) {
    return Status::OK();
  }
  return tensorflow::errors::Internal(
      absl::StrFormat("%s:%d: CUDA operation %s failed: %s", file, line, expr,
                      cudaGetErrorString(s)));
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

StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType element_type) {
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
#if defined(__CUDA_BF16_TYPES_EXIST__)
    case BF16:
      return ncclBfloat16;
#endif
    default:
      return tensorflow::errors::InvalidArgument(absl::StrFormat(
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

template <typename K, typename V>
class ThreadSafeMap {
 public:
  V& operator[](const K& key) {
    absl::MutexLock lock(&mutex_);
    std::unique_ptr<V>& value = map_[key];
    if (value == nullptr) value = std::make_unique<V>();
    return *value;
  }

  void ForEachValue(const std::function<void(V&)>& fn) {
    absl::MutexLock lock(&mutex_);
    for (const auto& it : map_) fn(*it.second);
  }

 private:
  absl::Mutex mutex_;
  absl::flat_hash_map<K, std::unique_ptr<V>> map_ ABSL_GUARDED_BY(mutex_);
};

StatusOr<std::string> LocalNcclUniqueIdCallback(const NcclCliqueKey&) {
  ncclUniqueId id;
  XLA_CUDA_RETURN_IF_ERROR(ncclGetUniqueId(&id));
  return std::string(id.internal, NCCL_UNIQUE_ID_BYTES);
}

void WaitAndLogIfStuck(absl::Mutex& mutex, const absl::Condition& condition) {
  constexpr absl::Duration kTimeout = absl::Seconds(10);
  if (mutex.AwaitWithTimeout(condition, kTimeout)) {
    return;
  }

  LOG(ERROR) << "This thread has been waiting for "
             << absl::ToInt64Seconds(kTimeout) << "s and may be stuck:";
  mutex.Await(condition);
  LOG(ERROR) << "Thread is unstuck! Warning above was a false-positive. "
                "Perhaps the timeout is too short.";
}

// A rendezvous for a group of threads.
//
// The group of threads identifies itself with a key that must be unique to the
// the group. When all threads have arrived at the rendezvous, one thread
// executes the given function and all threads received the result.
// TODO(cjfj): Replace XLA rendezvous code with this simpler implementation.
template <typename R, typename K>
std::shared_ptr<R> Rendezvous(const K& key, size_t num_threads,
                              const std::function<R()>& fn) {
  // Fast-path (DO NOT REMOVE: the logic below doesn't work for single thread).
  if (num_threads == 1) return std::make_shared<R>(fn());

  struct State {
    absl::Mutex mutex;
    size_t num_threads_arrived ABSL_GUARDED_BY(mutex) = 0;
    std::shared_ptr<R> result ABSL_GUARDED_BY(mutex);
  };

  static auto& states = *new ThreadSafeMap<K, State>;
  State& state = states[key];

  absl::MutexLock lock(&state.mutex);
  ++state.num_threads_arrived;

  std::shared_ptr<R> result;
  if (state.num_threads_arrived == num_threads) {
    // Last thread to arrive executes the function.
    CHECK(state.result == nullptr);
    result = std::make_shared<R>(fn());
    state.result = result;
    state.num_threads_arrived = 0;
  } else {
    absl::Condition result_ready(
        +[](std::shared_ptr<R>* ptr) { return ptr->get() != nullptr; },
        &state.result);
    WaitAndLogIfStuck(state.mutex, result_ready);

    // There is one use of the result in the shared state, plus one use for each
    // thread that has already retrieved the result.
    if (state.result.use_count() < num_threads) {
      result = state.result;
    } else {
      // Last thread to retrieve the result takes the result from the state,
      // allowing the other threads to exit the function.
      return std::move(state.result);
    }
  }

  // Wait for all threads to have retrieved the result. Without this, a thread
  // could duplicate or delete its copy of the result, invalidating the use
  // count logic above.
  absl::Condition result_taken(
      +[](std::shared_ptr<R>* ptr) { return ptr->get() == nullptr; },
      &state.result);
  WaitAndLogIfStuck(state.mutex, result_taken);
  return result;
}

struct NcclCliqueState {
  ncclUniqueId unique_id;
  int64_t run_id = -1;
};

using NcclClique = Lockable<NcclCliqueState>;

std::shared_ptr<StatusOr<NcclClique::Lock>> AcquireNcclClique(
    RunId run_id, OpId op_id, NcclCliqueKey clique_key,
    const NcclUniqueIdCallback& unique_id_callback,
    size_t num_local_participants) {
  static auto& cliques = *new ThreadSafeMap<NcclCliqueKey, NcclClique>;

  auto rendezvous_key = std::make_tuple(run_id, op_id, std::move(clique_key));

  return Rendezvous<StatusOr<NcclClique::Lock>>(
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
      });
}

void CheckNcclAsyncError(NcclComm& lockable_comm) {
  ncclComm_t comm = *lockable_comm.Acquire();
  if (comm == nullptr) return;

  Status status = [comm] {
    ncclResult_t async_err;
    XLA_CUDA_RETURN_IF_ERROR(ncclCommGetAsyncError(comm, &async_err));
    if (async_err != ncclSuccess) {
      LOG(ERROR) << "Async NCCL error. Aborting communicator: " << comm;
      XLA_CUDA_RETURN_IF_ERROR(ncclCommAbort(comm));
    }
    return XLA_CUDA_STATUS(async_err);
  }();

  if (!status.ok()) LOG(ERROR) << status.ToString();
}

}  // namespace

StatusOr<std::pair<ncclDataType_t, int>> ToNcclDataTypeAndCountMultiplier(
    PrimitiveType element_type) {
  TF_ASSIGN_OR_RETURN(ncclDataType_t dtype, ToNcclDataType(element_type));
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

  static NcclUniqueIdCallback local_callback(LocalNcclUniqueIdCallback);
  return &local_callback;
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

  auto comm_key = std::make_pair(std::move(clique_key), rank);
  static auto& comms = *new ThreadSafeMap<decltype(comm_key), NcclComm>;

  // Launch a thread that periodically checks all NCCL communicators for
  // asynchronous errors. If an asynchronous error is observed, the communicator
  // is aborted and an error message logged.
  static auto check_async_error_thread =
      tensorflow::Env::Default()->StartThread(
          tensorflow::ThreadOptions(), "nccl_async_error_thread", [&] {
            while (true) {
              absl::SleepFor(absl::Seconds(30));
              comms.ForEachValue(CheckNcclAsyncError);
            }
          });
  (void)check_async_error_thread;  // Silence unused variable warning.

  NcclComm::Lock comm = comms[comm_key].Acquire();
  if (*comm == nullptr) {
    int nranks = comm_key.first.devices().size();
    const ncclUniqueId& id = (**clique)->unique_id;
    XLA_CUDA_RETURN_IF_ERROR(ncclCommInitRank(comm.get(), nranks, id, rank));
  }
  return comm;
}

}  // namespace gpu
}  // namespace xla
