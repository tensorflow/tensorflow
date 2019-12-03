/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/nccl/nccl_manager.h"

#include <utility>

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
#endif

namespace tensorflow {

#if GOOGLE_CUDA
using se::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
using se::rocm::ScopedActivateExecutorContext;
// Local hipify of cuda symbols
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaGetErrorString hipGetErrorString
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaSuccess hipSuccess
int NcclManager::instance_count = 0;
#endif

#define NCCL_RETURN_IF_ERROR(...)                               \
  do {                                                          \
    ncclResult_t nccl_status = (__VA_ARGS__);                   \
    if (nccl_status != ncclSuccess) {                           \
      return errors::Internal(ncclGetErrorString(nccl_status)); \
    }                                                           \
  } while (0)

#define CUDA_RETURN_IF_ERROR(...)                               \
  do {                                                          \
    cudaError_t cuda_status = (__VA_ARGS__);                    \
    if (cuda_status != cudaSuccess) {                           \
      return errors::Internal(cudaGetErrorString(cuda_status)); \
    }                                                           \
  } while (0)

// Contains data for a single stream used for nccl communication; this includes
// a background thread that calls NcclManager::LoopKernelLaunches.
struct NcclManager::NcclStream : public core::RefCounted {
 public:
  NcclStream() = default;
  ~NcclStream() = default;

  se::StreamExecutor* executor = nullptr;

  // The stream on which to run the nccl collective.
  // This is a different stream than the tensorflow compute stream.
#if TENSORFLOW_USE_ROCM
  // On ROCm, we borrow the nccl stream from the device context.
  se::Stream* stream = nullptr;
#else
  std::unique_ptr<se::Stream> stream;
#endif

  // `mu` protects access to `pending_launches_`, which is the list of
  // collectives ready but whose kernels are yet to be launched.  When the
  // NcclManager object that owns this NcclStream object is destroyed, it
  // signals `cv` to unblock the thread waiting on more collectives.
  mutex mu;
  condition_variable cv;
  // Has (collective, participant_idx) pairs.
  std::deque<std::pair<Collective*, int>> pending_launches_ GUARDED_BY(mu);
  bool shutdown_requested GUARDED_BY(mu) = false;
};

struct NcclManager::CommunicatorMember {
 public:
  CommunicatorMember() {}
  ~CommunicatorMember() {
    if (nccl_comm != nullptr) ncclCommDestroy(nccl_comm);
  }

  ncclComm_t nccl_comm = nullptr;
  // Owned by NcclManager::device_to_comm_streams_ and LoopKernelLaunches.
  NcclStream* nccl_stream = nullptr;
};

struct NcclManager::Communicator {
 public:
  explicit Communicator(std::vector<CommunicatorMember> members,
                        const string& key)
      : num_devices(members.size()), members(std::move(members)), key(key) {}

  const int num_devices;
  const std::vector<CommunicatorMember> members;
  const string key;
};

namespace {

ncclDataType_t ToNcclType(DataType t) {
  switch (t) {
    case DT_HALF:
      return ncclHalf;
    case DT_FLOAT:
      return ncclFloat;
    case DT_DOUBLE:
      return ncclDouble;
    case DT_INT32:
      return ncclInt;
    case DT_INT64:
      return ncclInt64;
    default:
      return ncclFloat;
  }
}

void StringToNcclUniqueId(const string& str_id, ncclUniqueId* nccl_id) {
  if (str_id.size() == NCCL_UNIQUE_ID_BYTES) {
    memcpy(nccl_id->internal, str_id.data(), NCCL_UNIQUE_ID_BYTES);
  }
}

}  // namespace

// A `Collective` encapsulates state for a collective instance at one node.
// Typically, an instance in TensorFlow context would be defined by a collective
// group and the (step, frame iteration) for that execution.
//
// For each collective instance there will be one `Collective` object per node.
// For example,  a NCCL collective that runs on a single node with 4 GPUs would
// have a single `Collective` per step.  However, a collective that executes on
// 3 nodes with 4 GPUs each would have a `Collective` per node, each of which is
// tracking the 4 GPUs local to that node.
struct NcclManager::Collective : public core::RefCounted {
  Collective(const string& collective_key_in, DataType data_type_in,
             CollectiveType type_in, ncclRedOp_t reduction_op_in,
             int num_local_devices_in, int num_global_devices_in,
             const string& communicator_key_in)
      : collective_key(collective_key_in),
        data_type(data_type_in),
        type(type_in),
        reduction_op(reduction_op_in),
        num_local_devices(num_local_devices_in),
        num_global_devices(num_global_devices_in),
        single_node(num_local_devices_in == num_global_devices_in),
        communicator_key(communicator_key_in) {
    participants.reserve(num_local_devices_in);
#if TENSORFLOW_USE_ROCM
    // On ROCm platform, this allows caller to either use the singleton instance
    // or to manage one non-singleton NcclManager instance.
    // For example, the nccl_manager_test will use both paradigms in the same
    // executable, but not running concurrently (which would hang otherwise).
    if (NcclManager::instance_count > 1) {
      status = errors::Internal(
          "ROCm cannot use multi-node NCCL collectives on a single node");
    }
#endif
  }

  const string collective_key;  // A unique key for debugging.
  const DataType data_type;
  const CollectiveType type;
  const ncclRedOp_t reduction_op;  // applies when <type> is a reduction.
  const int num_local_devices;     // devices local to this node
  const int num_global_devices;    // devices across all nodes
  const bool single_node;          // true if all devices are at one node
  const string communicator_key;

  Communicator* communicator = nullptr;

  // All collective participants.
  //
  // Adding values in this vector is guarded by the mutex of the containing
  // NcclManager.
  std::vector<std::unique_ptr<Participant>> participants;

  // For collective types that have a root (e.g. the root of broadcast is the
  // sender), this is the rank of the root.
  int root_rank = -1;

  // How many participants have been registered so far. The Collective is
  // eligible for running with <available_participants> == num_local_devices.
  //
  // If this is a multi-node collective, we additionally have to synchronize
  // across nodes.  The caller would need to signal multi node readiness by
  // calling NcclManager::SignalMultiNodeReady, which sets `multi_node_ready` to
  // true.
  //
  // Guarded by the mutex of the containing Communicator.
  int available_participants = 0;
  bool multi_node_ready = false;

  Status status;
};

NcclManager::NcclManager() {
  VLOG(2) << "New NcclManager " << this;
#if TENSORFLOW_USE_ROCM
  ++instance_count;
#endif
}
NcclManager::~NcclManager() {
  VLOG(2) << "~NcclManager " << this;
#if TENSORFLOW_USE_ROCM
  --instance_count;
#endif
  for (auto& it : device_to_comm_streams_) {
    for (NcclStream* nccl_stream : it.second) {
      {
        mutex_lock l(nccl_stream->mu);
        nccl_stream->shutdown_requested = true;
        nccl_stream->cv.notify_all();
      }
      nccl_stream->Unref();
    }
  }
}
NcclManager* NcclManager::instance() {
  static NcclManager* instance = new NcclManager();
#if TENSORFLOW_USE_ROCM
  // singleton does not count against total instances
  // see comment above in Collective constructor concerning ROCm platform
  static std::once_flag once;
  std::call_once(once, [] { --NcclManager::instance_count; });
#endif
  return instance;
}

string NcclManager::GenerateCommunicatorKey() {
  ncclUniqueId nccl_id;
  ncclGetUniqueId(&nccl_id);
  return string(nccl_id.internal, NCCL_UNIQUE_ID_BYTES);
}

Status NcclManager::GetCommunicator(NcclManager::Collective* collective,
                                    NcclManager::Communicator** communicator) {
  // Sort by device ID, executor, and global rank to make ordering of
  // participants deterministic.
  std::sort(collective->participants.begin(), collective->participants.end(),
            [](const std::unique_ptr<Participant>& a,
               const std::unique_ptr<Participant>& b) {
              if (a->gpu_device_id != b->gpu_device_id) {
                return a->gpu_device_id < b->gpu_device_id;
              }
              if (a->executor != b->executor) {
                return a->executor < b->executor;
              }
              return a->global_rank < b->global_rank;
            });

  mutex_lock l(mu_);

  if (collective->communicator_key.empty()) {
    // For single-node collectives, when the caller does not specify a
    // `communicator_key`, we identify a communicator uniquely by the set of
    // devices participating in the collective.  For example, if a collective is
    // for GPUs 0, 1, and 2 then this will scan to find the communicator for
    // GPUs 0, 1, and 2.
    //
    // Note that each executor identifies a context on one device, so this is
    // the same as getting the communicator connecting the devices in the
    // collective. A device can be in different communicators as well - for
    // example, a communicator for GPUs 0 and 1 is separate from one for GPUs 0,
    // 1, and 2.
    //
    // Since it's expected that a small number of distinct communicators will
    // be needed, communicators_ is not garbage collected currently.
    //
    // Launching of kernels must be serialized so that, given collectives A and
    // B, and an order of them (e.g., A before B), then for each comm_stream
    // involved, the kernel for A is launched before the kernel for B. This is
    // guaranteed currently be a global mutex controlling additions of the
    // kernels to per-stream launch queues.  The launch queues are processed by
    // LoopKernelLaunches.
    for (auto& comm : communicators_) {
      if (comm->num_devices == collective->num_global_devices) {
        int i;
        for (i = 0; i < collective->num_local_devices; ++i) {
          if (comm->members[i].nccl_stream->executor !=
              collective->participants[i]->executor) {
            break;
          }
        }
        if (i == collective->num_local_devices) {
          *communicator = comm.get();
          return Status::OK();
        }
      }
    }
  } else {
#if NCCL_MAJOR < 2
    return errors::Internal(
        "Cannot use multi-node NCCL collectives with NCCL 1.x");
#endif
    if (collective->communicator_key.size() != NCCL_UNIQUE_ID_BYTES) {
      return errors::Internal("Expected communicator_key of size ",
                              NCCL_UNIQUE_ID_BYTES, " but found size ",
                              collective->communicator_key.size());
    }
    // This is an instance of multi-node collective.  We have previously
    // created a NCCL unique id and shared with all workers.  Now we find the
    // `Communicator` corresponding to this id.
    for (auto& comm : communicators_) {
      if (comm->key == collective->communicator_key) {
        *communicator = comm.get();
        return Status::OK();
      }
    }
  }

  auto* env = Env::Default();
  std::set<NcclStream*> used_streams;

  // Create and initialize a new communicator.
  // Note that this is done under the lock; performance is not expected to
  // matter as this happens a very small number of times.
  std::vector<CommunicatorMember> members(collective->num_local_devices);
  std::vector<int> devices(collective->num_local_devices);
  for (int i = 0; i < collective->num_local_devices; ++i) {
    auto* executor = collective->participants[i]->executor;

    // Find a communication stream to use for the device.
    auto& streams = device_to_comm_streams_[executor];
    NcclStream* nccl_stream = nullptr;
    for (const auto& s : streams) {
      if (used_streams.insert(s).second) {
        nccl_stream = s;
        break;
      }
    }
    if (nccl_stream == nullptr) {
      nccl_stream = new NcclStream();
      nccl_stream->executor = executor;
#if TENSORFLOW_USE_ROCM
      nccl_stream->stream = collective->participants[i]->context->nccl_stream();
#else
      nccl_stream->stream.reset(new se::Stream(executor));
      nccl_stream->stream->Init();
#endif

      streams.emplace_back(nccl_stream);
      used_streams.insert(nccl_stream);

      nccl_stream->Ref();
      env->SchedClosure([this, nccl_stream]() {
        LoopKernelLaunches(nccl_stream);
        nccl_stream->Unref();
      });
    }

    members[i].nccl_stream = nccl_stream;
    devices[i] = collective->participants[i]->gpu_device_id;
  }

  std::vector<ncclComm_t> nccl_comms(collective->num_local_devices);
#if NCCL_MAJOR >= 2
  // For NCCL 2, we always initialize using ncclCommInitRank guarded by NCCL
  // group primitives.
  ncclUniqueId nccl_id;
  if (collective->single_node) {
    NCCL_RETURN_IF_ERROR(ncclGetUniqueId(&nccl_id));
  } else {
    StringToNcclUniqueId(collective->communicator_key, &nccl_id);
  }
  int saved_device = 0;
  CUDA_RETURN_IF_ERROR(cudaGetDevice(&saved_device));
  NCCL_RETURN_IF_ERROR(ncclGroupStart());
  for (int i = 0; i < collective->num_local_devices; ++i) {
    // Set rank to `participant->global_rank` if provided, else `i`.
    const int rank = collective->participants[i]->global_rank >= 0
                         ? collective->participants[i]->global_rank
                         : i;
    CUDA_RETURN_IF_ERROR(cudaSetDevice(devices[i]));
    NCCL_RETURN_IF_ERROR(ncclCommInitRank(
        nccl_comms.data() + i, collective->num_global_devices, nccl_id, rank));
  }
  NCCL_RETURN_IF_ERROR(ncclGroupEnd());
  CUDA_RETURN_IF_ERROR(cudaSetDevice(saved_device));
#else
  // Since NCCL 1 is single node only, we use ncclCommInitAll.  We could have
  // used ncclCommInitRank with NCCL 1 as well, but then we would have to
  // issue each init call from a different thread
  // (https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/nccl1.html).
  NCCL_RETURN_IF_ERROR(ncclCommInitAll(
      nccl_comms.data(), collective->num_local_devices, devices.data()));
#endif

  for (int i = 0; i < collective->num_local_devices; ++i) {
    members[i].nccl_comm = nccl_comms[i];
  }
  communicators_.emplace_back(
      new Communicator(std::move(members), collective->communicator_key));
  *communicator = communicators_.back().get();
  return Status::OK();
}

void NcclManager::AddToAllReduce(std::unique_ptr<Participant> participant,
                                 const Context& context,
                                 ncclRedOp_t reduction_op) {
  AddParticipant(std::move(participant), context, kAllReduce, reduction_op);
}

void NcclManager::AddToAllGather(std::unique_ptr<Participant> participant,
                                 const Context& context) {
  AddParticipant(std::move(participant), context, kAllGather,
                 ncclSum /* unused */);
}

void NcclManager::AddBroadcastSend(std::unique_ptr<Participant> participant,
                                   const Context& context) {
  participant->root = true;
  AddParticipant(std::move(participant), context, kBroadcast,
                 ncclSum /* unused */);
}

void NcclManager::AddBroadcastRecv(std::unique_ptr<Participant> participant,
                                   const Context& context) {
  AddParticipant(std::move(participant), context, kBroadcast,
                 ncclSum /* unused */);
}

void NcclManager::AddReduceSend(std::unique_ptr<Participant> participant,
                                const Context& context,
                                ncclRedOp_t reduction_op) {
  AddParticipant(std::move(participant), context, kReduce, reduction_op);
}

void NcclManager::AddReduceRecv(std::unique_ptr<Participant> participant,
                                const Context& context,
                                ncclRedOp_t reduction_op) {
  AddParticipant(std::move(participant), context, kReduce, reduction_op);
}

void NcclManager::SignalMultiNodeReady(const string& collective_key) {
  Collective* to_run = nullptr;
  {
    mutex_lock l(mu_);
    auto collective_it = collectives_.find(collective_key);
    if (collective_it != collectives_.end()) {
      Collective* collective = collective_it->second;
      collective->multi_node_ready = true;
      if (CheckReady(collective_key, collective)) {
        to_run = collective;
      }
      VLOG(2) << "SignalMultiNodeReady collective " << collective_key
              << " to_run " << to_run;
    }
  }

  if (to_run != nullptr) RunCollective(to_run);
}

void NcclManager::AddParticipant(std::unique_ptr<Participant> participant,
                                 const Context& context,
                                 CollectiveType collective_type,
                                 ncclRedOp_t reduction_op) {
  Collective* to_run = nullptr;
  DataType data_type;
  if (participant->input != nullptr) {
    data_type = participant->input->dtype();
  } else {
    data_type = participant->output->dtype();
  }
  {
    mutex_lock l(mu_);
    auto collective_it = collectives_.find(context.collective_key);
    Collective* collective = nullptr;
    if (collective_it == collectives_.end()) {
      collective =
          new Collective(context.collective_key, data_type, collective_type,
                         reduction_op, context.num_local_devices,
                         context.num_global_devices, context.communicator_key);
      collectives_.emplace(context.collective_key, collective);
    } else {
      collective = collective_it->second;
    }

    // Check `collective` is correct and consistent.
    if (collective->status.ok() && !collective->single_node &&
        collective->communicator_key.empty()) {
      collective->status = errors::Internal(
          "Collective ", reduction_op, " is multi node with num_local_devices=",
          collective->num_local_devices,
          " and num_global_devices=", collective->num_global_devices,
          " but has an empty communicator_key");
    }
    if (collective->status.ok() && collective->communicator_key.size() !=
                                       context.communicator_key.size()) {
      collective->status =
          errors::Internal("Collective ", reduction_op,
                           " mismatch in member communicator_key with size ",
                           collective->communicator_key.size(),
                           " and arg communicator_key with size ",
                           context.communicator_key.size());
    }
    if (collective->status.ok() && collective->type != collective_type) {
      collective->status = errors::Internal(
          "Collective ", reduction_op, " previously initialized with type ",
          collective->type, " but now got type ", collective_type);
    }
    if (collective->status.ok() &&
        collective->num_global_devices != context.num_global_devices) {
      collective->status =
          errors::Internal("Collective ", reduction_op,
                           " previously initialized with num_global_devices ",
                           collective->num_global_devices, " but now got ",
                           context.num_global_devices);
    }
    if (collective->status.ok() &&
        collective->num_local_devices != context.num_local_devices) {
      collective->status =
          errors::Internal("Collective ", reduction_op,
                           "previously initialized with num_local_devices ",
                           collective->num_local_devices, " but now got ",
                           context.num_local_devices);
    }
    if (collective->status.ok() &&
        collective->participants.size() >= collective->num_local_devices) {
      collective->status = errors::Internal(
          "Collective ", reduction_op, " expected ",
          collective->num_local_devices, " participants but now has ",
          collective->participants.size(),
          " with one more participant being added");
    }
    if (collective->status.ok() && collective->root_rank >= 0 &&
        context.source_rank >= 0 &&
        collective->root_rank != context.source_rank) {
      collective->status = errors::Internal(
          "Collective ", collective->collective_key, " already has root_rank ",
          collective->root_rank, " but new participant has root_rank ",
          context.source_rank);
    }

    if (context.source_rank >= 0) {
      collective->root_rank = context.source_rank;
    }
    collective->participants.emplace_back(std::move(participant));
    ++collective->available_participants;

    if (CheckReady(context.collective_key, collective)) {
      to_run = collective;
    }
  }

  if (to_run != nullptr) RunCollective(to_run);
}

bool NcclManager::CheckReady(const string& collective_key,
                             Collective* collective) {
  if (collective->available_participants == collective->num_local_devices) {
    if (collective->num_global_devices == collective->num_local_devices ||
        collective->multi_node_ready) {
      // Ownership transferred to callee.
      collectives_.erase(collective_key);
      return true;
    }
  }
  return false;
}

void NcclManager::RunCollective(Collective* collective) {
  static mutex collective_mu(LINKER_INITIALIZED);

  Status status = collective->status;
  if (status.ok()) {
    status = GetCommunicator(collective, &collective->communicator);
  }

  for (int i = 0; status.ok() && i < collective->num_local_devices; ++i) {
    Participant* p = collective->participants[i].get();
    NcclStream* nccl_stream = collective->communicator->members[i].nccl_stream;
    CHECK(nccl_stream != nullptr);
    const int rank = p->global_rank >= 0 ? p->global_rank : i;

    if (p->input != nullptr) {
      // Wait to ensure that the kernel that produces the data in the input
      // tensor has finished running before the nccl kernel runs on the
      // communication stream.
      nccl_stream->stream->ThenWaitFor(p->input_event.get());
    }
    if (p->root) {
      if (collective->root_rank == -1) {
        collective->root_rank = rank;
      } else if (collective->root_rank != rank) {
        status = errors::Internal(
            "Inconsistent root rank ", collective->root_rank, " and GPU id ",
            p->gpu_device_id, " rank ", rank, " also marked as root.");
      }
    }
    VLOG(2) << "RunCollective rank " << rank << " global_rank "
            << p->global_rank << " root_rank " << collective->root_rank;
  }

  if (status.ok() && collective->type == kBroadcast &&
      collective->root_rank < 0) {
    status = errors::Internal("Root rank not indicated for collective ",
                              collective->collective_key);
  }

  if (!status.ok()) {
    for (int i = 0; i < collective->num_local_devices; ++i) {
      collective->participants[i]->done_callback(status);
    }
    collective->Unref();
    return;
  }

  {
    // Allow only one collective at a time to queue kernels for launching. This
    // is to prevent collectives from deadlocking each other.
    // Note that it would be possible to run multiple collectives at once, if
    // they have non-intersecting sets of devices.
    mutex_lock l(collective_mu);
    for (int i = 0; i < collective->num_local_devices; ++i) {
      NcclStream* nccl_stream =
          collective->communicator->members[i].nccl_stream;
      mutex_lock l(nccl_stream->mu);
      nccl_stream->pending_launches_.push_front(std::make_pair(collective, i));
      // Ownership is shared between LoopKernelLaunches for each stream in this
      // collective.
      collective->Ref();
      nccl_stream->cv.notify_all();
    }
  }
  collective->Unref();
}

void NcclManager::LoopKernelLaunches(NcclStream* nccl_stream) {
#if TENSORFLOW_USE_ROCM
  se::Stream* comm_stream = nccl_stream->stream;
#else
  se::Stream* comm_stream = nccl_stream->stream.get();
#endif
  ScopedActivateExecutorContext scoped_context(nccl_stream->executor);
  const cudaStream_t* cu_stream = reinterpret_cast<const cudaStream_t*>(
      comm_stream->implementation()->GpuStreamMemberHack());

  while (true) {
    // Find collective to run.
    std::pair<Collective*, int> next_launch;
    {
      VLOG(3) << "Locking mutex nccl_stream " << nccl_stream;
      mutex_lock l(nccl_stream->mu);
      while (nccl_stream->pending_launches_.empty()) {
        if (nccl_stream->shutdown_requested) {
          // No work and shutdown requested, exit.
          return;
        }
        nccl_stream->cv.wait(l);
      }
      next_launch = nccl_stream->pending_launches_.back();
      nccl_stream->pending_launches_.pop_back();
    }

    // Launch the nccl kernel.
    Collective* collective = next_launch.first;
    ncclDataType_t data_type = ToNcclType(collective->data_type);
    int p_idx = next_launch.second;
    Participant* p = collective->participants[p_idx].get();
    auto nccl_comm = collective->communicator->members[p_idx].nccl_comm;
    ncclResult_t nccl_result = ncclSuccess;
    switch (collective->type) {
      case kAllReduce: {
        const void* sendbuff = p->input->tensor_data().data();
        void* recvbuff = const_cast<char*>(p->output->tensor_data().data());

        VLOG(2) << "call NcclAllReduce collective_key "
                << collective->collective_key << " participant " << p_idx
                << " sendbuff " << sendbuff << " recvbuff " << recvbuff
                << " nccl_comm " << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        nccl_result = ncclAllReduce(sendbuff, recvbuff, p->input->NumElements(),
                                    data_type, collective->reduction_op,
                                    nccl_comm, *cu_stream);
        break;
      }
      case kBroadcast: {
        const void* sendbuff = nullptr;
        void* recvbuff = nullptr;
        int num_elements = -1;
        if (p->input) {
          sendbuff = p->input->tensor_data().data();
          num_elements = p->input->NumElements();
        }
        if (p->output) {
          recvbuff = const_cast<char*>(p->output->tensor_data().data());
          num_elements = p->output->NumElements();
        } else {
          // Operate in-place if no output (for the src node).
          recvbuff = const_cast<void*>(sendbuff);
        }
        if (num_elements < 0) {
          p->done_callback(errors::Internal(
              "Both input and output are null in ncclBroadcast"));
          collective->Unref();
          continue;
        }
        VLOG(2) << "call NcclBroadcast collective_key "
                << collective->collective_key << " participant " << p_idx
                << " sendbuff " << sendbuff << " recvbuff " << recvbuff
                << " nccl_comm " << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        nccl_result =
            ncclBroadcast(sendbuff, recvbuff, num_elements, data_type,
                          collective->root_rank, nccl_comm, *cu_stream);
        break;
      }
      case kReduce: {
        const void* sendbuff = p->input->tensor_data().data();
        void* recvbuff =
            p->output ? const_cast<char*>(p->output->tensor_data().data())
                      : nullptr;
        nccl_result = ncclReduce(sendbuff, recvbuff, p->input->NumElements(),
                                 data_type, collective->reduction_op,
                                 collective->root_rank, nccl_comm, *cu_stream);
        break;
      }
      case kAllGather: {
        const void* sendbuff = p->input->tensor_data().data();
        void* recvbuff = const_cast<char*>(p->output->tensor_data().data());

        VLOG(2) << "call NcclAllGather collective_key "
                << collective->collective_key << " participant " << p_idx
                << " sendbuff " << sendbuff << " sendcount "
                << p->input->NumElements() << " recvbuff " << recvbuff
                << " recvcount " << p->output->NumElements() << " nccl_comm "
                << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        nccl_result = ncclAllGather(sendbuff, recvbuff, p->input->NumElements(),
                                    data_type, nccl_comm, *cu_stream);
        break;
      }
    }

    // Run the done_callback when the nccl kernel finishes running.
    auto done_callback = [collective, p_idx, nccl_result]() {
      VLOG(2) << "done Nccl kernel collective_key "
              << collective->collective_key << " participant " << p_idx
              << " ncclResult " << nccl_result;
      if (nccl_result == ncclSuccess) {
        collective->participants[p_idx]->done_callback(Status::OK());
      } else {
        // Propagate the error, but note that if other members of the collective
        // did launch their kernels, then they are hanging.
        collective->participants[p_idx]->done_callback(errors::Unknown(
            "Error invoking NCCL: ", ncclGetErrorString(nccl_result)));
      }
      collective->Unref();
    };
    p->event_mgr->ThenExecute(comm_stream, done_callback);
  }
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
