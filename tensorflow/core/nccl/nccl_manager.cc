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

#ifdef GOOGLE_CUDA

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

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

using se::cuda::ScopedActivateExecutorContext;

// Contains data for a single stream used for nccl communication; this includes
// a background thread that calls NcclManager::LoopKernelLaunches.
struct NcclManager::NcclStream {
 public:
  NcclStream() {}
  ~NcclStream() {
    mutex_lock l(mu);
    shutdown_requested = true;
    cv.notify_all();
  }

  se::StreamExecutor* executor = nullptr;

  // The stream on which to run the nccl collective.
  // This is a different stream than the tensorflow compute stream.
  std::unique_ptr<se::Stream> stream;

  // See NcclManager::LoopKernelLaunches for information on these.
  std::unique_ptr<Thread> thread;
  mutex mu;
  condition_variable cv;
  // Has collective,participant_idx pairs.
  std::deque<std::pair<Collective*, int>> pending_launches_ GUARDED_BY(mu);
  bool shutdown_requested GUARDED_BY(mu) = false;
};

struct NcclManager::CommunicatorMember {
 public:
  CommunicatorMember() {}
  ~CommunicatorMember() {
    if (nccl_comm != nullptr) ncclCommDestroy(nccl_comm);
  }
  ncclComm_t nccl_comm;

  // Owned by NcclManager::device_to_comm_streams_.
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
struct NcclManager::Collective {
  Collective(DataType data_type_in, CollectiveType type_in,
             ncclRedOp_t reduction_op_in, int num_local_devices_in,
             int num_global_devices_in, const string& communicator_key_in)
      : data_type(data_type_in),
        type(type_in),
        reduction_op(reduction_op_in),
        num_local_devices(num_local_devices_in),
        num_global_devices(num_global_devices_in),
        single_node(num_local_devices_in == num_global_devices_in),
        communicator_key(communicator_key_in),
        remaining_participants(num_local_devices_in) {
    participants.reserve(num_local_devices_in);
  }

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

  mutable std::atomic_int_fast32_t remaining_participants;

  Status status;
};

NcclManager::NcclManager() {}
NcclManager::~NcclManager() {}
NcclManager* NcclManager::instance() {
  static NcclManager* instance = new NcclManager();
  return instance;
}

string NcclManager::GenerateCommunicatorKey() {
  ncclUniqueId nccl_id;
  ncclGetUniqueId(&nccl_id);
  return string(nccl_id.internal, NCCL_UNIQUE_ID_BYTES);
}

Status NcclManager::GetCommunicator(NcclManager::Collective* collective,
                                    NcclManager::Communicator** communicator) {
  // Sort by executor to make ordering of executors deterministic.
  std::sort(collective->participants.begin(), collective->participants.end(),
            [](const std::unique_ptr<Participant>& a,
               const std::unique_ptr<Participant>& b) {
              return a->executor < b->executor;
            });

  mutex_lock l(mu_);

  if (collective->single_node) {
    // For single-node collectives, we identify a communicator uniquely by the
    // set of devices participating in the collective.  For example, if a
    // collective is for GPUs 0, 1, and 2 then this will scan to find the
    // communicator for GPUs 0, 1, and 2.
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
      if (used_streams.insert(s.get()).second) {
        nccl_stream = s.get();
        break;
      }
    }
    if (nccl_stream == nullptr) {
      nccl_stream = new NcclStream();
      nccl_stream->executor = executor;
      nccl_stream->stream.reset(new se::Stream(executor));
      nccl_stream->stream->Init();

      streams.emplace_back(nccl_stream);
      used_streams.insert(nccl_stream);

      nccl_stream->thread.reset(env->StartThread(
          ThreadOptions(), "nccl_kernel_launch",
          [this, nccl_stream] { LoopKernelLaunches(nccl_stream); }));
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
      Collective* collective = collective_it->second.get();
      collective->multi_node_ready = true;
      to_run = CheckReady(collective_key, collective);
    }
  }

  if (to_run != nullptr) RunCollective(to_run);
}

void NcclManager::AddParticipant(std::unique_ptr<Participant> participant,
                                 const Context& context,
                                 CollectiveType collective_type,
                                 ncclRedOp_t reduction_op) {
  Collective* to_run = nullptr;
  const DataType data_type = participant->input->dtype();
  {
    mutex_lock l(mu_);
    auto collective_it = collectives_.find(context.collective_key);
    Collective* collective = nullptr;
    if (collective_it == collectives_.end()) {
      auto collective_unique_ptr = absl::make_unique<Collective>(
          data_type, collective_type, reduction_op, context.num_local_devices,
          context.num_global_devices, context.communicator_key);
      collective = collective_unique_ptr.get();
      collectives_.emplace(context.collective_key,
                           std::move(collective_unique_ptr));
    } else {
      collective = collective_it->second.get();
    }

    // Check `collective` is correct and consistent.
    if (collective->status.ok() && collective->single_node &&
        !collective->communicator_key.empty()) {
      collective->status =
          errors::Internal("Collective ", reduction_op,
                           " is single node but has communicator_key of size ",
                           collective->communicator_key.size());
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

    collective->participants.emplace_back(std::move(participant));
    ++collective->available_participants;

    to_run = CheckReady(context.collective_key, collective);
  }

  if (to_run != nullptr) RunCollective(to_run);
}

NcclManager::Collective* NcclManager::CheckReady(const string& collective_key,
                                                 Collective* collective) {
  Collective* to_run = nullptr;
  if (collective->available_participants == collective->num_local_devices) {
    if (collective->num_global_devices == collective->num_local_devices ||
        collective->multi_node_ready) {
      // Ownership transferred to callee.
      to_run = collective;
      auto collectives_it = collectives_.find(collective_key);
      collectives_it->second.release();
      collectives_.erase(collectives_it);
    }
  }
  return to_run;
}

void NcclManager::RunCollective(Collective* collective) {
  static mutex collective_mu(LINKER_INITIALIZED);

  Status s = collective->status;
  if (s.ok()) {
    s = GetCommunicator(collective, &collective->communicator);
  }
  if (!s.ok()) {
    for (int i = 0; i < collective->num_local_devices; ++i) {
      collective->participants[i]->done_callback(s);
    }
    delete collective;
    return;
  }

  for (int i = 0; i < collective->num_local_devices; ++i) {
    Participant* p = collective->participants[i].get();
    NcclStream* nccl_stream = collective->communicator->members[i].nccl_stream;
    CHECK(nccl_stream != nullptr);
    const int rank = p->global_rank >= 0 ? p->global_rank : i;

    if (p->input != nullptr) {
      // Wait to ensure that the kernel that produces the data in the input
      // tensor has finished running before the nccl kernel runs on the
      // communication stream.
      nccl_stream->stream->ThenWaitFor(p->tensor_stream);
    }
    if (p->root) {
      CHECK_EQ(collective->root_rank, -1);
      collective->root_rank = rank;
    }
  }

  if (collective->type == kBroadcast) {
    CHECK_NE(collective->root_rank, -1);
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
      nccl_stream->cv.notify_all();
    }
  }
}

void NcclManager::LoopKernelLaunches(NcclStream* nccl_stream) {
  se::Stream* comm_stream = nccl_stream->stream.get();
  ScopedActivateExecutorContext scoped_context(nccl_stream->executor);
  const cudaStream_t* cu_stream = reinterpret_cast<const cudaStream_t*>(
      comm_stream->implementation()->GpuStreamMemberHack());

  while (true) {
    // Find collective to run.
    std::pair<Collective*, int> next_launch;
    {
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

        VLOG(2) << "call NcclAllReduce participant " << p_idx << " sendbuff "
                << sendbuff << " recvbuff " << recvbuff << " nccl_comm "
                << nccl_comm << " comm_stream " << comm_stream
                << " cuda_stream " << cu_stream;
        nccl_result = ncclAllReduce(sendbuff, recvbuff, p->input->NumElements(),
                                    data_type, collective->reduction_op,
                                    nccl_comm, *cu_stream);
        break;
      }
      case kBroadcast: {
        const Tensor* buf_t = p->input ? p->input : p->output;
        void* buf = const_cast<char*>(buf_t->tensor_data().data());
        nccl_result = ncclBcast(buf, buf_t->NumElements(), data_type,
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

        VLOG(2) << "call NcclAllGather participant " << p_idx << " sendbuff "
                << sendbuff << " sendcount " << p->input->NumElements()
                << " recvbuff " << recvbuff << " recvcount "
                << p->output->NumElements() << " nccl_comm " << nccl_comm
                << " comm_stream " << comm_stream << " cuda_stream "
                << cu_stream;
        nccl_result = ncclAllGather(sendbuff, recvbuff, p->input->NumElements(),
                                    data_type, nccl_comm, *cu_stream);
        break;
      }
    }

    // Run the done_callback when the nccl kernel finishes running.
    auto done_callback = [collective, p_idx, nccl_result]() {
      if (nccl_result == ncclSuccess) {
        collective->participants[p_idx]->done_callback(Status::OK());
      } else {
        // Propagate the error, but note that if other members of the collective
        // did launch their kernels, then they are hanging.
        collective->participants[p_idx]->done_callback(errors::Unknown(
            "Error invoking NCCL: ", ncclGetErrorString(nccl_result)));
      }

      // TODO(cwhipkey): use RefCounted after figuring out how to use in a
      // custom op library.
      // See tensorflow/core/lib/core/refcount.h for details on this locking.
      if (collective->remaining_participants.load(std::memory_order_acquire) ==
              1 ||
          collective->remaining_participants.fetch_sub(1) == 1) {
        delete collective;
      }
    };
    p->event_mgr->ThenExecute(comm_stream, done_callback);
  }
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
