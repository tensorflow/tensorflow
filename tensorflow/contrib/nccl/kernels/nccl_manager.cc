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
#include "tensorflow/contrib/nccl/kernels/nccl_manager.h"

#include <utility>

#ifdef GOOGLE_CUDA

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

using ::perftools::gputools::cuda::ScopedActivateExecutorContext;

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

  perftools::gputools::StreamExecutor* executor = nullptr;

  // The stream on which to run the nccl collective.
  // This is a different stream than the tensorflow compute stream.
  std::unique_ptr<perftools::gputools::Stream> stream;

  // See NcclManager::LoopKernelLaunches for information on these.
  std::unique_ptr<Thread> thread;
  mutex mu;
  condition_variable cv;
  // Has collective,rank pairs.
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
  explicit Communicator(std::vector<CommunicatorMember> members)
      : num_devices(members.size()), members(std::move(members)) {}

  const int num_devices;
  const std::vector<CommunicatorMember> members;  // indexed by rank.
};

namespace {
ncclDataType_t ToNcclType(DataType t) {
  switch (t) {
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
}  // namespace

// A participant in a Collective.  See <Collective> below.
struct NcclManager::Participant {
  Participant(const Tensor* in_t, Tensor* out_t, EventMgr* event_mgr,
              perftools::gputools::Stream* tensor_stream,
              perftools::gputools::StreamExecutor* executor, int gpu_device_id,
              NcclManager::DoneCallback done_callback)
      : in_t(in_t),
        out_t(out_t),
        event_mgr(event_mgr),
        tensor_stream(tensor_stream),
        executor(executor),
        gpu_device_id(gpu_device_id),
        done_callback(std::move(done_callback)) {
    DCHECK(executor != nullptr);
    DCHECK(event_mgr != nullptr);
    DCHECK(tensor_stream != nullptr);
  }
  // Owned by the caller, who must keep it live until <done_callback> is called.
  // Is NULL for participants that only receive data.
  const Tensor* in_t;

  // Owned by the caller, who must keep it live until <done_callback> is called.
  // Is NULL for participants that only send data.
  Tensor* out_t;

  // Owned by the caller, who must keep it live until <done_callback> is called.
  EventMgr* const event_mgr;

  // Owned by the caller, who must keep it live until <done_callback> is called.
  perftools::gputools::Stream* const tensor_stream;

  // Matches the executor in CommunicatorMember::stream. Expected to be live for
  // process lifetime.
  perftools::gputools::StreamExecutor* const executor = nullptr;

  const int gpu_device_id;

  NcclManager::DoneCallback done_callback;

  bool root = false;
};

// A Collective tracks a single communicator operation (e.g., a single
// AllReduce call).
struct NcclManager::Collective {
  Collective(DataType data_type_in, CollectiveType type_in,
             ncclRedOp_t reduction_op_in, int num_devices)
      : data_type(data_type_in),
        type(type_in),
        reduction_op(reduction_op_in),
        remaining_participants(num_devices) {
    participants.reserve(num_devices);
  }

  const DataType data_type;
  const CollectiveType type;
  const ncclRedOp_t reduction_op;  // applies when <type> is a reduction.

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
  // eligible for running with <available_participants> == participants.size().
  //
  // Guarded by the mutex of the containing Communicator.
  int available_participants = 0;

  mutable std::atomic_int_fast32_t remaining_participants;
};

NcclManager::NcclManager() {}
NcclManager::~NcclManager() {}
NcclManager* NcclManager::instance() {
  static NcclManager* instance = new NcclManager();
  return instance;
}

NcclManager::Communicator* NcclManager::GetCommunicator(
    NcclManager::Collective* collective) {
  // Sort by executor to make ordering of executors deterministic.
  std::sort(collective->participants.begin(), collective->participants.end(),
            [](const std::unique_ptr<Participant>& a,
               const std::unique_ptr<Participant>& b) {
              return a->executor < b->executor;
            });
  const int num_devices = collective->participants.size();

  mutex_lock l(mu_);

  // Scan to find an existing communicator that provides nccl communication
  // between the executors used by the participants in the collective. For
  // example, if a collective is for GPUs 0, 1, and 2 then this will scan
  // to find the communicator for GPUs 0, 1, and 2.
  //
  // Note that each executor identifies a context on one device, so this is the
  // same as getting the communicator connecting the devices in the collective.
  // A device can be in different communicators as well - for example, a
  // communicator for GPUs 0 and 1 is separate from one for GPUs 0, 1, and 2.
  //
  // Since it's expected that a small number of distinct communicators will
  // be needed, communicators_ is not garbage collected currently.
  //
  // Launching of kernels must be serialized so that, given collectives A and B,
  // and an order of them (e.g., A before B), then for each comm_stream
  // involved, the kernel for A is launched before the kernel for B. This is
  // guaranteed currently be a global mutex controlling additions of the kernels
  // to per-stream launch queues.  The launch queues are processed by
  // LoopKernelLaunches.
  for (auto& comm : communicators_) {
    if (comm->num_devices == num_devices) {
      int i;
      for (i = 0; i < num_devices; ++i) {
        if (comm->members[i].nccl_stream->executor !=
            collective->participants[i]->executor) {
          break;
        }
      }
      if (i == num_devices) return comm.get();
    }
  }

  auto* env = Env::Default();
  std::set<NcclStream*> used_streams;

  // Create and initialize a new communicator.
  // Note that this is done under the lock; performance is not expected to
  // matter as this happens a very small number of times.
  std::vector<CommunicatorMember> members(num_devices);
  std::vector<int> devices(num_devices);
  for (int i = 0; i < num_devices; ++i) {
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
      nccl_stream->stream.reset(new perftools::gputools::Stream(executor));
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

  std::vector<ncclComm_t> nccl_comms(num_devices);
  auto result = ncclCommInitAll(nccl_comms.data(), num_devices, devices.data());
  CHECK_EQ(result, ncclSuccess) << ncclGetErrorString(result);
  for (int rank = 0; rank < num_devices; ++rank) {
    members[rank].nccl_comm = nccl_comms[rank];
  }
  communicators_.emplace_back(new Communicator(std::move(members)));
  return communicators_.back().get();
}

void NcclManager::AddToAllReduce(int num_devices, const string& key,
                                 ncclRedOp_t reduction_op,
                                 perftools::gputools::StreamExecutor* executor,
                                 int gpu_device_id, EventMgr* event_mgr,
                                 perftools::gputools::Stream* tensor_stream,
                                 const Tensor* in_t, Tensor* out_t,
                                 const DoneCallback& done_callback) {
  std::unique_ptr<Participant> participant(
      new Participant(in_t, out_t, event_mgr, tensor_stream, executor,
                      gpu_device_id, done_callback));
  AddParticipant(num_devices, key, std::move(participant), in_t->dtype(),
                 kAllReduce, reduction_op);
}

void NcclManager::AddBroadcastSend(
    int num_devices, const string& key,
    perftools::gputools::StreamExecutor* executor, int gpu_device_id,
    EventMgr* event_mgr, perftools::gputools::Stream* tensor_stream,
    const Tensor* in_t, DoneCallback done_callback) {
  std::unique_ptr<Participant> participant(
      new Participant(in_t, nullptr /* out_t */, event_mgr, tensor_stream,
                      executor, gpu_device_id, std::move(done_callback)));
  participant->root = true;
  AddParticipant(num_devices, key, std::move(participant), in_t->dtype(),
                 kBroadcast, ncclSum /* unused */);
}

void NcclManager::AddBroadcastRecv(
    int num_devices, const string& key,
    perftools::gputools::StreamExecutor* executor, int gpu_device_id,
    EventMgr* event_mgr, perftools::gputools::Stream* tensor_stream,
    Tensor* out_t, DoneCallback done_callback) {
  std::unique_ptr<Participant> participant(
      new Participant(nullptr /* in_t */, out_t, event_mgr, tensor_stream,
                      executor, gpu_device_id, std::move(done_callback)));
  AddParticipant(num_devices, key, std::move(participant), out_t->dtype(),
                 kBroadcast, ncclSum /* unused */);
}

void NcclManager::AddReduceSend(int num_devices, const string& key,
                                ncclRedOp_t reduction_op,
                                perftools::gputools::StreamExecutor* executor,
                                int gpu_device_id, EventMgr* event_mgr,
                                perftools::gputools::Stream* tensor_stream,
                                const Tensor* in_t, Tensor* temp_t,
                                DoneCallback done_callback) {
  std::unique_ptr<Participant> participant(
      new Participant(in_t, temp_t, event_mgr, tensor_stream, executor,
                      gpu_device_id, std::move(done_callback)));
  AddParticipant(num_devices, key, std::move(participant), in_t->dtype(),
                 kReduce, reduction_op);
}

void NcclManager::AddReduceRecv(int num_devices, const string& key,
                                ncclRedOp_t reduction_op,
                                perftools::gputools::StreamExecutor* executor,
                                int gpu_device_id, EventMgr* event_mgr,
                                perftools::gputools::Stream* tensor_stream,
                                const Tensor* in_t, Tensor* out_t,
                                DoneCallback done_callback) {
  std::unique_ptr<Participant> participant(
      new Participant(in_t, out_t, event_mgr, tensor_stream, executor,
                      gpu_device_id, std::move(done_callback)));
  participant->root = true;
  AddParticipant(num_devices, key, std::move(participant), in_t->dtype(),
                 kReduce, reduction_op);
}

void NcclManager::AddParticipant(int num_devices, const string& key,
                                 std::unique_ptr<Participant> participant,
                                 DataType data_type,
                                 CollectiveType collective_type,
                                 ncclRedOp_t reduction_op) {
  Collective* to_run = nullptr;
  {
    mutex_lock l(mu_);
    auto& collective_ptr = collectives_[key];
    if (collective_ptr == nullptr) {
      collective_ptr.reset(new Collective(data_type, collective_type,
                                          reduction_op, num_devices));
    }
    Collective* collective = collective_ptr.get();
    DCHECK_EQ(collective->type, collective_type);
    DCHECK_LT(collective->participants.size(), num_devices);
    collective->participants.emplace_back(std::move(participant));
    ++collective->available_participants;

    if (collective->available_participants == num_devices) {
      to_run = collective;

      // Ownership is going to be transferred to RunCollective.
      collective_ptr.release();
      collectives_.erase(key);
    }
  }

  if (to_run != nullptr) {
    RunCollective(key, to_run);
  }
}

void NcclManager::RunCollective(const string& key, Collective* collective) {
  static mutex collective_mu;

  auto* communicator = GetCommunicator(collective);
  collective->communicator = communicator;
  const int size = communicator->num_devices;

  for (int rank = 0; rank < size; ++rank) {
    Participant* p = collective->participants[rank].get();
    NcclStream* nccl_stream = communicator->members[rank].nccl_stream;
    CHECK(nccl_stream != nullptr);

    if (p->in_t != nullptr) {
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
    for (int rank = 0; rank < size; ++rank) {
      NcclStream* nccl_stream = communicator->members[rank].nccl_stream;
      mutex_lock l(nccl_stream->mu);
      nccl_stream->pending_launches_.push_front(
          std::make_pair(collective, rank));
      nccl_stream->cv.notify_all();
    }
  }
}

void NcclManager::LoopKernelLaunches(NcclStream* nccl_stream) {
  perftools::gputools::Stream* comm_stream = nccl_stream->stream.get();
  ScopedActivateExecutorContext scoped_context(nccl_stream->executor);
  const cudaStream_t* cu_stream = reinterpret_cast<const cudaStream_t*>(
      comm_stream->implementation()->CudaStreamMemberHack());

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
    Collective* collective = next_launch.first;
    int rank = next_launch.second;

    // Launch the nccl kernel.
    ncclDataType_t data_type = ToNcclType(collective->data_type);
    Participant* p = collective->participants[rank].get();

    auto nccl_comm = collective->communicator->members[rank].nccl_comm;
    ncclResult_t nccl_result = ncclSuccess;
    switch (collective->type) {
      case kAllReduce: {
        const void* sendbuff = p->in_t->tensor_data().data();
        void* recvbuff = const_cast<char*>(p->out_t->tensor_data().data());

        nccl_result =
            ncclAllReduce(sendbuff, recvbuff, p->in_t->NumElements(), data_type,
                          collective->reduction_op, nccl_comm, *cu_stream);
        break;
      }
      case kBroadcast: {
        const Tensor* buf_t = p->in_t ? p->in_t : p->out_t;
        void* buf = const_cast<char*>(buf_t->tensor_data().data());
        nccl_result = ncclBcast(buf, buf_t->NumElements(), data_type,
                                collective->root_rank, nccl_comm, *cu_stream);
        break;
      }
      case kReduce: {
        const void* sendbuff = p->in_t->tensor_data().data();
        void* recvbuff = const_cast<char*>(p->out_t->tensor_data().data());
        nccl_result = ncclReduce(sendbuff, recvbuff, p->in_t->NumElements(),
                                 data_type, collective->reduction_op,
                                 collective->root_rank, nccl_comm, *cu_stream);
        break;
      }
    }

    // Run the done_callback when the nccl kernel finishes running.
    auto done_callback = [collective, rank, nccl_result]() {
      if (nccl_result == ncclSuccess) {
        collective->participants[rank]->done_callback(Status::OK());
      } else {
        // Propagate the error, but note that if other members of the collective
        // did launch their kernels, then they are hanging.
        collective->participants[rank]->done_callback(errors::Unknown(
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
