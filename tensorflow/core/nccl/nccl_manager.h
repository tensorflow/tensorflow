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
#ifndef TENSORFLOW_CORE_NCCL_NCCL_MANAGER_H_
#define TENSORFLOW_CORE_NCCL_NCCL_MANAGER_H_

#ifdef GOOGLE_CUDA

#include <unordered_map>
#include <vector>

// TODO(rmlarsen): Get rid of this workaround. "gpu_assert" is defined when
// setting EIGEN_USE_THREADS. But when defining EIGEN_USE_THREADS here,
// incAtomic and other CUDA specific symbols are no longer recognized.
#ifndef gpu_assert
#define gpu_assert(x)
#endif

#include "third_party/nccl/nccl.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

// The communicator is used to make the asynchronous communicator calls and to
// manage the per-device streams used for communication.
//
// See nccl_ops.cc for example usage, including description of memory
// management and stream synchronization.
class NcclManager {
 public:
  typedef std::function<void(Status)> DoneCallback;
  NcclManager();
  ~NcclManager();

  static NcclManager* instance();

  // Add one participant to an all-reduce, sending in data from <in_t> and
  // receiving the result of the all-reduce in <out_t>.  The device for this
  // participant is managed by <executor>, and its events are polled by
  // <event_mgr>.
  //
  // This is an asynchronous call. When <done_callback> is called, <out_t> has
  // been set to the all-reduce result (note: the stream may not yet have been
  // synced).
  //
  // <tensor_stream> is the stream that should be waited on to ensure <in_t>'s
  // data is available on the GPU for the communication stream to access. It
  // is also the stream that will use the produced data; <done_callback> is
  // not called until the next kernel launched on <stream> would see the data.
  void AddToAllReduce(int num_devices, const string& key,
                      ncclRedOp_t reduction_op, se::StreamExecutor* executor,
                      int gpu_device_id, EventMgr* event_mgr,
                      se::Stream* tensor_stream, const Tensor* in_t,
                      Tensor* out_t, const DoneCallback& done_callback);

  // AddBroadcastSend and AddBroadcastRecv combine to sent data from one sender
  // to all receivers.
  void AddBroadcastSend(int num_devices, const string& key,
                        se::StreamExecutor* executor, int gpu_device_id,
                        EventMgr* event_mgr, se::Stream* tensor_stream,
                        const Tensor* in_t, DoneCallback done_callback);
  void AddBroadcastRecv(int num_devices, const string& key,
                        se::StreamExecutor* executor, int gpu_device_id,
                        EventMgr* event_mgr, se::Stream* tensor_stream,
                        Tensor* out_t, DoneCallback done_callback);

  // AddReduceSend and AddReduceRecv combine to sent data from all senders
  // to one receiver.
  void AddReduceSend(int num_devices, const string& key,
                     ncclRedOp_t reduction_op, se::StreamExecutor* executor,
                     int gpu_device_id, EventMgr* event_mgr,
                     se::Stream* tensor_stream, const Tensor* in_t,
                     DoneCallback done_callback);
  void AddReduceRecv(int num_devices, const string& key,
                     ncclRedOp_t reduction_op, se::StreamExecutor* executor,
                     int gpu_device_id, EventMgr* event_mgr,
                     se::Stream* tensor_stream, const Tensor* in_t,
                     Tensor* out_t, DoneCallback done_callback);

 private:
  enum CollectiveType {
    kAllReduce = 1,
    kBroadcast = 2,
    kReduce = 3,
  };
  struct Collective;
  struct Communicator;
  struct CommunicatorMember;
  struct NcclStream;
  struct Participant;

  Communicator* GetCommunicator(Collective* collective);

  void AddParticipant(int num_devices, const string& key,
                      std::unique_ptr<Participant> participant,
                      DataType data_type, CollectiveType collective_type,
                      ncclRedOp_t reduction_op);

  // Run <collective>.  This calls takes ownership of <collective>.
  void RunCollective(const string& key, Collective* collective);
  void LoopKernelLaunches(NcclStream* stream);

  mutex mu_;

  // Maps key to collectives currently being assembled or run.
  std::unordered_map<string, std::unique_ptr<Collective>> collectives_
      GUARDED_BY(mu_);

  // Maps a device to the communication streams that make up its collective.
  // This is used to share the stream across different communicators that
  // include the same device.
  std::map<se::StreamExecutor*, std::vector<std::unique_ptr<NcclStream>>>
      device_to_comm_streams_ GUARDED_BY(mu_);

  std::vector<std::unique_ptr<Communicator>> communicators_;

  TF_DISALLOW_COPY_AND_ASSIGN(NcclManager);
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_NCCL_NCCL_MANAGER_H_
