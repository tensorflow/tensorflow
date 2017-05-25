/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_MPI_MPI_RENDEZVOUS_MGR_H_
#define TENSORFLOW_CONTRIB_MPI_MPI_RENDEZVOUS_MGR_H_

#ifdef TENSORFLOW_USE_MPI

#include <queue>
#include <thread>
#include <list>
#include <string>
#include <memory>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include <iostream>

#include "tensorflow/contrib/mpi/mpi_utils.h"
#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/contrib/mpi/mpi_msg.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

#define TAG_REQTENSOR 1010
#define TAG_SENDTENSOR 2020
#define TAG_SENDTENSOR2 3030

namespace tensorflow {

class MPISendTensorCall {
 public:
  char* send_buffer_;
  char* send_buffer2_;

  MPI_Request msg1_;
  MPI_Request msg2_;
  int done1_;  // Int instead of bool for simpler IsFinished logic
  int done2_;
  MPIRecvTensorResponse mRes_;
  Notification n_;

  MPISendTensorCall()
      : send_buffer_(nullptr), send_buffer2_(nullptr), done1_(1), done2_(1) {}

  ~MPISendTensorCall() {
    MPI_CHECK(MPI_Wait(&msg1_, MPI_STATUS_IGNORE));
    n_.Notify();
    MPI_CHECK(MPI_Free_mem(send_buffer_));
    //    delete[] send_buffer_;
    delete[] send_buffer2_;
  }

  MPISendTensorCall(MPISendTensorCall&&) = delete;

  void Init(const Rendezvous::ParsedKey& parsed, const int64 step_id,
            const bool is_dead) {
    mRes_.set_key(parsed.FullKey().ToString());
    mRes_.set_step_id(step_id);
    mRes_.mutable_response()->set_is_dead(is_dead);
    mRes_.mutable_response()->set_send_start_micros(
        Env::Default()->NowMicros());
    mRes_.set_singlesend(true);
  }

  bool IsFinished() {
    MPI_Status status;
    if (!done1_) MPI_CHECK(MPI_Test(&msg1_, &done1_, &status));
    if (!done2_) MPI_CHECK(MPI_Test(&msg2_, &done2_, &status));
    return done1_ && done2_;
  }
};

class MPIRequestTensorCall {
 public:
  Rendezvous::DoneCallback done_;
  RecvTensorRequest req_;
  MPI_Request mpi_request_;
  char* request_buffer_;
  size_t request_buffer_size_;
  std::function<void(MPIRecvTensorResponse)> recv_call_;

  MPIRequestTensorCall() : request_buffer_(nullptr) {}
  ~MPIRequestTensorCall() {
    MPI_CHECK(MPI_Wait(&mpi_request_, MPI_STATUS_IGNORE));
    // delete[] request_buffer_;
    MPI_CHECK(MPI_Free_mem(request_buffer_));
  }

  void Init(const Rendezvous::ParsedKey& parsed, const int64 step_id) {
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(parsed.FullKey().data(), parsed.FullKey().size());
    request_buffer_size_ = req_.ByteSize();
    //   request_buffer_ = new char[request_buffer_size_];
    //  req_.SerializeToArray(request_buffer_, request_buffer_size_);
  }
};

class MPIRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  MPIRemoteRendezvous(const WorkerEnv* env, int64 step_id, const MPIUtils* util,
                      BaseRendezvousMgr* mgr_)
      : BaseRemoteRendezvous(env, step_id, false),
        mpiutils_(util),
        rendezvous_mgr_(mgr_) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~MPIRemoteRendezvous() override;

  const MPIUtils* mpiutils_;
  BaseRendezvousMgr* rendezvous_mgr_;

  TF_DISALLOW_COPY_AND_ASSIGN(MPIRemoteRendezvous);
};

class MPIRendezvousMgr : public BaseRendezvousMgr {
 public:
  explicit MPIRendezvousMgr(const WorkerEnv* env);
  ~MPIRendezvousMgr() {
    delete mpiutils_;
    fprintf(stderr, "Delete MPIRendezvousMgr \n");
    // TODO(jbedorf) stop background_thread_
    MPI_CHECK(MPI_Finalize());
  }

  void QueueRequest(std::string key, int64 step_id,
                    std::function<void()> request_call,
                    MPIRequestTensorCall* rCall) {
    mutex_lock l(mrq_);
    request_queue_.push(RequestQueueEntry(key, std::move(request_call)));
    recv_tensor_map_[step_id][key] =
        std::shared_ptr<MPIRequestTensorCall>(rCall);
  }

  void RemoveStepID(const int64 step_id) {
    mutex_lock l(mrq_);
    CHECK(recv_tensor_map_[step_id].size() == 0) << "Removing unfinished step";
    recv_tensor_map_.erase(step_id);
    // TODO(jbedorf) Should we verify that the step_id is clear before remove?
  }

 protected:
  BaseRemoteRendezvous* Create(int64 step_id,
                               const WorkerEnv* worker_env) override;

 private:
  typedef std::function<MPISendTensorCall*(
      const Status&, const Rendezvous::Args&, const Rendezvous::Args&,
      const Tensor&, const bool, MPISendTensorCall*)> MPIRecvTensorCallBack;

  typedef std::pair<std::string, std::function<void()>> RequestQueueEntry;
  typedef std::pair<std::string, std::function<MPISendTensorCall*()>>
      SendQueueEntry;

  const WorkerEnv* worker_env_2;
  std::thread background_thread_;
  MPIUtils* mpiutils_;
  bool use_optimal_transfer_;

  mutex msq_;
  mutex mrq_;

  std::queue<SendQueueEntry> send_queue_ GUARDED_BY(msq_);
  std::queue<RequestQueueEntry> request_queue_ GUARDED_BY(mrq_);
  std::map<int64, std::unordered_map<std::string,
                                     std::shared_ptr<MPIRequestTensorCall>>>
      recv_tensor_map_ GUARDED_BY(mrq_);

  void AddRequest(RecvTensorRequest, const int);
  void MPIBackgroundThread();

  void QueueSendRequest(SendQueueEntry req) {
    mutex_lock l(msq_);
    send_queue_.push(req);
  }

  void GetRecvCall(const int64 step_id, const std::string& key,
                   std::shared_ptr<MPIRequestTensorCall>* call) {
    mutex_lock l(mrq_);
    if (recv_tensor_map_.find(step_id) == recv_tensor_map_.end()) {
      LOG(FATAL) << "Step not found in recv_tensor_map_, step: " << step_id
                 << " key:  " << key << std::endl;
    }
    if (recv_tensor_map_[step_id].find(key) !=
        recv_tensor_map_[step_id].end()) {
      *call = recv_tensor_map_[step_id][key];
    } else {
      LOG(FATAL) << "Key not found in recv_tensor_map_, step: " << step_id
                 << " key:  " << key << std::endl;
    }
  }

  void RemoveRecvCall(const int64 step_id, const std::string& key) {
    mutex_lock l(mrq_);
    recv_tensor_map_[step_id].erase(key);
  }

  bool GetRequest(RequestQueueEntry* req) {
    mutex_lock l(mrq_);
    if (!request_queue_.empty()) {
      *req = request_queue_.front();
      request_queue_.pop();
      return true;
    }
    return false;
  }

  bool GetResponse(SendQueueEntry* send) {
    mutex_lock l(msq_);
    if (!send_queue_.empty()) {
      *send = send_queue_.front();
      send_queue_.pop();
      return true;
    }
    return false;
  }

  template <typename T>
  int ProbeForData(const int tag, MPI_Status* status, T* obj) {
    int flag = 0, msg_size = 0;
    MPI_Message msg;
    // Receive the message, probe as size is variable
    MPI_CHECK(
        MPI_Improbe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &msg, status));
    if (flag) {
      MPI_CHECK(MPI_Get_count(status, MPI_CHAR, &msg_size));
      MPI_Status stat2;
      std::vector<char> request_buffer_(msg_size);
      MPI_Mrecv(&request_buffer_[0], msg_size, MPI_CHAR, &msg, &stat2);
      bool res = obj->ParseFromArray(&request_buffer_[0], msg_size);
      CHECK(res) << "Failed to parse incomming message";
    }
    return flag;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(MPIRendezvousMgr);
};  // MPIRendezvousMgr
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_MPI
#endif  // TENSORFLOW_CONTRIB_MPI_MPI_RENDEZVOUS_MGR_H_
