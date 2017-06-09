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

#ifdef TENSORFLOW_USE_MPI

#include "tensorflow/contrib/mpi/mpi_rendezvous_mgr.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"

namespace tensorflow {

MPIRendezvousMgr::MPIRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env), worker_env_2(env), use_optimal_transfer_(false) {

  const char* mpienv = getenv("MPI_OPTIMAL_PATH");
  if (mpienv && mpienv[0] == '1') {
    LOG(INFO) << "MPI Optimal copy path enabled (Requires CUDA-Aware MPI when "
                 "using GPUs)\n";
    use_optimal_transfer_ = true;
  }

  // extract worker-name
  auto parsed = env->local_devices[0]->parsed_name();
  const std::string task_id = strings::StrCat(parsed.job, ":", parsed.replica);

  mpiutils_ = new MPIUtils(task_id);
  background_thread_ =
      std::thread(&MPIRendezvousMgr::MPIBackgroundThread, this);
}

BaseRemoteRendezvous* MPIRendezvousMgr::Create(int64 step_id,
                                               const WorkerEnv* worker_env) {
  return new MPIRemoteRendezvous(worker_env, step_id, mpiutils_, this);
}

void MPIRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {

  Status s = Status::OK();
  MPIRequestTensorCall* rendezvous_call = new MPIRequestTensorCall();

  VLOG(2) << "MPI User requested " << parsed.FullKey()
          << " @ step: " << step_id_;

  std::string src_task =
      strings::StrCat(parsed.src.job, ":", parsed.src.replica);
  const int dst = mpiutils_->GetSourceID(src_task);

  Device* dst_device;
  if (s.ok()) {
    s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_device);
    CHECK(s.ok()) << "Device lookup failed";
  } else {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }

  // Set properties of the request object and create the request function
  rendezvous_call->Init(parsed, step_id_);

  std::function<void()> request_call = [parsed, dst, rendezvous_call]() {
    // Use MPI_Alloc_mem here to force allocation inside MPI thread
    // this is not optimal, but prevents memory corruption and segmentation
    // faults during inter-server transfers...
    MPI_CHECK(MPI_Alloc_mem(rendezvous_call->request_buffer_size_,
                            MPI_INFO_NULL, &rendezvous_call->request_buffer_));
    rendezvous_call->req_.SerializeToArray(
        rendezvous_call->request_buffer_,
        rendezvous_call->request_buffer_size_);
    MPI_CHECK(MPI_Isend(rendezvous_call->request_buffer_,
                        rendezvous_call->request_buffer_size_, MPI_CHAR, dst,
                        TAG_REQTENSOR, MPI_COMM_WORLD,
                        &rendezvous_call->mpi_request_));
  };

  // Create the function which is called when the Tensor is send by remote
  const int64 temp1 = step_id_;
  rendezvous_call->recv_call_ =
      [this, parsed, recv_args, done, dst, temp1, rendezvous_call](
          MPIRecvTensorResponse mpi_response) {
    Status s;
    Device* dst_device;
    if (s.ok()) {
      s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_device);
      CHECK(s.ok()) << "Device lookup failed";
    }

    VLOG(3) << "MPI Received tensor " << parsed.FullKey()
            << " @ step: " << temp1
            << " single-send: " << mpi_response.singlesend();

    Tensor val;
    if (mpi_response.singlesend()) {
      dst_device->MakeTensorFromProto(mpi_response.response().tensor(),
                                      recv_args.alloc_attrs, &val);
    } else {
      TensorResponse tr;
      tr.InitAlloc(dst_device, recv_args.alloc_attrs);
      tr.InitPartial(mpi_response.response());
      const size_t nBytes = tr.tensor().TotalBytes();
      void* data = const_cast<void*>(DMAHelper::base(&tr.tensor()));
      MPI_Status status;
      MPI_CHECK(MPI_Recv(data, static_cast<int>(nBytes), MPI_BYTE, dst,
                         TAG_SENDTENSOR2, MPI_COMM_WORLD, &status));
      val = std::move(tr.tensor());
    }

    done(s, Args(), recv_args, val, mpi_response.response().is_dead());
  };

  MPIRendezvousMgr* mgr =
      reinterpret_cast<MPIRendezvousMgr*>(this->rendezvous_mgr_);
  mgr->QueueRequest(parsed.FullKey().ToString(), step_id_,
                    std::move(request_call), rendezvous_call);
}

MPIRemoteRendezvous::~MPIRemoteRendezvous() {
  MPIRendezvousMgr* mgr =
      reinterpret_cast<MPIRendezvousMgr*>(this->rendezvous_mgr_);
  mgr->RemoveStepID(step_id_);
}

/*
 * Add the request for one of our Tensors by a remote process
 * to the local send/table. The here created callback will
 * be called once the Tensor data has arrived and is
 * ready to be send to the remote requester.
 */
void MPIRendezvousMgr::AddRequest(RecvTensorRequest request,
                                  const int mpi_dst) {
  const int64 step_id = request.step_id();
  const std::string& key = request.rendezvous_key();
  Rendezvous::ParsedKey parsed;
  TF_CHECK_OK(Rendezvous::ParseKey(key, &parsed));

  MPIRecvTensorCallBack send_cb = [this, mpi_dst, parsed](
      const Status& status, const Rendezvous::Args& send_args,
      const Rendezvous::Args& recv_args, const Tensor& val, bool is_dead,
      MPISendTensorCall* mpi_send_call) {
    // TODO(jbedorf) this should be a loop over max size
    CHECK(mpi_send_call->mRes_.ByteSize() < INT_MAX)
        << "Buffer too large for single transfer";
    MPI_CHECK(MPI_Alloc_mem(mpi_send_call->mRes_.ByteSize(), MPI_INFO_NULL,
                            &mpi_send_call->send_buffer_));
    mpi_send_call->mRes_.SerializeToArray(mpi_send_call->send_buffer_,
                                          mpi_send_call->mRes_.ByteSize());

    MPI_CHECK(MPI_Isend(mpi_send_call->send_buffer_,
                        static_cast<int>(mpi_send_call->mRes_.ByteSize()),
                        MPI_CHAR, mpi_dst, TAG_SENDTENSOR, MPI_COMM_WORLD,
                        &(mpi_send_call->msg1_)));
    MPI_CHECK(MPI_Test(&mpi_send_call->msg1_, &mpi_send_call->done1_,
                       MPI_STATUS_IGNORE));

    if (!mpi_send_call->mRes_.singlesend()) {
      const int tensor_size = static_cast<int>(val.TotalBytes());
      void* temp = const_cast<void*>(DMAHelper::base(&val));

      // If the MPI library is not GPU aware there should be a data transfer
      // here to get the data on the host.
      // if(src_dev->tensorflow_gpu_device_info()) //memcpy to send_buffer2_

      // TODO(jbedorf)  this should be a loop over max size
      MPI_CHECK(MPI_Isend(temp, tensor_size, MPI_CHAR, mpi_dst, TAG_SENDTENSOR2,
                          MPI_COMM_WORLD, &mpi_send_call->msg2_));
      mpi_send_call->done2_ = 0;
    }
    return mpi_send_call;
  };

  // Wrapper around the read callback to place the callback on our queue
  Rendezvous::DoneCallback done_cb = [this, parsed, step_id, send_cb](
      const Status& status, const Rendezvous::Args& send_args,
      const Rendezvous::Args& recv_args, const Tensor& val, bool is_dead) {
    if (!status.ok()) {
      CHECK(status.ok()) << "RecvLocalAsync was not ok, key: "
                         << parsed.FullKey() << " step: " << step_id
                         << " error message: " << status.error_message();
      return;
    }

    VLOG(3) << "MPI Sending tensor " << parsed.FullKey()
            << " @ step: " << step_id << std::endl;

    auto mpi_send_call = new MPISendTensorCall();
    mpi_send_call->Init(parsed, step_id, is_dead);

    Device* src_dev = nullptr;
    Status s = this->worker_env_2->device_mgr->LookupDevice(parsed.src_device,
                                                            &src_dev);
    CHECK(s.ok()) << "src device not found";

    // Control if shape and data should be send together or if we can optimize
    // it in two different transfers, thereby reducing memory copies
    bool doOptimalTransfer = true;
    if (!DataTypeCanUseMemcpy(val.dtype())) doOptimalTransfer = false;
    if (val.TotalBytes() < 1024) doOptimalTransfer = false;

    doOptimalTransfer = doOptimalTransfer && use_optimal_transfer_;

    if (doOptimalTransfer) {
      // First send the Tensor description and in a follow up transfer the data
      mpi_send_call->mRes_.mutable_response()->mutable_tensor()->set_dtype(
          val.dtype());
      val.shape().AsProto(mpi_send_call->mRes_.mutable_response()
                              ->mutable_tensor()
                              ->mutable_tensor_shape());
      mpi_send_call->mRes_.set_singlesend(false);
    } else {
      // Send the Tensor description and data in a single transfer
      if (src_dev->tensorflow_gpu_device_info() &&
          (!send_args.alloc_attrs.on_host())) {
        Notification n;
        GPUUtil::SetProtoFromGPU(
            val, src_dev, send_args.device_context,
            mpi_send_call->mRes_.mutable_response()->mutable_tensor(), is_dead,
            [&n, &s](const Status& s_) {
              s = s_;
              n.Notify();
            });
        n.WaitForNotification();
      } else {
        val.AsProtoTensorContent(
            mpi_send_call->mRes_.mutable_response()->mutable_tensor());
      }
    }

    std::function<MPISendTensorCall*()> res = std::bind(
        send_cb, status, send_args, recv_args, val, is_dead, mpi_send_call);

    SendQueueEntry req(parsed.FullKey().ToString().c_str(), std::move(res));

    this->QueueSendRequest(req);

    // Wait for the notification that indicates the tensor has been
    // successfully transmitted to the remote process. Only needed if we
    // have not parsed the tensor to proto
    if (doOptimalTransfer) mpi_send_call->n_.WaitForNotification();
  };  // done_cb

  worker_env_2->compute_pool->Schedule([this, step_id, parsed, done_cb]() {
    this->RecvLocalAsync(step_id, parsed, done_cb);
  });
}

void MPIRendezvousMgr::MPIBackgroundThread() {
  std::list<std::unique_ptr<MPISendTensorCall>> active_sends;

  while (1) {
    MPI_Status status;

    // Check for incoming Tensor requests
    RecvTensorRequest request;
    if (ProbeForData(TAG_REQTENSOR, &status, &request)) {
      this->AddRequest(request, status.MPI_SOURCE);
    }

    // Check for incoming Tensor reply
    MPIRecvTensorResponse mRes;
    if (ProbeForData(TAG_SENDTENSOR, &status, &mRes)) {
      const int64 step_id = mRes.step_id();
      std::string key = mRes.key();

      std::shared_ptr<MPIRequestTensorCall> call;
      GetRecvCall(step_id, key, &call);
      call->recv_call_(mRes);
      RemoveRecvCall(step_id, key);
    }

    // Remove sends that have been completed
    active_sends.remove_if([](std::unique_ptr<MPISendTensorCall>& i) {
      return i->IsFinished();
    });

    // send a Tensor request
    RequestQueueEntry req;
    if (GetRequest(&req)) req.second();

    // Send a Tensor response
    SendQueueEntry send;
    if (GetResponse(&send)) {
      std::unique_ptr<MPISendTensorCall> p(send.second());
      active_sends.push_back(std::move(p));
    }

    //    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
}

}  // namespace tensorflow
#endif  // TENSORFLOW_USE_MPI
