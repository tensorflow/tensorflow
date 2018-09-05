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

#ifdef TENSORFLOW_USE_VERBS

#include "tensorflow/contrib/verbs/rdma_rendezvous_mgr.h"
#include <unordered_set>
#include "tensorflow/contrib/verbs/verbs_util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

class RdmaRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  RdmaRemoteRendezvous(const WorkerEnv* env, int64 step_id, RdmaMgr* rdma_mgr)
      : BaseRemoteRendezvous(env, step_id), step_id_(step_id), rdma_mgr_(rdma_mgr) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~RdmaRemoteRendezvous() override {}
  int64 step_id_;
  RdmaMgr* rdma_mgr_;

  TF_DISALLOW_COPY_AND_ASSIGN(RdmaRemoteRendezvous);
};

void RdmaRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  int64 start_usec = Env::Default()->NowMicros();
  Status s;
  // parse src_name and dst_name
  string src_name, dst_name, unused;
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &src_name,
                                        &unused) ||
      !DeviceNameUtils::SplitDeviceName(parsed.dst_device, &dst_name,
                                        &unused)) {
    s = errors::Internal("Could not parse src or dst name.");
  }
  if (!s.ok()) {
    LOG(ERROR) << "s is not ok, error code " << s.error_message();
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  
  CHECK(dst_name.compare(rdma_mgr_->local_worker()) == 0);
  RdmaChannel* rc = rdma_mgr_->FindChannel(src_name);
  string key(parsed.FullKey());
  string key_with_step_id = VerbsUtil::AppendStepidToKey(key, step_id_);

  Device* dst_dev;
  s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_dev);
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), true);
    return;
  }

  // Type-specialized logging for this method.
  WorkerCacheLogger* logger = rdma_mgr_->GetLogger();

  bool logging_active = logger->LoggingActive();
  DoneCallback wrapper_done;
  const DoneCallback* cb_to_use;
  if(!logging_active) { 
    cb_to_use = &done; //No additional work to do, so just use done directly
  } else {
     int64 step_id = step_id_;
     wrapper_done = [this, logger, step_id, parsed, done, start_usec] (const Status& s, const Args& send_args, const Args& recv_args,
  	                                                               const Tensor& recv_tensor, const bool is_dead) {
      
      if (logger->LoggingActive()) {
        int64 end_usec = Env::Default()->NowMicros();
        int64 bytes = recv_tensor.TotalBytes();
        logger->RecordRecvTensor(step_id_, start_usec, end_usec,
                                 parsed.edge_name.ToString(), parsed.src_device.ToString(), parsed.dst_device.ToString(),
                                 bytes);
      }
      done(s, send_args, recv_args, recv_tensor, is_dead);      
    };
    cb_to_use = &wrapper_done;
  }

  RdmaTensorRequest* request =
      rc->InsertTensorRequest(key, step_id_, dst_dev, recv_args, *cb_to_use);
  request->Start();
}

RdmaRendezvousMgr::RdmaRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {}

BaseRemoteRendezvous* RdmaRendezvousMgr::Create(int64 step_id,
                                                const WorkerEnv* worker_env) {
  return new RdmaRemoteRendezvous(worker_env, step_id, rdma_mgr_);
}

}  // end namespace tensorflow

#endif
