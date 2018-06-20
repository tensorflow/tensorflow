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
      : BaseRemoteRendezvous(env, step_id), rdma_mgr_(rdma_mgr) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~RdmaRemoteRendezvous() override {}
  RdmaMgr* rdma_mgr_;

  TF_DISALLOW_COPY_AND_ASSIGN(RdmaRemoteRendezvous);
};

void RdmaRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
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
  string key(std::move(parsed.FullKey().ToString()));
  string key_with_step_id = VerbsUtil::AppendStepidToKey(key, step_id_);

  Device* dst_dev;
  s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_dev);
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor(), true);
    return;
  }

  RdmaTensorRequest* request =
      rc->InsertTensorRequest(key, step_id_, dst_dev, recv_args, done);
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
