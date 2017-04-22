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
  RdmaRemoteRendezvous(const WorkerEnv* env, const string& worker_name,
                       int64 step_id, RdmaMgr* rdma_mgr)
      : BaseRemoteRendezvous(env, worker_name, step_id, true),
        rdma_mgr_(rdma_mgr) {}

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
                                        &unused)) {
    s = errors::Internal("Could not parse src name.");
  }
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  if (!DeviceNameUtils::SplitDeviceName(parsed.dst_device, &dst_name,
                                        &unused)) {
    s = errors::Internal("Could not parse dst name.");
  }
  CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
  if (!s.ok()) {
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }
  CHECK(dst_name.compare(rdma_mgr_->local_worker()) == 0);
  RdmaChannel* rc = rdma_mgr_->FindChannel(src_name);
  string key(std::move(parsed.FullKey().ToString()));
  string key_with_step_id = VerbsUtil::AppendStepidToKey(key, step_id_);
  // insert callback
  rc->InsertRecvCallback(key_with_step_id, [this, key, key_with_step_id, rc,
                                            recv_args, parsed, done]() {
    Status s;
    Device* src_dev;
    s = env_->device_mgr->LookupDevice("CPU:0", &src_dev);
    CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
    if (!s.ok()) {
      done(s, Args(), recv_args, Tensor(), true);
      return;
    }
    Device* dst_dev;
    s = env_->device_mgr->LookupDevice(parsed.dst_device, &dst_dev);
    CHECK(s.ok()) << "s is not ok, error code " << s.error_message();
    if (!s.ok()) {
      done(s, Args(), recv_args, Tensor(), true);
      return;
    }
    RdmaBuffer* rb = rc->FindBuffer(key);
    RdmaMessage rm;
    CHECK(rb->size_ >= RdmaMessage::kMessageTotalBytes);
    RdmaMessage::ParseMessage(rm, rb->buffer_);
    CHECK(rm.type_ == RDMA_MESSAGE_TENSOR_WRITE);
    Tensor val;
    if (!rm.is_dead_) {
      void* input = static_cast<char*>(rb->buffer_) +
                    RdmaMessage::kTensorBufferStartIndex;
      TensorProto proto;
      CHECK(rm.tensor_bytes_ + RdmaMessage::kTensorBufferStartIndex <=
            rb->size_);
      CHECK(ParseProtoUnlimited(&proto, input, rm.tensor_bytes_))
          << "fail to parse proto from array";
      s = dst_dev->MakeTensorFromProto(proto, recv_args.alloc_attrs, &val);
    }

    rc->RemoveRecvCallback(key_with_step_id);
    // create message
    RdmaMessage br;
    br.type_ = RDMA_MESSAGE_BUFFER_IDLE;
    br.name_size_ = key.size();
    br.name_ = key;
    string message = RdmaMessage::CreateMessage(br);
    RdmaBuffer* tb = rc->tx_message_buffer_;
    tb->EnqueueItem(message);
    tb->SendNextItem();
    done(s, Args(), recv_args, val, rm.is_dead_);
  });
  // append key to message queue
  RdmaBuffer* rb = rc->tx_message_buffer_;
  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_TENSOR_REQUEST;
  rm.name_size_ = key.size();
  rm.name_ = key;
  rm.step_id_ = step_id_;
  string message = RdmaMessage::CreateMessage(rm);
  rb->EnqueueItem(message);
  rb->SendNextItem();
}

RdmaRendezvousMgr::RdmaRendezvousMgr(const WorkerEnv* env,
                                     const string& worker_name,
                                     WorkerCacheInterface* worker_cache)
    : BaseRendezvousMgr(env, worker_name) {}

BaseRemoteRendezvous* RdmaRendezvousMgr::Create(int64 step_id,
                                                const WorkerEnv* worker_env,
                                                const string& worker_name) {
  return new RdmaRemoteRendezvous(worker_env, worker_name, step_id, rdma_mgr_);
}

}  // end namespace tensorflow

#endif
