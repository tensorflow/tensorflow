/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_RMA_LOCAL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_RMA_LOCAL_H_

#include "tensorflow/core/common_runtime/buf_rendezvous.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/rendezvous.h"

namespace tensorflow {

// Basic implementation of PerStepCollectiveRemoteAccess.
class CollectiveRemoteAccessLocal : public CollectiveRemoteAccess {
 public:
  CollectiveRemoteAccessLocal(const DeviceMgr* dev_mgr,
                              DeviceResolverInterface* dev_resolver,
                              int64_t step_id)
      : dev_mgr_(dev_mgr),
        dev_resolver_(dev_resolver),
        buf_rendezvous_(step_id, dev_mgr),
        step_id_(step_id) {}

  ~CollectiveRemoteAccessLocal() override = default;

  void StartAbort(const Status& s) override;

  void RecvFromPeer(const string& peer_device, const string& peer_task,
                    bool peer_is_local, const string& key, Device* to_device,
                    DeviceContext* to_device_ctx,
                    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
                    const DeviceLocality& client_locality,
                    int dev_to_dev_stream_index,
                    CancellationManager* cancellation_manager,
                    const StatusCallback& done) override;

  void PostToPeer(const string& peer_device, const string& peer_task,
                  const string& key, Device* from_device,
                  DeviceContext* from_device_ctx,
                  const AllocatorAttributes& from_alloc_attr,
                  const Tensor* from_tensor,
                  const DeviceLocality& client_locality,
                  CancellationManager* cancellation_manager,
                  const StatusCallback& done) override;

  void CheckPeerHealth(const string& peer_task, int64_t timeout_in_ms,
                       const StatusCallback& done) override;

  BufRendezvous* buf_rendezvous() override { return &buf_rendezvous_; }

  // Copy utility that always copies bytes from src to dst even if
  // they are on the same device, unlike CopyTensor::ViaDMA which will
  // just change the dst buffer pointer in that case.
  static void MemCpyAsync(DeviceContext* src_dev_ctx,
                          DeviceContext* dst_dev_ctx, Device* src_dev,
                          Device* dst_dev, const AllocatorAttributes& src_attr,
                          const AllocatorAttributes& dst_attr,
                          const Tensor* src, Tensor* dst,
                          int dev_to_dev_stream_index,
                          const StatusCallback& done);

 protected:
  const DeviceMgr* dev_mgr_;               // not owned
  DeviceResolverInterface* dev_resolver_;  // not owned
  BufRendezvous buf_rendezvous_;
  int64 step_id_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_RMA_LOCAL_H_
