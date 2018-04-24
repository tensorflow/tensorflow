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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_BASE_COLLECTIVE_EXECUTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_BASE_COLLECTIVE_EXECUTOR_H_

#include <string>
#include "tensorflow/core/common_runtime/buf_rendezvous.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {
class Broadcaster;
class DeviceMgr;
class RingReducer;

// Helper interface that aliases regular subfields of a Tensor as separate
// Tensors for in-place update.
class CollectiveAdapter {
 public:
  virtual ~CollectiveAdapter() {}

  // Move the backing tensor to 'output' with its original storage and
  // shape. After this call this CollectiveAdapter object should be
  // deleted immediately without calling any of its other methods.
  virtual void ConsumeFinalValue(Tensor* output) = 0;

  // const access to entire intermediate value for debugging
  virtual const Tensor& Value() const = 0;

  // Returns tensor for chunk i which aliases the backing buffer.
  virtual Tensor ChunkAlias(int i) = 0;

  // Returns tensor allocated on the same device but with its own
  // separate backing buffer.  Will have same type and size as
  // chunk i.
  virtual Tensor TempChunk(int i) const = 0;

  // Bytes in chunk i
  virtual int64 ChunkBytes(int i) const = 0;

  // Generate a CPU RAM scalar tensor of the same DataType as the
  // backing tensor with the given integer value.
  virtual Tensor Scalar(int v) const = 0;

  // Generate a scalar tensor of same DataType and on the same device
  // as the backing tensor.
  virtual Tensor Scalar(Allocator* a) const = 0;

  // Debugging string describing buffer location
  virtual string TBounds(const Tensor& t) const = 0;

  virtual string DebugString() const = 0;

  // Computes the number of elements per alias chunk tensor.
  //
  // A CHECK in tensor.cc expects that the memory buffer backing a
  // Tensor will be aligned according to EIGEN_MAX_ALIGN_BYTES.  To
  // ensure that all chunk aliasing Tensors maintain this alignment we
  // need to pick a chunk size that preserves it.  Note than in extreme
  // cases (impractical, but possible with very small tensors) one or
  // more tail chunks can end up emptby.
  static int64 AlignedChunkElts(int64 elt_bytes, int64 total_elts,
                                int64 num_chunks);
};

// Create a CollectiveAdaptor wrapping 'output', specialized to its
// data-type and shape.
CollectiveAdapter* MakeCollectiveAdapter(Tensor* output, int num_chunks,
                                         Allocator* allocator);

// Default implementation of CollectiveExecutor.  Delegates the actual
// work of moving data to a class specialized for the operation type,
// arguments and device+interconnect topology.
class BaseCollectiveExecutor : public CollectiveExecutor {
 public:
  BaseCollectiveExecutor(CollectiveExecutorMgrInterface* cem,
                         PerStepCollectiveRemoteAccess* remote_access,
                         int64 step_id, const DeviceMgr* dev_mgr)
      : CollectiveExecutor(cem),
        step_id_(step_id),
        dev_mgr_(dev_mgr),
        remote_access_(remote_access) {}

  ~BaseCollectiveExecutor() override;

  void StartAbort(const Status& s) override;

  void ExecuteAsync(OpKernelContext* ctx, const CollectiveParams& col_params,
                    const string& exec_key, StatusCallback done) override;

  PerStepCollectiveRemoteAccess* remote_access() override {
    return remote_access_.get();
  }

  void RecvFromPeer(const string& peer_device, const string& peer_task,
                    bool peer_is_local, const string& key, Device* to_device,
                    DeviceContext* to_device_ctx,
                    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
                    const DeviceLocality& client_locality,
                    const StatusCallback& done) override {
    remote_access_->RecvFromPeer(peer_device, peer_task, peer_is_local, key,
                                 to_device, to_device_ctx, to_alloc_attr,
                                 to_tensor, client_locality, done);
  }

  void PostToPeer(const string& peer_device, const string& peer_task,
                  const string& key, Device* from_device,
                  DeviceContext* from_device_ctx,
                  const AllocatorAttributes& from_alloc_attr,
                  const Tensor* from_tensor,
                  const DeviceLocality& client_locality,
                  const StatusCallback& done) override {
    remote_access_->PostToPeer(peer_device, peer_task, key, from_device,
                               from_device_ctx, from_alloc_attr, from_tensor,
                               client_locality, done);
  }

 protected:
  const int64 step_id_;
  const DeviceMgr* dev_mgr_;  // Not owned.
  std::unique_ptr<PerStepCollectiveRemoteAccess> remote_access_;

 private:
  RingReducer* CreateReducer(OpKernelContext* ctx,
                             OpKernelContext::Params* params,
                             const CollectiveParams& col_params,
                             const string& exec_key, int64 step_id,
                             const Tensor* input, Tensor* output,
                             string* error);

  Broadcaster* CreateBroadcaster(OpKernelContext* ctx,
                                 OpKernelContext::Params* params,
                                 const CollectiveParams& col_params,
                                 const string& exec_key, int64 step_id,
                                 Tensor* output, string* error);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_BASE_COLLECTIVE_EXECUTOR_H_
