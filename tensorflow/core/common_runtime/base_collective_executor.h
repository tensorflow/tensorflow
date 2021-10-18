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

#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/buf_rendezvous.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/unbounded_work_queue.h"

namespace tensorflow {
class CollectiveImplementation;
class DeviceMgr;
class Device;

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
  virtual int64_t ChunkBytes(int i) const = 0;

  // Generate a CPU RAM scalar tensor of the same DataType as the
  // backing tensor with the given integer value.
  virtual Tensor Scalar(int v) const = 0;

  // Generate a scalar tensor of same DataType and on the same device
  // as the backing tensor.
  virtual Tensor Scalar(Allocator* a,
                        const AllocationAttributes& attr) const = 0;

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
  static int64_t AlignedChunkElts(int64_t elt_bytes, int64_t total_elts,
                                  int64_t num_chunks);
};

// Create a CollectiveAdaptor wrapping 'output', specialized to its
// data-type and shape.  If align_chunks == true then chunk size may
// be larger than output->NumElements() / num_chunks and one or more
// of the suffix chunks may be empty.  Chunks will be arranged to start
// and end on alignment boundaries.  If align_chunks == false then
// output->NumElements() % num_chunks must be 0 and all chunks will
// have exactly the same size, ignoring alignment issues.
CollectiveAdapter* MakeCollectiveAdapter(Tensor* output, int num_chunks,
                                         Allocator* allocator,
                                         bool align_chunks = true);

// Default implementation of CollectiveExecutor.  Delegates the actual
// work of moving data to a class specialized for the operation type,
// arguments and device+interconnect topology.
class BaseCollectiveExecutor : public CollectiveExecutor {
 public:
  BaseCollectiveExecutor(CollectiveExecutorMgrInterface* cem,
                         CollectiveRemoteAccess* remote_access, int64_t step_id,
                         const DeviceMgr* dev_mgr,
                         std::shared_ptr<UnboundedWorkQueue> work_queue)
      : CollectiveExecutor(cem),
        step_id_(step_id),
        dev_mgr_(dev_mgr),
        remote_access_(remote_access),
        work_queue_(std::move(work_queue)) {}

  ~BaseCollectiveExecutor() override;

  void StartAbort(const Status& s) override TF_LOCKS_EXCLUDED(status_mu_);

  void ExecuteAsync(OpKernelContext* ctx, const CollectiveParams* col_params,
                    const string& exec_key, StatusCallback done) override;

  void CompleteParamsAsync(const DeviceAttributes& device, CollectiveParams* cp,
                           CancellationManager* cancel_mgr,
                           StatusCallback done) override;

  CollectiveRemoteAccess* remote_access() override {
    return remote_access_.get();
  }

  void RunClosure(std::function<void()> closure) override {
    work_queue_->Schedule(std::move(closure));
  }

  // If we need to enforce an ordering on any portion of collective
  // implementation, and the ordering is encoded via attribute on the collective
  // op, this function will block until all dependencies for this collective
  // have completed.
  void WaitForDependencies(const CollectiveParams& col_params) override;
  // Record that this collective has completed the portion of the implementation
  // that needs to be ordered wrt other collectives, to unblock any of its
  // dependent ops.
  void UnblockDependencies(const CollectiveParams& col_params) override;

 protected:
  const int64_t step_id_;
  const DeviceMgr* dev_mgr_;  // Not owned.
  std::unique_ptr<CollectiveRemoteAccess> remote_access_;
  // Ownership of `work_queue_` is shared between `this` and
  // `CollectiveExecutorMgr`.
  std::shared_ptr<UnboundedWorkQueue> work_queue_;
  mutex launch_mu_;
  condition_variable launch_cv_;
  // collective instance key -> number of local devices for which NCCL ops have
  // been launched.
  std::unordered_map<int32, int32> launched_ TF_GUARDED_BY(launch_mu_);
  mutex status_mu_;
  Status status_ TF_GUARDED_BY(status_mu_);

 private:
  Status CreateCollective(const CollectiveParams& col_params,
                          CollectiveImplementationInterface** col_impl);
  // Check if all ops on which this collective depends on have launched.
  bool CheckDependencies(const CollectiveParams& col_params)
      TF_EXCLUSIVE_LOCKS_REQUIRED(launch_mu_);
  // Tries to return the status that is the original error. It returns the
  // aborted status if the collective executor is aborted.
  Status GetStatus(const Status& s) TF_LOCKS_EXCLUDED(status_mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_BASE_COLLECTIVE_EXECUTOR_H_
