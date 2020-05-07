/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_RING_ALG_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_RING_ALG_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/framework/collective.h"

namespace tensorflow {
class Device;

// Basic ring-algorithm implementation to be further specialized
// for specific collective functions.
class RingAlg : public CollectiveImplementationInterface {
 public:
  explicit RingAlg(CollectiveType type, const string& name);
  ~RingAlg() override {}

  // Establishes the requested number of subdivision permutations based on the
  // ring order implicit in the device order.
  Status InitializeCollectiveParams(CollectiveParams* col_params) override;

  // Initializes members of CollectiveContext not yet initialized, i.e. device
  // and device_locality.  Also saves the CollectiveContext in this object.
  Status InitializeCollectiveContext(CollectiveContext* col_ctx) override;

  // No-op for ring alg.
  Status InitializeCollectiveGroupRuntimeDetails(
      CollGroupRuntimeDetails*) override {
    return Status::OK();
  }

 protected:
  // Called when a bad status is received that implies we should terminate
  // execution and return a bad status.
  void StartAbort(const Status& s);
  void Finish(bool ok);

  // Current status of a RingField
  enum RingFieldAction {
    RF_INIT = 0,    // Just initialized for a pass
    RF_RECV,        // Recv pending
    RF_REDUCE,      // Reduce pending
    RF_FINALIZE,    // FinalOp pending
    RF_SEND_READY,  // Ready to send
    RF_SEND,        // Send pending
    RF_DONE,        // No more work
  };

  // Tracks progress of actions on a single subfield of the entire tensor.
  struct RingField {
    int16 chunk_idx;     // major division index
    int16 subdiv_idx;    // minor division index
    int16 sc_idx;        // subchunk index
    int16 rank;          // rank within subdiv permutation
    int16 recv_dev_idx;  // dev from which value should be recv'd
    RingFieldAction action;
    bool second_pass;
    bool recv_is_remote = false;
    bool send_is_remote = false;
    bool do_send = false;   // is the value sent in this pass?
    bool do_recv = false;   // is the value recv'd in this pass?
    bool is_final = false;  // is the last field in the pass for this rank
    Tensor chunk;           // alias to field values
    Tensor tmp_chunk;
    Status status;
    string DebugString() const;
  };
  virtual void InitRingField(RingField* rf, int chunk_idx, int subdiv_idx,
                             int field_idx);
  void AdvanceToSecondPass(RingField* rf);
  void DispatchSend(RingField* rf, const StatusCallback& done);
  void DispatchRecv(RingField* rf, const StatusCallback& done);

  // For constructing log messages for debugging.
  string FieldState();
  string TensorDebugString(const Tensor& tensor);

  // Producer/Consumer Queue of RingField structs.
  class PCQueue {
   public:
    void Enqueue(RingField* rf);
    RingField* Dequeue();

   private:
    mutex pcq_mu_;
    condition_variable cv_;
    int waiter_count_ TF_GUARDED_BY(pcq_mu_) = 0;
    std::deque<RingField*> deque_ TF_GUARDED_BY(pcq_mu_);
  };

  const CollectiveType type_;
  const string name_;
  CollectiveContext* col_ctx_;          // Not owned
  const CollectiveParams* col_params_;  // Not owned
  StatusCallback done_;
  int group_size_;
  int num_subdivs_;
  Tensor group_size_tensor_;
  Notification group_size_tensor_ready_;
  std::unique_ptr<CollectiveAdapter> ca_;
  mutex status_mu_;
  Status status_ TF_GUARDED_BY(status_mu_);
  std::vector<RingField> rfv_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_RING_ALG_H_
