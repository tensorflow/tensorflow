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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_BUF_RENDEZVOUS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_BUF_RENDEZVOUS_H_

#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
class Device;
class DeviceContext;
class DeviceMgr;
class Tensor;

// EXPERIMENTAL: RDMA oriented producer/consumer rendezvous on a local
// Tensor value for which DMAHelper::CanUseDMA() is true, i.e. dense
// numeric types.  Similar to Rendezvous but never owns a Ref on the
// tensor, instead it uses an explicit callback to the producer when
// the consumer side is finished with the value.  This allows the
// producer to perform in-place updates on the source buffer or to take
// other actions that depend on knowing the consumer has passed a certain
// execution point.
class BufRendezvous {
 public:
  explicit BufRendezvous(uint64 step_id, const DeviceMgr* dev_mgr)
      : step_id_(step_id), dev_mgr_(dev_mgr) {}

  virtual ~BufRendezvous();

  // Inform all waiting parties that this BufRendezvous is defunct because of
  // an error Status interrupting the Step.
  void StartAbort(const absl::Status& s);

  struct Hook;
  // Provided by the consumer to be called when access to the buffer
  // is available.  If the Status arg is not OK, then hook will not
  // be populated.  Ownership of Hook passes to consumer with the
  // callback.
  typedef std::function<void(const absl::Status&, Hook*)> ConsumerCallback;
  // Provided by the producer to be called when the consumer has finished
  // reading the buffer and will no longer access it.
  typedef std::function<void(const absl::Status&)> ProducerCallback;

  struct Hook {
    Device* prod_dev;
    DeviceContext* prod_ctx;
    const Tensor* prod_value;
    AllocatorAttributes prod_attr;
    ProducerCallback prod_cb;
    ConsumerCallback cons_cb;
    CancellationManager* cancellation_manager;
    CancellationToken cancellation_token;
    explicit Hook(CancellationManager* cancellation_manager,
                  CancellationToken cancellation_token)
        : prod_dev(nullptr),
          prod_ctx(nullptr),
          prod_value(nullptr),
          prod_cb(nullptr),
          cons_cb(nullptr),
          cancellation_manager(cancellation_manager),
          cancellation_token(cancellation_token) {}
    string DebugString() const;
  };

  // Called to advertise availability of a Tensor value corresponding
  // to key.  That value must stay valid until done is called.
  //
  // If a non-null cancellation manager is provided, this function registers a
  // callback to delete the hook and invoke provider/consumer callbacks with
  // cancelled error.
  void ProvideBuf(const string& key, Device* dev, DeviceContext* dev_ctx,
                  const Tensor* v, const AllocatorAttributes& attr,
                  const ProducerCallback& done,
                  CancellationManager* cancellation_manager);

  // Called to request access to a Tensor value corresponding to key.
  // Consumer is provided with a Hook as soon as available.
  //
  // This function also checks that the current incarnation number of the
  // `device` that produced this value matches the `incarnation` expected by the
  // consumer, and invokes `done` with `FailedPrecondition` status and
  // `nullptr` hook if it does not match.
  //
  // If a non-null cancellation manager is provided, this function registers a
  // callback to delete the hook and invoke provider/consumer callbacks with
  // cancelled error.
  virtual void ConsumeBuf(const string& key, const string& device,
                          const uint64 incarnation,
                          const ConsumerCallback& done,
                          CancellationManager* cancellation_manager);

  // Cancel the rendezvous entry corresponding to `key`.  Triggered by the
  // cancellation manager. No-op if the rendezvous was already successful.
  void CancelHook(const string& key);

  // Consumer must call this function when it's done reading the Hook provided
  // by the ConsumerCallback.  This function will invoke the producer callback
  // and then delete h.
  static void DoneWithHook(Hook* h);

  // Write the current contents of the table to the INFO log.
  void LogContents();

 protected:
  const uint64 step_id_;
  const DeviceMgr* const dev_mgr_;  // Not owned.
  mutex mu_;
  absl::Status status_ TF_GUARDED_BY(mu_);
  typedef absl::flat_hash_map<string, Hook*> HookTable;
  HookTable hook_table_ TF_GUARDED_BY(mu_);

  void PurgeTable(const absl::Status& s, HookTable* table);
};
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_BUF_RENDEZVOUS_H_
