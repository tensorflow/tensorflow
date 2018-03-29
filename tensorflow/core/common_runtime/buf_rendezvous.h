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
#ifndef TENSORFLOW_COMMON_RUNTIME_BUF_RENDEZVOUS_H_
#define TENSORFLOW_COMMON_RUNTIME_BUF_RENDEZVOUS_H_

#include <functional>
#include <string>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
class Device;
class DeviceContext;
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
  explicit BufRendezvous(uint64 step_id) : step_id_(step_id) {}

  ~BufRendezvous();

  // Inform all all waiting parties that this BufRendezvous is defunct
  // because of an error Status interrupting the Step.
  void StartAbort(const Status& s);

  struct Hook;
  // Provided by the consumer to be called when access to the buffer
  // is available.  If the Status arg is not OK, then hook will not
  // be populated.  Ownership of Hook passes to consumer with the
  // callback.
  typedef std::function<void(const Status&, Hook*)> ConsumerCallback;
  // Provided by the producer to be called when the consumer has finished
  // reading the buffer and will no longer access it.
  typedef std::function<void(const Status&)> ProducerCallback;

  struct Hook {
    Device* prod_dev;
    DeviceContext* prod_ctx;
    const Tensor* prod_value;
    AllocatorAttributes prod_attr;
    ProducerCallback prod_cb;
    ConsumerCallback cons_cb;
    Hook()
        : prod_dev(nullptr),
          prod_ctx(nullptr),
          prod_value(nullptr),
          prod_cb(nullptr),
          cons_cb(nullptr) {}
    string DebugString() const;
  };

  // Called to advertise availability of a Tensor value corresponding
  // to key.  That value must stay valid until done is called.
  void ProvideBuf(const string& key, Device* dev, DeviceContext* dev_ctx,
                  const Tensor* v, const AllocatorAttributes& attr,
                  const ProducerCallback& done);

  // Called to request access to a Tensor value corresponding to key.
  // Consumer is provide with a Hook as soon as availble.
  void ConsumeBuf(const string& key, const ConsumerCallback& done);

  // Consumer must call this function when it's done reading the Hook provided
  // by the ConsumerCallback.  This function will invoke the producer callback
  // and then delete h.
  static void DoneWithHook(Hook* h);

  // Write the current contents of the table to the INFO log.
  void LogContents();

 protected:
  const uint64 step_id_;
  mutex mu_;
  Status status_ GUARDED_BY(mu_);
  typedef gtl::FlatMap<string, Hook*> HookTable;
  HookTable hook_table_ GUARDED_BY(mu_);

  void PurgeTable(const Status& s, HookTable* table);
};
}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_BUF_RENDEZVOUS_H_
