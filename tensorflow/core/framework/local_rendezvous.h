/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_LOCAL_RENDEZVOUS_H_
#define TENSORFLOW_CORE_FRAMEWORK_LOCAL_RENDEZVOUS_H_

#include <memory>
#include <optional>
#include <vector>

#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Implements the basic logic of matching Send and Recv operations. See
// RendezvousInterface for more details.
//
// NOTE: Most users will use a class that wraps LocalRendezvous, such as
// IntraProcessRendezvous or RemoteRendezvous. This class does not implement
// RendezvousInterface because virtual dispatch to LocalRendezvous methods
// is not expected to be needed.
class LocalRendezvous {
 public:
  // If the class wrapping LocalRendezvous is refcounted (i.e., extending
  // Rendezvous), pass in its pointer in constructor so the LocalRendezvous
  // can make sure it outlives the async recv requests.
  // Pass in nullptr if the wrapping class is not refcounted.
  explicit LocalRendezvous(Rendezvous* owner, int num_shards)
      : num_buckets_(num_shards > 0 ? num_shards : 1),
        rc_owner_(owner),
        table_buckets_(std::make_unique<TableBucket[]>(num_buckets_)) {}
  ~LocalRendezvous();

  absl::Status Send(const Rendezvous::ParsedKey& key,
                    const Rendezvous::Args& send_args, const Tensor& val,
                    bool is_dead);
  void RecvAsync(const Rendezvous::ParsedKey& key,
                 const Rendezvous::Args& recv_args,
                 Rendezvous::DoneCallback done);
  void StartAbort(const absl::Status& status);
  absl::Status status();

  // Releases all the references to the aborted rendezvous. Used in unit tests.
  static void ReleaseAbortedRendezvous() {
    mutex_lock l(aborted_rendezs_mu_);
    aborted_rendezs_.clear();
  }

 private:
  void DoAbort(const absl::Status& status);

  tsl::core::RefCountPtr<Rendezvous> GetOwnerRefCountPtr();

  struct Item;

  // By invariant, the item queue under each key is of the form
  //   [item.type == kSend]* meaning each item is a sent message.
  // or
  //   [item.type == kRecv]* meaning each item is a waiter.
  struct ItemQueue {
    void push_back(Item* item);

    Item* head = nullptr;
    Item* tail = nullptr;
  };

  typedef gtl::FlatMap<uint64, ItemQueue> Table;

  const int num_buckets_;
  // Pointer to the owner class of this LocalRendezvous if it is refcounted,
  // nullptr otherwise.
  Rendezvous* rc_owner_;

  struct TableBucket {
    mutex mu;
    Table table TF_GUARDED_BY(mu);

    // Track the number of pening callbacks using a counter.
    int pending_callback_counter TF_GUARDED_BY(mu) = 0;
    condition_variable pending_callback_cond_var TF_GUARDED_BY(mu);
  };

  // Immutable set of buckets. This uses less memory than std::vector.
  const std::unique_ptr<TableBucket[]> table_buckets_;
  mutex mu_;
  absl::Status status_ TF_GUARDED_BY(mu_);

  // We deliberately leak one reference of the aborted rendezvous here, so that
  // they won't be destructed, and lose the status_.
  // This is necessary because subsequent calls to RendezvousMgr::Find() will
  // return the aborted rendezvous, and proper errors will be propagated.
  // TODO(hhb): find a better way to manage rendezvous lifespan.
  static mutex& aborted_rendezs_mu_;
  static std::vector<tsl::core::RefCountPtr<Rendezvous> >& aborted_rendezs_
      TF_GUARDED_BY(aborted_rendezs_mu_);

  LocalRendezvous(const LocalRendezvous&) = delete;
  void operator=(const LocalRendezvous&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_LOCAL_RENDEZVOUS_H_
