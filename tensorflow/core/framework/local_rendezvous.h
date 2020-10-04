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
  explicit LocalRendezvous(Rendezvous* owner) : rc_owner_(owner) {}
  ~LocalRendezvous();

  Status Send(const Rendezvous::ParsedKey& key,
              const Rendezvous::Args& send_args, const Tensor& val,
              const bool is_dead);
  void RecvAsync(const Rendezvous::ParsedKey& key,
                 const Rendezvous::Args& recv_args,
                 Rendezvous::DoneCallback done);
  void StartAbort(const Status& status);

 private:
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

  // Pointer to the owner class of this LocalRendezvous if it is refcounted.
  const Rendezvous* rc_owner_;

  // TODO(zhifengc): shard table_.
  mutex mu_;
  Table table_ TF_GUARDED_BY(mu_);
  Status status_ TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(LocalRendezvous);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_LOCAL_RENDEZVOUS_H_
