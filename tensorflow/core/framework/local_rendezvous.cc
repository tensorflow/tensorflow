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

#include "tensorflow/core/framework/local_rendezvous.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "tensorflow/core/activity_watcher/activity.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/platform/mutex.h"
#include "tsl/platform/refcount.h"

namespace tensorflow {

// Represents a blocked Send() or Recv() call in the rendezvous.
// Item hols a reference to the owner rendezvous, to make
// sure the local rendezvous outlives any pending requests and callbacks.
struct LocalRendezvous::Item {
  enum Type { kSend = 0, kRecv = 1 };

  Item(tsl::core::RefCountPtr<Rendezvous> rc_owner, Rendezvous::Args send_args,
       const Tensor& value, bool is_dead,
       activity_watcher::ActivityScope activity_scope)
      : Item(std::move(rc_owner), send_args, kSend, std::move(activity_scope)) {
    send_state.value.Init(value);
    send_state.is_dead = is_dead;
  }

  Item(tsl::core::RefCountPtr<Rendezvous> rc_owner, Rendezvous::Args recv_args,
       Rendezvous::DoneCallback waiter, CancellationToken cancellation_token,
       activity_watcher::ActivityScope activity_scope)
      : Item(std::move(rc_owner), recv_args, kRecv, std::move(activity_scope)) {
    recv_state.waiter.Init(std::move(waiter));
    recv_state.cancellation_token = cancellation_token;
  }

  ~Item() {
    if (args.device_context) {
      args.device_context->Unref();
    }
    if (type == kSend) {
      send_state.value.Destroy();
    } else {
      recv_state.waiter.Destroy();
    }
  }

  const Rendezvous::Args args;
  const Type type;
  tsl::core::RefCountPtr<Rendezvous> rc_owner;

  // Link to next item in an ItemQueue.
  Item* next = nullptr;

  // The validity of `send_state` or `recv_state` is determined by `type ==
  // kSend` or `type == kRecv` respectively.
  union {
    struct {
      ManualConstructor<Tensor> value;
      bool is_dead;
    } send_state;
    struct {
      ManualConstructor<Rendezvous::DoneCallback> waiter;
      CancellationToken cancellation_token;
    } recv_state;
  };

  activity_watcher::ActivityScope scope;

 private:
  Item(tsl::core::RefCountPtr<Rendezvous> rc_owner, Rendezvous::Args args,
       Type type, activity_watcher::ActivityScope activity_scope)
      : args(args),
        type(type),
        rc_owner(std::move(rc_owner)),
        scope(std::move(activity_scope)) {
    if (args.device_context) {
      args.device_context->Ref();
    }
  }
};

void LocalRendezvous::ItemQueue::push_back(Item* item) {
  if (TF_PREDICT_TRUE(head == nullptr)) {
    // The queue is empty.
    head = item;
    tail = item;
  } else {
    DCHECK_EQ(tail->type, item->type);
    tail->next = item;
    tail = item;
  }
}

LocalRendezvous::~LocalRendezvous() {
  // Before destroying this rendezvous instance, make sure all the done-callback
  // calls have finished and the tensors have been released from the queue.
  bool table_not_empty = false;
  for (int i = 0; i < num_buckets_; ++i) {
    auto& bucket = table_buckets_[i];
    {
      mutex_lock l(bucket.mu);
      while (bucket.pending_callback_counter != 0) {
        bucket.pending_callback_cond_var.wait_for(
            l, std::chrono::milliseconds(50));
      }
    }
    if (!bucket.table.empty()) {
      table_not_empty = true;
    }
  }
  if (table_not_empty) {
    DoAbort(absl::CancelledError("LocalRendezvous deleted"));
  }
}

namespace {
class KeyHash {
 public:
  // We use salted hashing (see go/totw/189) to reduce the likelihood of hash
  // collisions. Note: if the strings are long, then it would be better to
  // generate both hashes while iterating once over the string, but in practice,
  // it's hard to beat absl::Hash, which is highly optimized.
  explicit KeyHash(absl::string_view key) {
    // We use absl::HashOf instead of tsl::Hash64 because it's faster, and we
    // don't need a deterministic hash function.
    bucket_hash_ = absl::HashOf(key);
    constexpr int kArbitraryConstant = 100;
    // Note: it's important that the arbitrary constant is passed to HashOf
    // before `key` so that the different initial hash state cascades while
    // hashing the string contents.
    table_hash_ = absl::HashOf(kArbitraryConstant, key);
  }
  uint64_t bucket(uint64_t num_buckets) const {
    return bucket_hash_ % num_buckets;
  }
  uint64_t table_hash() const { return table_hash_; }
  std::string ToString() const {
    return absl::StrFormat("bucket_hash: %#x, table_hash: %#x", bucket_hash_,
                           table_hash_);
  }

 private:
  uint64_t bucket_hash_;
  uint64_t table_hash_;
};
}  // namespace

absl::Status LocalRendezvous::Send(const Rendezvous::ParsedKey& key,
                                   const Rendezvous::Args& send_args,
                                   const Tensor& val, const bool is_dead) {
  KeyHash key_hash = KeyHash(key.FullKey());
  DVLOG(2) << "Send " << this << " " << key_hash.ToString() << " "
           << key.FullKey();

  if (is_dead) {
    static auto* rendezvous_dead_values_sent = monitoring::Counter<2>::New(
        "/tensorflow/core/rendezvous_dead_values_sent",
        "The number of dead values sent between a pair of devices.",
        "send_device", "recv_device");
    rendezvous_dead_values_sent
        ->GetCell(std::string(key.src_device), std::string(key.dst_device))
        ->IncrementBy(1);
  }

  int bucket_index = key_hash.bucket(num_buckets_);
  auto& bucket = table_buckets_[bucket_index];
  bucket.mu.lock();

  if (auto s = status(); !s.ok()) {
    bucket.mu.unlock();
    return s;
  }

  auto it = bucket.table.insert({key_hash.table_hash(), ItemQueue()}).first;
  ItemQueue* queue = &it->second;
  if (queue->head == nullptr || queue->head->type == Item::kSend) {
    // There is no waiter for this message. Append the message
    // into the queue. The waiter will pick it up when arrives.
    // Only send-related fields need to be filled.
    // TODO(b/143786186): Investigate moving the allocation of `Item` outside
    // the lock.
    auto rc_owner = tsl::core::GetNewRef(rc_owner_);
    DVLOG(2) << "Enqueue Send Item (key:" << key.FullKey() << "). ";
    activity_watcher::ActivityScope activity_scope(
        [&]() {
          return std::make_unique<activity_watcher::Activity>(
              "LocalRendezvous::Send",
              activity_watcher::ActivityCategory::kRendezvous,
              activity_watcher::Activity::Attributes{
                  {"Rendezvous", absl::StrFormat("%p", this)},
                  {"key", std::string(key.FullKey())},
                  {"key_hash", key_hash.ToString()},
              });
        },
        /*level=*/1);
    queue->push_back(new Item(std::move(rc_owner), send_args, val, is_dead,
                              std::move(activity_scope)));
    bucket.mu.unlock();
    return absl::OkStatus();
  }

  DVLOG(2) << "Consume Recv Item (key:" << key.FullKey() << "). ";
  // There is an earliest waiter to consume this message.
  Item* item = queue->head;

  // Delete the queue when the last element has been consumed.
  if (item->next == nullptr) {
    DVLOG(2) << "Clean up Send/Recv queue (key:" << key.FullKey() << "). ";
    bucket.table.erase(it);
  } else {
    queue->head = item->next;
  }
  bucket.pending_callback_counter++;
  // Invoke the done-callback, without holding the lock.
  bucket.mu.unlock();

  DCHECK_EQ(item->type, Item::kRecv);
  (*item->recv_state.waiter)(absl::OkStatus(), send_args, item->args, val,
                             is_dead);
  {
    mutex_lock l(bucket.mu);
    bucket.pending_callback_counter--;
    if (bucket.pending_callback_counter == 0) {
      bucket.pending_callback_cond_var.notify_all();
    }
  }
  // Delete the item at last since it may unref and destruct the rendezvous.
  delete item;
  return absl::OkStatus();
}

void LocalRendezvous::RecvAsync(const Rendezvous::ParsedKey& key,
                                const Rendezvous::Args& recv_args,
                                Rendezvous::DoneCallback done) {
  KeyHash key_hash = KeyHash(key.FullKey());
  DVLOG(2) << "Recv " << this << " " << key_hash.ToString() << " "
           << key.FullKey();
  tsl::core::RefCountPtr<Rendezvous> rc_keep_alive;

  int bucket_index = key_hash.bucket(num_buckets_);
  auto& bucket = table_buckets_[bucket_index];
  bucket.mu.lock();

  if (auto s = status(); !s.ok()) {
    bucket.mu.unlock();
    // Rendezvous has been aborted.
    done(s, Rendezvous::Args(), recv_args, Tensor(), false);
    return;
  }

  auto it = bucket.table.insert({key_hash.table_hash(), ItemQueue()}).first;
  ItemQueue* queue = &it->second;
  if (queue->head == nullptr || queue->head->type == Item::kRecv) {
    // There is no message to pick up.
    // Only recv-related fields need to be filled.
    CancellationManager* cm = recv_args.cancellation_manager;
    CancellationToken token = CancellationManager::kInvalidToken;
    bool already_cancelled = false;
    if (cm != nullptr) {
      // Take a reference for the cancellation callback so that it does not
      // access LocalRendezvous after it is destroyed. It's dropped either
      // at the end of the callback, or when callback registration fails,
      // or when the cancellation callback is cancelled.
      if (rc_owner_) {
        rc_owner_->Ref();
      }
      token = cm->get_cancellation_token();
      already_cancelled = !cm->RegisterCallback(token, [this, token, key_hash,
                                                        &bucket] {
        tsl::core::RefCountPtr<Rendezvous> rc_owner(rc_owner_);
        Item* item = nullptr;
        {
          mutex_lock l(bucket.mu);

          auto it = bucket.table.find(key_hash.table_hash());
          if (it != bucket.table.end()) {
            ItemQueue* queue = &it->second;
            // Find an item in the queue with a cancellation token that matches
            // `token`, and remove it.
            if (queue->head != nullptr && queue->head->type == Item::kRecv) {
              for (Item *prev = nullptr, *curr = queue->head; curr != nullptr;
                   prev = curr, curr = curr->next) {
                if (curr->recv_state.cancellation_token == token) {
                  item = curr;
                  if (queue->head->next == nullptr) {
                    // We have a single-element queue, so we can erase it from
                    // the table.
                    bucket.table.erase(it);
                  } else {
                    // Remove the current item from the queue.
                    if (curr == queue->head) {
                      DCHECK_EQ(prev, nullptr);
                      queue->head = curr->next;
                    } else {
                      DCHECK_NE(prev, nullptr);
                      prev->next = curr->next;
                    }
                    if (queue->tail == curr) {
                      queue->tail = prev;
                    }
                  }
                  break;
                }
              }
            }
          }
        }

        if (item != nullptr) {
          (*item->recv_state.waiter)(
              StatusGroup::MakeDerived(
                  absl::CancelledError("RecvAsync is cancelled.")),
              Rendezvous::Args(), item->args, Tensor(), /*is_dead=*/false);
          delete item;
        }
      });
    }
    if (already_cancelled) {
      if (queue->head == nullptr) {
        bucket.table.erase(it);
      }
      bucket.mu.unlock();
      done(StatusGroup::MakeDerived(
               absl::CancelledError("RecvAsync is cancelled.")),
           Rendezvous::Args(), recv_args, Tensor(), /*is_dead=*/false);
      if (rc_owner_) {
        rc_owner_->Unref();
      }
      return;
    }

    DVLOG(2) << "Enqueue Recv Item (key:" << key.FullKey() << "). ";

    // TODO(b/143786186): Investigate moving the allocation of `Item` outside
    // the lock.
    activity_watcher::ActivityScope activity_scope(
        [&]() {
          return std::make_unique<activity_watcher::Activity>(
              "LocalRendezvous::RecvAsync",
              activity_watcher::ActivityCategory::kRendezvous,
              activity_watcher::Activity::Attributes{
                  {"Rendezvous", absl::StrFormat("%p", this)},
                  {"key", std::string(key.FullKey())},
                  {"key_hash", key_hash.ToString()},
              });
        },
        /*level=*/1);
    auto rc_owner = tsl::core::GetNewRef(rc_owner_);
    if (cm != nullptr) {
      // NOTE(mrry): We must wrap `done` with code that deregisters the
      // cancellation callback before calling the `done` callback, because the
      // cancellation manager may no longer be live after `done` is called.
      queue->push_back(new Item(
          std::move(rc_owner), recv_args,
          [this, cm, token, done = std::move(done)](
              const absl::Status& s, const Rendezvous::Args& send_args,
              const Rendezvous::Args& recv_args, const Tensor& v, bool dead) {
            // TryDeregisterCallback returns true when the cancellation callback
            // is successfully deregistered. If it fails because the CM already
            // StartAbort, Unref will happen inside the cancellation callback
            // when called by the CM.
            if (cm->TryDeregisterCallback(token)) {
              if (rc_owner_) {
                rc_owner_->Unref();
              }
            }
            done(s, send_args, recv_args, v, dead);
          },
          token, std::move(activity_scope)));
    } else {
      queue->push_back(new Item(std::move(rc_owner), recv_args, std::move(done),
                                token, std::move(activity_scope)));
    }

    bucket.mu.unlock();
    return;
  }

  DVLOG(2) << "Consume Send Item (key:" << key.FullKey() << "). ";
  // A message has already arrived and is queued in the table under
  // this key.  Consumes the message and invokes the done closure.
  Item* item = queue->head;

  // Delete the queue when the last element has been consumed.
  if (item->next == nullptr) {
    DVLOG(2) << "Clean up Send/Recv queue (key:" << key.FullKey() << "). ";
    bucket.table.erase(it);
  } else {
    queue->head = item->next;
  }
  bucket.pending_callback_counter++;
  // Invoke the done-callback, without holding the lock.
  bucket.mu.unlock();

  DCHECK_EQ(item->type, Item::kSend);
  done(absl::OkStatus(), item->args, recv_args, *item->send_state.value,
       item->send_state.is_dead);
  {
    mutex_lock l(bucket.mu);
    bucket.pending_callback_counter--;
    if (bucket.pending_callback_counter == 0) {
      bucket.pending_callback_cond_var.notify_all();
    }
  }
  // Delete the item at last since it may unref and destruct the rendezvous.
  delete item;
}

mutex& LocalRendezvous::aborted_rendezs_mu_ = *new mutex();

std::vector<tsl::core::RefCountPtr<Rendezvous> >&
    LocalRendezvous::aborted_rendezs_ =
        *new std::vector<tsl::core::RefCountPtr<Rendezvous> >();

void LocalRendezvous::StartAbort(const absl::Status& status) {
  DoAbort(status);

  if (rc_owner_) {
    mutex_lock l(aborted_rendezs_mu_);
    aborted_rendezs_.push_back(tsl::core::GetNewRef(rc_owner_));
  }
}

void LocalRendezvous::DoAbort(const absl::Status& status) {
  CHECK(!status.ok());
  {
    mutex_lock l(mu_);
    status_.Update(status);
  }

  // OUT_OF_RANGE implies a normal end of sequence (e.g. for tf.data),
  // so we suppress the warning to avoid log noise.
  if (status.code() != absl::StatusCode::kOutOfRange) {
    LOG_EVERY_POW_2(WARNING)
        << "Local rendezvous is aborting with status: " << status;
  }

  // Keeps one Item to make sure the current rendezvous won't be destructed.
  std::unique_ptr<Item> to_delete;
  for (int i = 0; i < num_buckets_; ++i) {
    auto& bucket = table_buckets_[i];
    Table table;
    {
      mutex_lock l(bucket.mu);
      bucket.table.swap(table);
    }
    for (auto& p : table) {
      Item* item = p.second.head;
      DCHECK(item);  // we delete all empty lists from the table
      while (item != nullptr) {
        switch (item->type) {
          case Item::kRecv:
            (*item->recv_state.waiter)(status, Rendezvous::Args(),
                                       Rendezvous::Args(), Tensor(), false);
            LOG(INFO) << "Local rendezvous recv item cancelled. Key hash: "
                      << p.first;
            break;
          case Item::kSend:
            LOG(INFO) << "Local rendezvous send item cancelled. Key hash: "
                      << p.first;
            break;
        }
        to_delete.reset(item);
        item = item->next;
      }
    }
  }
}

absl::Status LocalRendezvous::status() {
  tf_shared_lock ml(mu_);
  return status_;
}

}  // namespace tensorflow
