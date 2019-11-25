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

#include "tensorflow/core/framework/rendezvous.h"

#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

Rendezvous::ParsedKey& Rendezvous::ParsedKey::operator=(const ParsedKey& b) {
  const char* b_base = b.buf_.data();
  buf_ = b.buf_;
  src_device = StringPiece(buf_.data() + (b.src_device.data() - b_base),
                           b.src_device.size());
  src = b.src;
  src_incarnation = b.src_incarnation;
  dst_device = StringPiece(buf_.data() + (b.dst_device.data() - b_base),
                           b.dst_device.size());
  dst = b.dst;
  edge_name = StringPiece(buf_.data() + (b.edge_name.data() - b_base),
                          b.edge_name.size());
  return *this;
}

/*  static */
string Rendezvous::CreateKey(const string& src_device, uint64 src_incarnation,
                             const string& dst_device, const string& name,
                             const FrameAndIter& frame_iter) {
  // NOTE: ';' is not used in the device name's job name.
  //
  // We include both sender and receiver in the key to facilitate
  // debugging. For correctness, we only need to encode the receiver.
  //
  // "src_incarnation" is used to distinguish a worker when it
  // restarts.
  char buf[strings::kFastToBufferSize];
  return strings::StrCat(
      src_device, ";", strings::Uint64ToHexString(src_incarnation, buf), ";",
      dst_device, ";", name, ";", frame_iter.frame_id, ":", frame_iter.iter_id);
}

// Return the prefix of "*s" up to the next occurrence of "delim", or
// the whole remaining string if "delim" is not found.  "*s" is advanced
// past the string returned plus the delimiter (if found).
static StringPiece ConsumeNextPart(StringPiece* s, char delim) {
  for (size_t offset = 0; offset < s->size(); offset++) {
    if ((*s)[offset] == delim) {
      StringPiece result(s->data(), offset);
      s->remove_prefix(offset + 1);  // +1: remove delim, as well
      return result;
    }
  }
  // No delimiter found: return rest of string
  StringPiece result(s->data(), s->size());
  s->remove_prefix(s->size());
  return result;
}

/* static */
Status Rendezvous::ParseKey(StringPiece key, ParsedKey* out) {
  if (key.data() == out->buf_.data()) {
    // Caller used our buf_ string directly, so we don't need to copy.  (The
    // SendOp and RecvOp implementations do this, for example).
    DCHECK_EQ(key.size(), out->buf_.size());
  } else {
    // Make a copy that our StringPieces can point at a copy that will persist
    // for the lifetime of the ParsedKey object.
    out->buf_.assign(key.data(), key.size());
  }
  StringPiece s(out->buf_);
  StringPiece parts[5];
  for (int i = 0; i < 5; i++) {
    parts[i] = ConsumeNextPart(&s, ';');
  }
  if (s.empty() &&          // Consumed the whole string
      !parts[4].empty() &&  // Exactly five parts
      DeviceNameUtils::ParseFullName(parts[0], &out->src) &&
      strings::HexStringToUint64(parts[1], &out->src_incarnation) &&
      DeviceNameUtils::ParseFullName(parts[2], &out->dst) &&
      !parts[3].empty()) {
    out->src_device = StringPiece(parts[0].data(), parts[0].size());
    out->dst_device = StringPiece(parts[2].data(), parts[2].size());
    out->edge_name = StringPiece(parts[3].data(), parts[3].size());
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid  rendezvous key: ", key);
}

Rendezvous::~Rendezvous() {}

Status Rendezvous::Recv(const ParsedKey& key, const Args& recv_args,
                        Tensor* val, bool* is_dead, int64 timeout_ms) {
  Status ret;
  Notification n;
  RecvAsync(key, recv_args,
            [&ret, &n, val, is_dead](const Status& s, const Args& send_args,
                                     const Args& recv_args, const Tensor& v,
                                     const bool dead) {
              ret = s;
              *val = v;
              *is_dead = dead;
              n.Notify();
            });
  if (timeout_ms > 0) {
    int64 timeout_us = timeout_ms * 1000;
    bool notified = WaitForNotificationWithTimeout(&n, timeout_us);
    if (!notified) {
      return Status(error::DEADLINE_EXCEEDED,
                    "Timed out waiting for notification");
    }
  } else {
    n.WaitForNotification();
  }
  return ret;
}

Status Rendezvous::Recv(const ParsedKey& key, const Args& args, Tensor* val,
                        bool* is_dead) {
  const int64 no_timeout = 0;
  return Recv(key, args, val, is_dead, no_timeout);
}

namespace {
class LocalRendezvousImpl : public Rendezvous {
 public:
  explicit LocalRendezvousImpl() {}

  Status Send(const ParsedKey& key, const Args& send_args, const Tensor& val,
              const bool is_dead) override {
    uint64 key_hash = KeyHash(key.FullKey());
    DVLOG(2) << "Send " << this << " " << key_hash << " " << key.FullKey();

    mu_.lock();
    if (!status_.ok()) {
      // Rendezvous has been aborted.
      Status s = status_;
      mu_.unlock();
      return s;
    }

    ItemQueue* queue = &table_[key_hash];
    if (queue->head == nullptr || queue->head->type == Item::kSend) {
      // There is no waiter for this message. Append the message
      // into the queue. The waiter will pick it up when arrives.
      // Only send-related fields need to be filled.
      // TODO(b/143786186): Investigate moving the allocation of `Item` outside
      // the lock.
      DVLOG(2) << "Enqueue Send Item (key:" << key.FullKey() << "). ";
      queue->push_back(new Item(send_args, val, is_dead));
      mu_.unlock();
      return Status::OK();
    }

    DVLOG(2) << "Consume Recv Item (key:" << key.FullKey() << "). ";
    // There is an earliest waiter to consume this message.
    Item* item = queue->head;

    // Delete the queue when the last element has been consumed.
    if (item->next == nullptr) {
      DVLOG(2) << "Clean up Send/Recv queue (key:" << key.FullKey() << "). ";
      table_.erase(key_hash);
    } else {
      queue->head = item->next;
    }
    mu_.unlock();

    // Notify the waiter by invoking its done closure, outside the
    // lock.
    DCHECK_EQ(item->type, Item::kRecv);
    (*item->recv_state.waiter)(Status::OK(), send_args, item->args, val,
                               is_dead);
    delete item;
    return Status::OK();
  }

  void RecvAsync(const ParsedKey& key, const Args& recv_args,
                 DoneCallback done) override {
    uint64 key_hash = KeyHash(key.FullKey());
    DVLOG(2) << "Recv " << this << " " << key_hash << " " << key.FullKey();

    mu_.lock();
    if (!status_.ok()) {
      // Rendezvous has been aborted.
      Status s = status_;
      mu_.unlock();
      done(s, Args(), recv_args, Tensor(), false);
      return;
    }

    ItemQueue* queue = &table_[key_hash];
    if (queue->head == nullptr || queue->head->type == Item::kRecv) {
      // There is no message to pick up.
      // Only recv-related fields need to be filled.
      CancellationManager* cm = recv_args.cancellation_manager;
      CancellationToken token = CancellationManager::kInvalidToken;
      bool already_cancelled = false;
      if (cm != nullptr) {
        token = cm->get_cancellation_token();
        already_cancelled = !cm->RegisterCallback(token, [this, token,
                                                          key_hash] {
          Item* item = nullptr;
          {
            mutex_lock l(mu_);
            ItemQueue* queue = &table_[key_hash];
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
                    table_.erase(key_hash);
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

          if (item != nullptr) {
            (*item->recv_state.waiter)(
                StatusGroup::MakeDerived(
                    errors::Cancelled("RecvAsync is cancelled.")),
                Args(), item->args, Tensor(), /*is_dead=*/false);
            delete item;
          }
        });
      }
      if (already_cancelled) {
        mu_.unlock();
        done(StatusGroup::MakeDerived(
                 errors::Cancelled("RecvAsync is cancelled.")),
             Args(), recv_args, Tensor(), /*is_dead=*/false);
        return;
      }

      DVLOG(2) << "Enqueue Recv Item (key:" << key.FullKey() << "). ";

      // TODO(b/143786186): Investigate moving the allocation of `Item` outside
      // the lock.
      if (cm != nullptr) {
        // NOTE(mrry): We must wrap `done` with code that deregisters the
        // cancellation callback before calling the `done` callback, because the
        // cancellation manager may no longer be live after `done` is called.
        queue->push_back(new Item(
            recv_args,
            [cm, token, done = std::move(done)](
                const Status& s, const Args& send_args, const Args& recv_args,
                const Tensor& v, bool dead) {
              cm->TryDeregisterCallback(token);
              done(s, send_args, recv_args, v, dead);
            },
            token));
      } else {
        queue->push_back(new Item(recv_args, std::move(done), token));
      }

      mu_.unlock();
      return;
    }

    DVLOG(2) << "Consume Send Item (key:" << key.FullKey() << "). ";
    // A message has already arrived and is queued in the table under
    // this key.  Consumes the message and invokes the done closure.
    Item* item = queue->head;

    // Delete the queue when the last element has been consumed.
    if (item->next == nullptr) {
      DVLOG(2) << "Clean up Send/Recv queue (key:" << key.FullKey() << "). ";
      table_.erase(key_hash);
    } else {
      queue->head = item->next;
    }
    mu_.unlock();

    // Invoke done() without holding the table lock.
    DCHECK_EQ(item->type, Item::kSend);
    done(Status::OK(), item->args, recv_args, *item->send_state.value,
         item->send_state.is_dead);
    delete item;
  }

  void StartAbort(const Status& status) override {
    CHECK(!status.ok());
    Table table;
    {
      mutex_lock l(mu_);
      status_.Update(status);
      table_.swap(table);
    }
    for (auto& p : table) {
      Item* item = p.second.head;
      while (item != nullptr) {
        if (item->type == Item::kRecv) {
          (*item->recv_state.waiter)(status, Args(), Args(), Tensor(), false);
        }
        Item* to_delete = item;
        item = item->next;
        delete to_delete;
      }
    }
  }

 private:
  typedef LocalRendezvousImpl ME;

  // Represents a blocked Send() or Recv() call in the rendezvous.
  struct Item {
    enum Type { kSend = 0, kRecv = 1 };

    Item(Args send_args, const Tensor& value, bool is_dead)
        : Item(send_args, kSend) {
      send_state.value.Init(value);
      send_state.is_dead = is_dead;
    }

    Item(Args recv_args, DoneCallback waiter,
         CancellationToken cancellation_token)
        : Item(recv_args, kRecv) {
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

    const Args args;
    const Type type;

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
        ManualConstructor<DoneCallback> waiter;
        CancellationToken cancellation_token;
      } recv_state;
    };

   private:
    Item(Args args, Type type) : args(args), type(type) {
      if (args.device_context) {
        args.device_context->Ref();
      }
    }
  };

  // We key the hash table by KeyHash of the Rendezvous::CreateKey string
  static uint64 KeyHash(const StringPiece& k) {
    return Hash64(k.data(), k.size());
  }

  // By invariant, the item queue under each key is of the form
  //   [item.type == kSend]* meaning each item is a sent message.
  // or
  //   [item.type == kRecv]* meaning each item is a waiter.
  struct ItemQueue {
    void push_back(Item* item) {
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

    Item* head = nullptr;
    Item* tail = nullptr;
  };
  typedef gtl::FlatMap<uint64, ItemQueue> Table;

  // TODO(zhifengc): shard table_.
  mutex mu_;
  Table table_ GUARDED_BY(mu_);
  Status status_ GUARDED_BY(mu_);

  ~LocalRendezvousImpl() override {
    if (!table_.empty()) {
      StartAbort(errors::Cancelled("LocalRendezvousImpl deleted"));
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(LocalRendezvousImpl);
};
}  // namespace

Rendezvous* NewLocalRendezvous() { return new LocalRendezvousImpl(); }

}  // end namespace tensorflow
