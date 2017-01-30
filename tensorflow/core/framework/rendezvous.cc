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

#include <functional>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
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
  src_device.set(buf_.data() + (b.src_device.data() - b_base),
                 b.src_device.size());
  src = b.src;
  src_incarnation = b.src_incarnation;
  dst_device.set(buf_.data() + (b.dst_device.data() - b_base),
                 b.dst_device.size());
  dst = b.dst;
  edge_name.set(buf_.data() + (b.edge_name.data() - b_base),
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
    out->src_device.set(parts[0].data(), parts[0].size());
    out->dst_device.set(parts[2].data(), parts[2].size());
    out->edge_name.set(parts[3].data(), parts[3].size());
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
    bool notified = WaitForNotificationWithTimeout(&n, timeout_ms);
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

class LocalRendezvousImpl : public Rendezvous {
 public:
  explicit LocalRendezvousImpl(bool tolerate_dup_recv)
      : tolerate_dup_recv_(tolerate_dup_recv) {}

  Status Send(const ParsedKey& key, const Args& send_args, const Tensor& val,
              const bool is_dead) override {
    DoneCallback waiter = nullptr;
    Args recv_args;
    uint64 key_hash = KeyHash(key.FullKey());
    VLOG(2) << "Send " << this << " " << key_hash << " " << key.FullKey();
    {
      mutex_lock l(mu_);
      if (!status_.ok()) {
        return status_;
      }
      Item* item = nullptr;
      Table::iterator iter = table_.find(key_hash);
      if (iter == table_.end()) {
        // There is no waiter for this message. Insert the message
        // into the waiters table. The waiter will pick it up when
        // arrives.
        item = new Item;
        item->waiter = nullptr;
        item->value = val;
        item->is_dead = is_dead;
        if (send_args.device_context) {
          send_args.device_context->Ref();
          item->send_dev_context = send_args.device_context;
        }
        item->recv_dev_context = nullptr;

        // The allocator attributes of item->value.
        item->send_alloc_attrs = send_args.alloc_attrs;

        CHECK(table_.insert({key_hash, item}).second);
        return Status::OK();
      } else {
        item = iter->second;

        if (item->waiter == nullptr) {
          // There is already a message in the table under the key.
          // Should not happen unless it has a waiter.
          return errors::Aborted("Duplicated send: ", key.FullKey());
        }
        // Mark item as complete.
        item->has_been_recvd = true;

        // Get item->waiter function into waiter and set item->waiter to null
        std::swap(item->waiter, waiter);
        DCHECK(item->waiter == nullptr);
        DCHECK(waiter != nullptr);

        // The ref on recv_dev_context transfers below.
        recv_args.device_context = item->recv_dev_context;
        recv_args.alloc_attrs = item->recv_alloc_attrs;
        item->recv_dev_context = nullptr;
        if (tolerate_dup_recv_) {
          item->value = val;
          item->is_dead = is_dead;
          if (send_args.device_context) {
            send_args.device_context->Ref();
            item->send_dev_context = send_args.device_context;
          }
          item->send_alloc_attrs = send_args.alloc_attrs;
        }
      }
    }  // mutex
    // Notify the waiter by invoking its done closure, outside scope
    // of the table lock.
    waiter(Status::OK(), send_args, recv_args, val, is_dead);
    if (recv_args.device_context) recv_args.device_context->Unref();
    return Status::OK();
  }

  void RecvAsync(const ParsedKey& key, const Args& recv_args,
                 DoneCallback done) override {
    uint64 key_hash = KeyHash(key.FullKey());
    VLOG(2) << "Recv " << this << " " << key_hash << " " << key.FullKey();
    mu_.lock();
    if (!status_.ok()) {
      // Rendezvous has been aborted.
      Status s = status_;
      mu_.unlock();
      done(s, Args(), recv_args, Tensor(), false);
      return;
    }
    Table::iterator iter = table_.find(key_hash);
    if (iter != table_.end()) {
      Item* item = iter->second;
      if (item->has_been_recvd && !tolerate_dup_recv_) {
        mu_.unlock();
        done(errors::Aborted("Duplicated recv: ", key.FullKey()), Args(),
             recv_args, Tensor(), false);
      } else if (item->waiter == nullptr || tolerate_dup_recv_) {
        // A message has already arrived and is stored in the table
        // under this key.  Consumes the message and invokes the done
        // closure.
        Tensor v = item->value;
        if (!tolerate_dup_recv_) {
          item->value = Tensor();
        }
        item->has_been_recvd = true;
        // Before dropping the table lock, capture the item values.
        // DeviceContext is only non-null for non-CPU devices.
        // If we capture the send_dev_context, we need to hold a ref on
        // it.  Our caller will have a ref on the recv_dev_context,
        // which is not in our table.
        DeviceContext* send_dev_context = item->send_dev_context;
        if (send_dev_context) send_dev_context->Ref();
        bool is_dead = item->is_dead;
        Args send_args;
        send_args.device_context = item->send_dev_context;
        send_args.alloc_attrs = item->send_alloc_attrs;
        mu_.unlock();
        done(Status::OK(), send_args, recv_args, v, is_dead);
        if (send_dev_context) send_dev_context->Unref();
      } else {
        // Already have a waiter in the waiters table under this key,
        // which should not happen.
        mu_.unlock();
        done(errors::Aborted("Duplicated recv: ", key.FullKey()), Args(),
             recv_args, Tensor(), false);
      }
      return;
    }
    // Waiting for a message that has not arrived yet. Insert into the
    // waiting table. The done closure will be invoked when the
    // message arrives.
    Item* item = new Item;
    item->waiter = std::move(done);
    item->recv_alloc_attrs = recv_args.alloc_attrs;
    if (recv_args.device_context) {
      item->recv_dev_context = recv_args.device_context;
      item->recv_dev_context->Ref();
    }
    CHECK(table_.insert({key_hash, item}).second);
    mu_.unlock();
    return;
  }

  void StartAbort(const Status& status) override {
    CHECK(!status.ok());
    std::vector<Item*> items;
    {
      mutex_lock l(mu_);
      if (!status_.ok()) return;
      status_ = status;
      items.reserve(table_.size());
      for (const auto& p : table_) items.push_back(p.second);
      table_.clear();
    }
    for (Item* item : items) {
      if (item->waiter != nullptr) {
        item->waiter(status, Args(), Args(), Tensor(), false);
      }
      delete item;
    }
  }

 private:
  typedef LocalRendezvousImpl ME;
  const bool tolerate_dup_recv_;

  struct Item {
    DoneCallback waiter = nullptr;
    Tensor value;
    bool is_dead = false;
    bool has_been_recvd = false;
    DeviceContext* send_dev_context = nullptr;
    DeviceContext* recv_dev_context = nullptr;
    AllocatorAttributes send_alloc_attrs;
    AllocatorAttributes recv_alloc_attrs;

    ~Item() {
      if (send_dev_context) {
        send_dev_context->Unref();
      }
      if (recv_dev_context) {
        recv_dev_context->Unref();
      }
    }
  };
  // We key the hash table by KeyHash of the Rendezvous::CreateKey string
  static uint64 KeyHash(const StringPiece& k) {
    return Hash64(k.data(), k.size());
  }

  typedef gtl::FlatMap<uint64, Item*> Table;

  // TODO(zhifengc): shard table_.
  mutex mu_;
  Table table_ GUARDED_BY(mu_);
  Status status_;

  ~LocalRendezvousImpl() override {
    for (auto i : table_) {
      delete i.second;
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(LocalRendezvousImpl);
};

Rendezvous* NewLocalRendezvous(bool tolerate_dup_recv) {
  return new LocalRendezvousImpl(tolerate_dup_recv);
}

}  // end namespace tensorflow
