#include "tensorflow/core/framework/rendezvous.h"

#include <unordered_map>
#include <utility>

#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {

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
  return strings::StrCat(src_device, ";", strings::FpToString(src_incarnation),
                         ";", dst_device, ";", name, ";", frame_iter.frame_id,
                         ":", frame_iter.iter_id);
}

/* static */
Status Rendezvous::ParseKey(const string& key, ParsedKey* out) {
  // TODO(zhifengc): This code is not fast enough.
  std::vector<string> parts = str_util::Split(key, ';');
  if (parts.size() == 5 &&
      DeviceNameUtils::ParseFullName(parts[0], &out->src) &&
      strings::StringToFp(parts[1], &out->src_incarnation) &&
      DeviceNameUtils::ParseFullName(parts[2], &out->dst) &&
      !parts[3].empty()) {
    out->src_device = parts[0];
    out->dst_device = parts[2];
    out->edge_name = parts[3];
    return Status::OK();
  }
  return errors::InvalidArgument("Invalid rendezvous key: ", key);
}

Rendezvous::~Rendezvous() {}

Status Rendezvous::Recv(const string& key, const Args& recv_args, Tensor* val,
                        bool* is_dead) {
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
  n.WaitForNotification();
  return ret;
}

class LocalRendezvousImpl : public Rendezvous {
 public:
  explicit LocalRendezvousImpl(bool tolerate_dup_recv)
      : tolerate_dup_recv_(tolerate_dup_recv) {}

  Status Send(const string& key, const Args& send_args, const Tensor& val,
              const bool is_dead) override {
    VLOG(2) << "Send " << this << " " << key;
    DoneCallback waiter = nullptr;
    Args recv_args;
    {
      mutex_lock l(mu_);
      if (!status_.ok()) {
        return status_;
      }
      Item* item = nullptr;
      Table::iterator iter = table_.find(key);
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

        CHECK(table_.insert({key, item}).second);
        return Status::OK();
      } else {
        item = iter->second;
        if (item->waiter == nullptr) {
          // There is already a message in the table under the key.
          // Should not happen unless it has a waiter.
          return errors::Aborted("Duplicated send: ", key);
        }
        // Mark item as complete.
        item->has_been_recvd = true;
        waiter = item->waiter;
        item->waiter = nullptr;
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

  void RecvAsync(const string& key, const Args& recv_args,
                 DoneCallback done) override {
    VLOG(2) << "Recv " << this << " " << key;
    mu_.lock();
    if (!status_.ok()) {
      // Rendezvous has been aborted.
      Status s = status_;
      mu_.unlock();
      done(s, Args(), recv_args, Tensor(), false);
      return;
    }
    Table::iterator iter = table_.find(key);
    if (iter != table_.end()) {
      Item* item = iter->second;
      if (item->has_been_recvd && !tolerate_dup_recv_) {
        mu_.unlock();
        done(errors::Aborted("Duplicated recv: ", key), Args(), recv_args,
             Tensor(), false);
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
        mu_.unlock();
        Args send_args;
        send_args.device_context = item->send_dev_context;
        send_args.alloc_attrs = item->send_alloc_attrs;
        done(Status::OK(), send_args, recv_args, v, is_dead);
        if (send_dev_context) send_dev_context->Unref();
      } else {
        // Already have a waiter in the waiters table under this key,
        // which should not happen.
        mu_.unlock();
        done(errors::Aborted("Duplicated recv: ", key), Args(), recv_args,
             Tensor(), false);
      }
      return;
    }
    // Waiting for a message that has not arrived yet. Insert into the
    // waiting table. The done closure will be invoked when the
    // message arrives.
    Item* item = new Item;
    item->waiter = done;
    if (recv_args.device_context) {
      item->recv_dev_context = recv_args.device_context;
      item->recv_alloc_attrs = recv_args.alloc_attrs;
      item->recv_dev_context->Ref();
    }
    CHECK(table_.insert({key, item}).second);
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
  typedef std::unordered_map<string, Item*> Table;

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
