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

#include "tensorflow/core/framework/local_rendezvous.h"
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
    return OkStatus();
  }
  return errors::InvalidArgument("Invalid  rendezvous key: ", key);
}

RendezvousInterface::~RendezvousInterface() {}

Status RendezvousInterface::Recv(const ParsedKey& key, const Args& recv_args,
                                 Tensor* val, bool* is_dead,
                                 int64_t timeout_ms) {
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
    int64_t timeout_us = timeout_ms * 1000;
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

Status RendezvousInterface::Recv(const ParsedKey& key, const Args& args,
                                 Tensor* val, bool* is_dead) {
  const int64_t no_timeout = 0;
  return Recv(key, args, val, is_dead, no_timeout);
}

namespace {
class LocalRendezvousWrapper : public Rendezvous {
 public:
  LocalRendezvousWrapper() : impl_(this) {}

  Status Send(const ParsedKey& key, const Args& send_args, const Tensor& val,
              const bool is_dead) override {
    return impl_.Send(key, send_args, val, is_dead);
  }

  void RecvAsync(const ParsedKey& key, const Args& recv_args,
                 DoneCallback done) override {
    impl_.RecvAsync(key, recv_args, std::move(done));
  }

  void StartAbort(const Status& status) override { impl_.StartAbort(status); }

 private:
  LocalRendezvous impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(LocalRendezvousWrapper);
};
}  // namespace

Rendezvous* NewLocalRendezvous() { return new LocalRendezvousWrapper; }

}  // end namespace tensorflow
