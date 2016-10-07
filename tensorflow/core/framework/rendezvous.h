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

#ifndef TENSORFLOW_FRAMEWORK_RENDEZVOUS_H_
#define TENSORFLOW_FRAMEWORK_RENDEZVOUS_H_

#include <string>

#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

// A Rendezvous is an abstraction for passing a Tensor
// from a producer to a consumer, where the consumer may safely
// request the Tensor before or after it has been produced.  A
// producer never blocks when using a Rendezvous.  A consumer has the
// choice of making a blocking call or providing a callback: in either
// case, the consumer receives the Tensor as soon as it is available.
//
// A Rendezvous key encodes a single <producer, consumer> pair.  It is
// an error to call Send() or Recv*() more than once with the same
// key.
class Rendezvous : public core::RefCounted {
 public:
  struct Args {
    DeviceContext* device_context = nullptr;
    AllocatorAttributes alloc_attrs;
  };

  // Constructs a rendezvous key for the tensor of "name" sent from
  // "src_device" to "dst_device". The tensor is generated in the frame
  // and iteration specified by "frame_iter".
  static string CreateKey(const string& src_device, uint64 src_incarnation,
                          const string& dst_device, const string& name,
                          const FrameAndIter& frame_iter);

  // Parses the key constructed by CreateKey and parse src/dst device
  // names into structures respectively.
  struct ParsedKey {
    StringPiece src_device;
    DeviceNameUtils::ParsedName src;
    uint64 src_incarnation = 0;
    StringPiece dst_device;
    DeviceNameUtils::ParsedName dst;
    StringPiece edge_name;

    ParsedKey() {}
    ParsedKey(const ParsedKey& b) { *this = b; }

    ParsedKey& operator=(const ParsedKey& b);
    StringPiece FullKey() const { return buf_; }

   private:
    friend class Rendezvous;
    friend class SendOp;
    friend class RecvOp;
    string buf_;
  };
  static Status ParseKey(StringPiece key, ParsedKey* out);

  // The caller is a tensor producer and it sends a message (a tensor
  // "val" and a bool "is_dead") under the given "key".
  //
  // {val, is_dead} is bundled as a message sent and received.
  // Typically, is_dead is set by some control flow nodes
  // (e.g., a not-take branch).  args is passed by Send to the
  // Recv function to communicate any information that the Recv
  // function might need.  This is typically only necessary for
  // Send/Recv on the same worker.
  //
  // Send() never blocks.
  virtual Status Send(const ParsedKey& key, const Args& args, const Tensor& val,
                      const bool is_dead) = 0;

  // Callback provided by a tensor consumer waiting on the rendezvous.
  // It will be invoked when the tensor is available, or when a non-OK
  // status arises in the production of that tensor.  It also gets
  // two Rendezvous::Args, one provided by the sender, the other by the
  // receiver, which may be needed when a non-CPU device is in use
  // by either side.
  typedef std::function<void(const Status&, const Args&, const Args&,
                             const Tensor&, const bool)>
      DoneCallback;

  virtual void RecvAsync(const ParsedKey& key, const Args& args,
                         DoneCallback done) = 0;

  // Synchronous wrapper for RecvAsync.
  Status Recv(const ParsedKey& key, const Args& args, Tensor* val,
              bool* is_dead);

  // Aborts all pending and future Send/Recv with the given "status".
  //
  // StartAbort() does not wait for ongoing calls to finish.
  // REQUIRES: !status.ok()
  virtual void StartAbort(const Status& status) = 0;

 protected:
  ~Rendezvous() override;
};

// Returns a Rendezvous instance that is limited to use only by
// producers and consumers in the local process.  The caller assumes
// ownership of one Ref() on the returned object.
//
// If "tolerate_dup_recv" is true, then the Rendezvous will retain
// already Recv'd values and make them available to duplicate Recv
// calls.  This may be useful if the RPC layer is not reliable, but
// comes at the cost of higher memory consumption.
Rendezvous* NewLocalRendezvous(bool tolerate_dup_recv = false);

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_RENDEZVOUS_H_
