/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_OUTFEED_RECEIVER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_OUTFEED_RECEIVER_H_

#include <memory>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

class OutfeedReceiverImpl;

// Implements a multithreaded receiver of outfeeds from devices.
class OutfeedReceiver {
 public:
  // A callback takes: device, consumer id, received.
  using Callback =
      std::function<void(PjRtDevice*, uint32_t, std::shared_ptr<Literal>)>;

  // Constructs the receiver for the given clients and callback function.
  //
  // Args:
  //   callback: a function to be called when an outfeed is ready for
  //     processing.
  //   clients: the clients for whose devices to listen.
  //   max_callback_queue_size_bytes: the maximum number of bytes for all
  //     received outfeeds queued to be processed. When this limit is reached
  //     we pause receiving outfeeds from devices.
  OutfeedReceiver(Callback callback, absl::Span<PjRtClient* const> clients,
                  ssize_t max_callback_queue_size_bytes);

  OutfeedReceiver(const OutfeedReceiver&) = delete;
  OutfeedReceiver& operator=(const OutfeedReceiver&) = delete;

  // Blocks until all data has been received from devices and all data
  // in the queue has been passed to Python.
  ~OutfeedReceiver();

  // Starts the listener threads and the callback thread.
  void Start();

  // Adds to the computation builder the outfeed of the arrays.
  // Has the side-effect of registering the sent shape for the consumer_id.
  // Returns error status if the outfeed shape is different than the
  // previously used shape for the same consumer_id or the consumer id is
  // invalid.
  StatusOr<XlaOp> AddOutfeedToBuilder(XlaBuilder* builder, XlaOp token,
                                      uint32_t consumer_id,
                                      std::vector<XlaOp> arrays);

 private:
  std::unique_ptr<OutfeedReceiverImpl> p_impl_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_OUTFEED_RECEIVER_H_
