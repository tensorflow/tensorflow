/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CHANNEL_TRACKER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CHANNEL_TRACKER_H_

#include <map>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Tracks channels between computations in the XLA service. Channels
// are associated with a unique handle and can be resolved from the handle for
// later use.
//
// TODO(b/34027823): Destruct channels when all the associated computations that
// communicate via each channel are destructed.
class ChannelTracker {
 public:
  ChannelTracker();

  // A struct that keeps the current status of each channel. has_sender and
  // receiver_count fields are initialized with false and 0 respectively when
  // the struct is created and are updated by RegisterSend() and RegisterRecev()
  // as Send or Recv instructions using the channel are requested.
  struct Channel {
    bool has_sender;
    int64_t receiver_count;
    ChannelHandle::ChannelType type;
  };

  // Creates a new Channel object and returns the corresponding
  // ChannelHandle for it.
  StatusOr<ChannelHandle> NewChannel(ChannelHandle::ChannelType type);

  // Informs that the given channel handle is used for a Send operation.
  // Returns an error status if the handle is already used by another Send.
  Status RegisterSend(const ChannelHandle& handle);

  // Informs that the given channel handle is used for a Recv operation.
  // Returns an error status if the handle is already used by another Recv.
  Status RegisterRecv(const ChannelHandle& handle);

 private:
  // Bumps the next_channel_ number and returns the allocated number
  // wrapped in a ChannelHandle.
  ChannelHandle AllocateHandle(ChannelHandle::ChannelType type)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(channel_mutex_);

  Status RegisterSendInternal(const ChannelHandle& handle)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(channel_mutex_);

  Status RegisterRecvInternal(const ChannelHandle& handle)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(channel_mutex_);

  // Guards the channel mapping.
  absl::Mutex channel_mutex_;

  // The next sequence number to assign to a channel.
  int64_t next_channel_ ABSL_GUARDED_BY(channel_mutex_);

  // Mapping from ChannelHandle value to the corresponding registered
  // Channel object.
  absl::flat_hash_map<int64_t, Channel> opaque_to_channel_
      ABSL_GUARDED_BY(channel_mutex_);

  ChannelTracker(const ChannelTracker&) = delete;
  ChannelTracker& operator=(const ChannelTracker&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CHANNEL_TRACKER_H_
