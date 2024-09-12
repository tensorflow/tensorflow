/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CHANNEL_TRACKER_H_
#define XLA_SERVICE_CHANNEL_TRACKER_H_

#include "absl/status/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Tracks channels between computations in the XLA service. Channels
// are associated with a unique handle and can be resolved from the handle for
// later use.
//
// TODO(b/34027823): Destruct channels when all the associated computations that
// communicate via each channel are destructed.
class ChannelTracker {
 public:
  ChannelTracker() = default;

  // Creates a new Channel object and returns the corresponding
  // ChannelHandle for it.
  absl::StatusOr<ChannelHandle> NewChannel(ChannelHandle::ChannelType type);

 private:
  // Guards the channel mapping.
  absl::Mutex channel_mutex_;

  // The next sequence number to assign to a channel.
  int64_t next_channel_ ABSL_GUARDED_BY(channel_mutex_) = 1;

  ChannelTracker(const ChannelTracker&) = delete;
  ChannelTracker& operator=(const ChannelTracker&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_CHANNEL_TRACKER_H_
