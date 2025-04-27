/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_COMMON_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_COMMON_H_

#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/distributed_runtime/rpc/grpc_util.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {

// GenericCachingChannelCache that caches results to FindWorkerChannel() calls.
// To use instantiate with the type of channel cache needed.
// GenericCachingChannelCache allows using multiple channels to communiate with
// same target to provide throughput gains. When multiple channels exist for
// the same target they are chosen in a simple round robin fashion on each call
// to FindWorkerChannel.
template <typename ChannelCacheT>
class GenericCachingChannelCache : public ChannelCacheT {
 public:
  explicit GenericCachingChannelCache(int num_channels_per_target)
      : num_channels_per_target_(
            num_channels_per_target > 0 ? num_channels_per_target : 1) {}

  ~GenericCachingChannelCache() override {}

  SharedGrpcChannelPtr FindWorkerChannel(const string& target) override {
    {
      absl::MutexLock l(&mu_);
      auto iter = channels_.find(target);
      if (iter != channels_.end()) {
        return GetNextChannelPtrAndUpdateState(iter->second);
      }
    }
    ChannelState new_chan_state;
    for (int indx = 0; indx < num_channels_per_target_; indx++) {
      auto ch = FindChannelOnce(target);
      if (!ch) return nullptr;
      new_chan_state.channels.push_back(ch);
    }
    new_chan_state.last_used = num_channels_per_target_ - 1;

    {
      absl::MutexLock l(&mu_);
      typename absl::flat_hash_map<string, ChannelState>::iterator iter;
      bool was_inserted;
      std::tie(iter, was_inserted) = channels_.insert({target, new_chan_state});
      VLOG(2) << "Channel cache for target: " << target
              << " Size: " << new_chan_state.channels.size()
              << " insertion: " << was_inserted;
      return GetNextChannelPtrAndUpdateState(iter->second);
    }
  }

 protected:
  // Find the ClientChannel for "target".  Only called when no channel was
  // found in the channels_ cache for "target".  A non nullptr result will be
  // cached in channels_.
  virtual SharedGrpcChannelPtr FindChannelOnce(const string& target) = 0;

 private:
  struct ChannelState {
    std::vector<SharedGrpcChannelPtr> channels;
    int last_used;
  };

  // Should be called with mu_ held.
  SharedGrpcChannelPtr GetNextChannelPtrAndUpdateState(
      ChannelState& chan_state) {
    // Following statement is marked as Crash OK as this is an invariant of
    // code flow in this class.
    CHECK_EQ(chan_state.channels.size(), num_channels_per_target_);  // Crash OK
    chan_state.last_used =
        (chan_state.last_used + 1) % num_channels_per_target_;
    return chan_state.channels[chan_state.last_used];
  }

  const int num_channels_per_target_;
  // TODO(zhifengc): Eviction when the map becomes too big.
  absl::Mutex mu_;
  absl::flat_hash_map<string, ChannelState> channels_ TF_GUARDED_BY(mu_);
};

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_RPC_GRPC_CHANNEL_COMMON_H_
