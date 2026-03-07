/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(LocalRendezvous, StressAbort) {
  Rendezvous::ParsedKey key;
  TF_EXPECT_OK(Rendezvous::ParseKey(
      Rendezvous::CreateKey("/job:mnist/replica:1/task:2/cpu:0", 7890,
                            "/job:mnist/replica:1/task:2/cpu:1", "foo",
                            FrameAndIter(0, 0)),
      &key));
  for (size_t i = 0; i < 1000; ++i) {
    LocalRendezvous rendezvous(nullptr, 1);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < 100; ++i) {
      threads.emplace_back([&] {
        rendezvous.Send(key, Rendezvous::Args(), Tensor(), false).IgnoreError();
      });
      threads.emplace_back([&] {
        absl::Notification done;
        rendezvous.RecvAsync(key, Rendezvous::Args(),
                             [&](const absl::Status&, const Rendezvous::Args&,
                                 const Rendezvous::Args&, const Tensor&,
                                 bool) { done.Notify(); });
        done.WaitForNotification();
      });
    }
    // TODO: uncomment to trigger the deadlock.
    // rendezvous.StartAbort(absl::CancelledError("Cancelled"));
    for (auto& thread : threads) {
      thread.join();
    }
  }
}

}  // namespace
}  // namespace tensorflow
