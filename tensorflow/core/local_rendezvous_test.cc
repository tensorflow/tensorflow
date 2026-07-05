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

#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/refcount.h"

namespace tensorflow {
namespace {

TEST(LocalRendezvous, Stress) {
  Rendezvous::ParsedKey key;
  TF_EXPECT_OK(Rendezvous::ParseKey(
      Rendezvous::CreateKey("/job:mnist/replica:1/task:2/cpu:0", 7890,
                            "/job:mnist/replica:1/task:2/cpu:1", "foo",
                            FrameAndIter(0, 0)),
      &key));
  for (size_t i = 0; i < 10000; ++i) {
    std::atomic<size_t> recv_count = 0;
    std::unique_ptr<tsl::CancellationManager> cm;
    if (i % 3 == 0) {
      cm = std::make_unique<tsl::CancellationManager>();
    }
    tsl::core::RefCountPtr<Rendezvous> rendezvous(NewLocalRendezvous());
    Rendezvous::Args args{
        .cancellation_manager = cm.get(),
    };
    constexpr size_t kNumOps = 10;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < kNumOps; ++i) {
      threads.emplace_back(
          [&] { rendezvous->Send(key, args, Tensor(), false).IgnoreError(); });
      threads.emplace_back([&] {
        rendezvous->RecvAsync(key, args,
                              [&](const absl::Status&, const Rendezvous::Args&,
                                  const Rendezvous::Args&, const Tensor&,
                                  bool) { recv_count++; });
      });
    }
    if (i % 2 == 0) {
      threads.emplace_back(
          [&] { rendezvous->StartAbort(absl::CancelledError("Cancelled")); });
    }
    if (i % 4 == 0 && cm) {
      cm->StartCancelWithStatus(absl::CancelledError("Cancelled"));
    }
    for (auto& thread : threads) {
      thread.join();
    }
    ASSERT_EQ(kNumOps, recv_count);
  }
}

TEST(LocalRendezvous, CancelDestroyRace) {
  Rendezvous::ParsedKey key;
  TF_EXPECT_OK(Rendezvous::ParseKey(
      Rendezvous::CreateKey("/job:mnist/replica:1/task:2/cpu:0", 7890,
                            "/job:mnist/replica:1/task:2/cpu:1", "foo",
                            FrameAndIter(0, 0)),
      &key));
  for (size_t i = 0; i < 10000; ++i) {
    std::atomic<size_t> recv_count = 0;
    tsl::CancellationManager cm;
    tsl::core::RefCountPtr<Rendezvous> rendezvous(NewLocalRendezvous());
    Rendezvous::Args args{
        .cancellation_manager = &cm,
    };
    rendezvous->RecvAsync(
        key, args,
        [&](const absl::Status&, const Rendezvous::Args&,
            const Rendezvous::Args&, const Tensor&, bool) { recv_count++; });
    std::thread abort_thread(
        [&] { cm.StartCancelWithStatus(absl::CancelledError("Cancelled")); });
    rendezvous->Send(key, args, Tensor(), false).IgnoreError();
    rendezvous->StartAbort(absl::CancelledError("Cancelled"));
    rendezvous.reset();
    abort_thread.join();
    ASSERT_EQ(1, recv_count);
  }
}

TEST(LocalRendezvous, CancelRecvRace) {
  Rendezvous::ParsedKey key;
  TF_EXPECT_OK(Rendezvous::ParseKey(
      Rendezvous::CreateKey("/job:mnist/replica:1/task:2/cpu:0", 7890,
                            "/job:mnist/replica:1/task:2/cpu:1", "foo",
                            FrameAndIter(0, 0)),
      &key));
  for (size_t i = 0; i < 10000; ++i) {
    std::atomic<size_t> recv_count = 0;
    tsl::CancellationManager cm;
    tsl::core::RefCountPtr<Rendezvous> rendezvous(NewLocalRendezvous());
    Rendezvous::Args args{
        .cancellation_manager = &cm,
    };
    std::thread abort_thread(
        [&] { cm.StartCancelWithStatus(absl::CancelledError("Cancelled")); });
    rendezvous->RecvAsync(
        key, args,
        [&](const absl::Status&, const Rendezvous::Args&,
            const Rendezvous::Args&, const Tensor&, bool) { recv_count++; });
    abort_thread.join();
    ASSERT_EQ(1, recv_count);
  }
}

}  // namespace
}  // namespace tensorflow
