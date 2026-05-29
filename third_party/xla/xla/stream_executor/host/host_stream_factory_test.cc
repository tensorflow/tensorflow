/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/host/host_stream_factory.h"

#include <atomic>
#include <memory>

#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/host/host_stream.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"

namespace stream_executor {
namespace host {
namespace {

class DummyFactory : public HostStreamFactory {
 public:
  std::unique_ptr<HostStream> CreateStream(
      StreamExecutor* executor) const override {
    return nullptr;
  }
};

TEST(HostStreamFactoryTest, GetFactoryUAF) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          PlatformManager::PlatformWithName("Host"));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(0));

  std::atomic<bool> stop{false};
  absl::Notification reader_started;

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 2);

  pool.Schedule([&]() {
    reader_started.Notify();
    while (!stop) {
      std::shared_ptr<const HostStreamFactory> factory =
          HostStreamFactory::GetFactory();
      if (factory) {
        // Accessing virtual method. If the object was freed due to the lock
        // being released prematurely, ASAN will panic triggering a UAF here.
        factory->CreateStream(executor);
      }
    }
  });

  reader_started.WaitForNotification();

  pool.Schedule([&]() {
    // Repeatedly register new factories with increasing priority to force
    // the overwriting and deleting of the previous instance.
    for (int i = 200; i <= 2000; ++i) {
      HostStreamFactory::Register(std::make_unique<DummyFactory>(), i);
    }
    stop = true;
  });
}

}  // namespace
}  // namespace host
}  // namespace stream_executor
