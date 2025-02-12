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

#include "xla/tsl/platform/threadpool_async_executor.h"

#include "absl/synchronization/notification.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace tsl::thread {
namespace {

TEST(ThreadPoolAsyncExecutorTest, ExecuteTasks) {
  ThreadPool thread_pool(Env::Default(), "test", 4);
  ThreadPoolAsyncExecutor executor(&thread_pool);

  absl::Notification notification;
  executor.Execute([&] { notification.Notify(); });
  notification.WaitForNotification();
}

}  // namespace
}  // namespace tsl::thread
