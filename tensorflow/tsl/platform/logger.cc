/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/platform/logger.h"

#include "absl/base/call_once.h"
#include "absl/synchronization/notification.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/logging.h"

namespace tsl {
namespace {

class DefaultLogger : public Logger {
 private:
  void DoLogProto(google::protobuf::Any* proto) override {}
  void DoFlush() override {}
};

}  // namespace

Logger::FactoryFunc Logger::singleton_factory_ = []() -> Logger* {
  return new DefaultLogger();
};

struct LoggerSingletonContainer {
  // Used to kick off the construction of a new thread that will asynchronously
  // construct a Logger.
  absl::once_flag start_initialization_thread_flag;

  // The constructed logger, if there is one.
  Logger* logger;

  // The initializing thread notifies `logger_initialized` after storing the
  // constructed logger to `logger`.
  absl::Notification logger_initialized;

  // The thread used to construct the Logger instance asynchronously.
  std::unique_ptr<Thread> initialization_thread;

  // Used to kick off the joining and destruction of `initialization_thread`.
  absl::once_flag delete_initialization_thread_flag;
};

LoggerSingletonContainer* GetLoggerSingletonContainer() {
  static LoggerSingletonContainer* container = new LoggerSingletonContainer;
  return container;
}

struct AsyncSingletonImpl {
  static void InitializationThreadFn() {
    LoggerSingletonContainer* container = GetLoggerSingletonContainer();
    container->logger = Logger::singleton_factory_();
    container->logger_initialized.Notify();
  }

  static void StartInitializationThread(LoggerSingletonContainer* container) {
    Thread* thread =
        Env::Default()->StartThread(ThreadOptions{}, "logger-init-thread",
                                    AsyncSingletonImpl::InitializationThreadFn);
    container->initialization_thread.reset(thread);
  }
};

/*static*/ Logger* Logger::GetSingleton() {
  // Call the async version to kick off the initialization thread if necessary.
  (void)Logger::GetSingletonAsync();

  // And wait for the thread to finish.
  LoggerSingletonContainer* container = GetLoggerSingletonContainer();
  absl::call_once(container->delete_initialization_thread_flag,
                  [container]() { container->initialization_thread.reset(); });

  return container->logger;
}

/*static*/ Logger* Logger::GetSingletonAsync() {
  LoggerSingletonContainer* container = GetLoggerSingletonContainer();
  absl::call_once(container->start_initialization_thread_flag,
                  AsyncSingletonImpl::StartInitializationThread, container);

  if (container->logger_initialized.HasBeenNotified()) {
    // Wait for the initializing thread to finish to reclaim resources.
    absl::call_once(
        container->delete_initialization_thread_flag,
        [container]() { container->initialization_thread.reset(); });
    return container->logger;
  } else {
    return nullptr;
  }
}
}  // namespace tsl
