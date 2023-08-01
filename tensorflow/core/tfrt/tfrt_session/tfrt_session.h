/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_TFRT_SESSION_TFRT_SESSION_H_
#define TENSORFLOW_CORE_TFRT_TFRT_SESSION_TFRT_SESSION_H_

#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"

namespace tensorflow {

// Struct exposing a few threadpool configuration options. These
// correspond to the options in RunHandlerThreadWorkQueue::Options.
struct TfrtThreadpoolOptions {
  // Number of threads used for running graphs.
  int32_t num_main_threads = port::MaxParallelism();

  // Time to wait for the init function to complete.
  absl::Duration init_timeout = absl::Milliseconds(100);

  // Maximum number of concurrent RunHandlers.
  int32_t max_concurrent_handler = 128;

  // Number of sub thread pools.
  int32_t num_sub_thread_pool = 1;
};

struct TfrtSessionOptions {
  TfrtThreadpoolOptions threadpool_options;
  tensorflow::tfrt_stub::Runtime* runtime = nullptr;
  bool enable_mlrt = false;
};

// Factory class to create `TfrtSession` instances.
class TfrtSessionFactory : public tensorflow::SessionFactory {
 public:
  TfrtSessionFactory();

  Status Initialize(const TfrtSessionOptions& options)
      TF_LOCKS_EXCLUDED(mutex_);

  bool AcceptsOptions(const SessionOptions& options) override;

  Status NewSession(const SessionOptions& options,
                    Session** out_session) override TF_LOCKS_EXCLUDED(mutex_);

  static TfrtSessionFactory& Get();

  // Initializers will run at the end of the initialization, assuming lock held.
  static void RegisterInitializer(std::function<absl::Status()> initializer);

 private:
  class ThreadPoolManager;
  friend Status InitTpuForTfrtSessionLocked(bool only_set_fields)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  friend Status InitTpuForTfrtSession(bool only_set_fields)
      TF_LOCKS_EXCLUDED(mutex_);

  bool IsInitialized() const TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return runtime_ != nullptr;
  }
  Status InitializeLocked(const TfrtSessionOptions& options)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_;
  tensorflow::tfrt_stub::Runtime* runtime_ TF_GUARDED_BY(mutex_) = nullptr;
  std::unique_ptr<tensorflow::tfrt_stub::Runtime> owned_runtime_
      TF_GUARDED_BY(mutex_);
  TfrtDeviceInfraTarget device_target_ TF_GUARDED_BY(mutex_) =
      TfrtDeviceInfraTarget::kCpu;
  bool tpu_use_tpu_runner_ TF_GUARDED_BY(mutex_) = false;
  std::unique_ptr<ThreadPoolManager> thread_pool_manager_ TF_GUARDED_BY(mutex_);
  bool enable_mlrt_ TF_GUARDED_BY(mutex_) = false;
};

// Initiailzes and registers the TfrtSessionFactory. Calling this function makes
// available a new Tensorflow session target, tfrt_session, which can be used to
// create a TFRT based tensorflow Session implementation.
Status InitializeTfrtSession(const TfrtSessionOptions& options);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_TFRT_SESSION_TFRT_SESSION_H_
