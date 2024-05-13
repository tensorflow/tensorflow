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

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/mlir/tfrt/backend_compiler.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tsl/platform/thread_annotations.h"

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
  // Should only set one of `use_tpu` and `use_gpu` and `backend_compiler`.
  bool use_tpu = false;
  bool use_gpu = false;
  tensorflow::BackendCompiler* backend_compiler = nullptr;
};

// Factory class to create `TfrtSession` instances.
class TfrtSessionFactory : public tensorflow::SessionFactory {
 public:
  TfrtSessionFactory();

  bool AcceptsOptions(const SessionOptions& options) override;

  Status NewSession(const SessionOptions& options,
                    Session** out_session) override TF_LOCKS_EXCLUDED(mutex_);

  // This should only be used for the sake initializing resources for
  // Python executables. It should only be called before main.
  //
  // Due to lack of applications and a concern for the ordering of initializers,
  // this may only be called once.
  using RuntimeInitializer = absl::Status (*)(tfrt_stub::Runtime*);
  static void RegisterInitializer(RuntimeInitializer initializer);

  // May not be called within code holding mutex_.
  static tfrt_stub::Runtime* GetRuntime();

 private:
  class ThreadPoolManager;
  friend Status InitializeTfrtSession(const TfrtSessionOptions& options);
  friend Status UpdateTfrtSessionOptionsLocked(
      const TfrtSessionOptions& options);
  Status InitializeLocked(const TfrtSessionOptions& options)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  bool IsInitialized() const TF_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return runtime_ != nullptr;
  }

  mutable absl::Mutex mutex_;
  mutable absl::Mutex runtime_mutex_;
  tensorflow::tfrt_stub::Runtime* runtime_ TF_GUARDED_BY(mutex_) = nullptr;
  std::unique_ptr<tensorflow::tfrt_stub::Runtime> owned_runtime_
      TF_GUARDED_BY(mutex_);

  TfrtDeviceInfraTarget device_target_ TF_GUARDED_BY(mutex_) =
      TfrtDeviceInfraTarget::kCpu;
  bool tpu_use_tpu_runner_ TF_GUARDED_BY(mutex_) = false;
  bool use_gpu_ TF_GUARDED_BY(mutex_) = false;
  std::unique_ptr<ThreadPoolManager> thread_pool_manager_ TF_GUARDED_BY(mutex_);
  bool enable_mlrt_ TF_GUARDED_BY(mutex_) = false;
  tensorflow::BackendCompiler* backend_compiler_ TF_GUARDED_BY(mutex_);
};

// Configures the TfrtSessionFactory according to `options`. Should not be
// called within functions that are passed into
// `TfrtSessionFactory::RegisterInitializer`, because it acquires `mutex_`.
Status InitializeTfrtSession(const TfrtSessionOptions& options);

// Version of `InitializeTfrtSession` that can be used within functions passed
// into `TfrtSessionFactory::RegisterInitializer`.
Status UpdateTfrtSessionOptionsLocked(const TfrtSessionOptions& options);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_TFRT_SESSION_TFRT_SESSION_H_
