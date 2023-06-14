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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_TFRT_RUNTIME_H_
#define TENSORFLOW_CORE_TFRT_UTILS_TFRT_RUNTIME_H_

#include <memory>

#include "absl/synchronization/mutex.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"

namespace tensorflow {
namespace tfrt_stub {

// A class to hold global TFRT core runtime. The class is thread-safe.
// TODO(b/281750702) Unify this class with `tensorflow::tfrt_stub::Runtime`.
class TfrtRuntime {
 public:
  // Set Global TFRT Runtime.
  void SetRuntime(std::unique_ptr<Runtime> runtime);

  // Get Global TFRT Runtime.
  Runtime* GetRuntime();

  // Get a global instance of the TfrtRuntime.
  static TfrtRuntime& GetGlobalTfrtRuntime();

 private:
  absl::Mutex m_;
  // TFRT Runtime.
  std::unique_ptr<Runtime> runtime_ ABSL_GUARDED_BY(m_);
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_UTILS_TFRT_RUNTIME_H_
