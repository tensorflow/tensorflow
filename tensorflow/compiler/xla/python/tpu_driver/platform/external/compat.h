// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_PLATFORM_EXTERNAL_COMPAT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_PLATFORM_EXTERNAL_COMPAT_H_

#include <thread>  // NOLINT

#include "absl/strings/string_view.h"

namespace tpu_driver {

class Thread {
 public:
  template <class Function, class... Args>
  explicit Thread(Function&& f, Args&&... args)
      : thread_(std::forward<Function>(f), std::forward<Args>(args)...) {}
  void join() { thread_.join(); }

 private:
  std::thread thread_;
};

class TraceMe {
 public:
  explicit TraceMe(absl::string_view name, int level = 1) {}
  explicit TraceMe(std::string&& name, int level = 1) = delete;
  explicit TraceMe(const std::string& name, int level = 1) = delete;
  explicit TraceMe(const char* raw, int level = 1)
      : TraceMe(absl::string_view(raw), level) {}
  template <typename NameGeneratorT>
  explicit TraceMe(NameGeneratorT name_generator, int level = 1) {}
  ~TraceMe() {}
};

}  // namespace tpu_driver

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_PLATFORM_EXTERNAL_COMPAT_H_
