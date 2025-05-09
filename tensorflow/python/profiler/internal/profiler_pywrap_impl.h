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
#ifndef TENSORFLOW_PYTHON_PROFILER_INTERNAL_PROFILER_PYWRAP_IMPL_H_
#define TENSORFLOW_PYTHON_PROFILER_INTERNAL_PROFILER_PYWRAP_IMPL_H_

#include <memory>
#include <string>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/variant.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"

namespace tensorflow {
namespace profiler {
namespace pywrap {

class ProfilerSessionWrapper {
 public:
  absl::Status Start(
      const char* logdir,
      const absl::flat_hash_map<std::string,
                                std::variant<bool, int, std::string>>& options);
  absl::Status Stop(tensorflow::string* result);
  absl::Status ExportToTensorBoard();

 private:
  std::unique_ptr<tsl::ProfilerSession> session_;
  tensorflow::string logdir_;
};

}  // namespace pywrap
}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_PROFILER_INTERNAL_PROFILER_PYWRAP_IMPL_H_
