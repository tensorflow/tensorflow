/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/mlir_pass_instrumentation.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/platform/logging.h"

namespace mlir {

class MlirPassInstrumentationRegistry {
 public:
  static MlirPassInstrumentationRegistry& Instance() {
    static MlirPassInstrumentationRegistry* r =
        new MlirPassInstrumentationRegistry;
    return *r;
  }
  std::unordered_map<std::string,
                     std::function<std::unique_ptr<PassInstrumentation>()>>
      instrumentors_;
};

void RegisterPassInstrumentor(
    const std::string& name,
    std::function<std::unique_ptr<PassInstrumentation>()> creator) {
  MlirPassInstrumentationRegistry& r =
      MlirPassInstrumentationRegistry::Instance();
  auto result = r.instrumentors_.emplace(name, creator);
  if (!result.second) {
    VLOG(1) << "Duplicate MLIR pass instrumentor registration";
  }
}

std::vector<std::function<std::unique_ptr<PassInstrumentation>()>>
GetPassInstrumentors() {
  MlirPassInstrumentationRegistry& r =
      MlirPassInstrumentationRegistry::Instance();
  std::vector<std::function<std::unique_ptr<PassInstrumentation>()>> result;
  result.reserve(r.instrumentors_.size());

  std::transform(r.instrumentors_.begin(), r.instrumentors_.end(),
                 std::back_inserter(result), [](auto v) { return v.second; });

  return result;
}

}  // namespace mlir
