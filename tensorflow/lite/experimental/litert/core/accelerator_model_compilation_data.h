// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ACCELERATOR_MODEL_COMPILATION_DATA_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ACCELERATOR_MODEL_COMPILATION_DATA_H_

#include <memory>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/accelerator.h"

namespace litert::internal {

// Holds environment data that accelerators may need to prepare their
// delegates.
//
// These options are automatically added to the compilation options list
// during the creation of the compiled model.
struct ModelCompilationData : public LiteRtAcceleratorCompilationOptionsHeader {
  static constexpr absl::string_view kIdentifier =
      "environment-compilation-options";
  static constexpr LiteRtApiVersion kVersion = {1, 0, 0};

  struct Deleter {
    void operator()(ModelCompilationData* options) { delete options; }
  };

  using Ptr = std::unique_ptr<ModelCompilationData, Deleter>;

  static Expected<Ptr> Create() {
    Ptr data(new ModelCompilationData());
    LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorCompilationOptionsIdentifier(
        data.get(), kIdentifier.data()));
    LITERT_RETURN_IF_ERROR(
        LiteRtSetAcceleratorCompilationOptionsVersion(data.get(), kVersion));
    LITERT_RETURN_IF_ERROR(LiteRtSetAcceleratorCompilationOptionsDestructor(
        data.get(), [](LiteRtAcceleratorCompilationOptionsHeader* options) {
          Deleter()(reinterpret_cast<ModelCompilationData*>(options));
        }));
    return data;
  }

  // Pointer to the start of the model file memory allocation.
  const char* allocation_base;

 private:
  ModelCompilationData() = default;
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_ACCELERATOR_MODEL_COMPILATION_DATA_H_
