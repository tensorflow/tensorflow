/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_compile_interface.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/logging.h"

class TpuCompileInterfaceExternal : public TpuCompileInterface {
 public:
  uint64_t FingerprintString(absl::string_view str) override {
    return ::tensorflow::Fingerprint64(str);
  }
};

static TpuCompileInterface* impl_ = new TpuCompileInterfaceExternal;
TpuCompileInterface* TpuCompileInterface::Get() { return impl_; }

bool TpuCompileInterface::RegisterImplementation(TpuCompileInterface* impl) {
  VLOG(1) << "Updating TpuCompileInterface.";
  if (impl_ != nullptr) {
    delete impl_;
  }
  impl_ = impl;
  return true;
}
