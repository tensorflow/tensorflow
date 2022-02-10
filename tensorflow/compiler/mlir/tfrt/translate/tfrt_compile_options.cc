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

#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"

#include <ostream>

namespace tensorflow {

std::ostream& operator<<(std::ostream& os, TfrtTpuInfraTarget tpu_target) {
  switch (tpu_target) {
    case TfrtTpuInfraTarget::kNoTpu:
      return os << "NoTpu";
    case TfrtTpuInfraTarget::kTpurt:
      return os << "Tpurt";
    case TfrtTpuInfraTarget::kTfFallback:
      return os << "TfFallback";
    case TfrtTpuInfraTarget::kBridgeFallback:
      return os << "BridgeFallback";
  }
}

}  // namespace tensorflow
