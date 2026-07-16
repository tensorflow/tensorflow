/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/cpu/lite_aot/infer_lite_aot_deps_main_lib.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/service/cpu/executable.pb.h"

namespace xla {
namespace cpu {

absl::StatusOr<std::vector<std::string>> InferLiteAotDeps(
    const CompilationResultProto& compilation_result) {
  absl::flat_hash_set<std::string> deps;

  if (!compilation_result.has_thunk_sequence()) {
    // If there's no thunk sequence, there are no dynamic thunk SerDes deps.
    return std::vector<std::string>();
  }

  // LINT.IfChange
  for (const auto& thunk : compilation_result.thunk_sequence().thunks()) {
    switch (thunk.impl_case()) {
      case ThunkProto::kCollectiveThunk:
        deps.insert(SerDesPath("collective"));
        break;
      case ThunkProto::kConvolutionThunk:
        deps.insert(SerDesPath("convolution"));
        break;
      case ThunkProto::kCopyThunk:
        deps.insert(SerDesPath("copy"));
        break;
      case ThunkProto::kCustomCallThunk:
        deps.insert(SerDesPath("custom_call"));
        break;
      case ThunkProto::kDotThunk:
        deps.insert(SerDesPath("dot"));
        break;
      case ThunkProto::kFftThunk:
        deps.insert(SerDesPath("fft"));
        break;
      case ThunkProto::kYnnFusionThunk:
        deps.insert(SerDesPath("ynn_fusion"));
        break;
      default:
        break;
    }
  }
  // LINT.ThenChange(//third_party/tensorflow/compiler/xla/backends/cpu/runtime/thunk_serdes/BUILD)

  std::vector<std::string> result(deps.begin(), deps.end());
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace cpu
}  // namespace xla
