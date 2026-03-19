/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/llvm_gpu_backend/utils.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace gpu {

std::string ReplaceFilenameExtension(absl::string_view filename,
                                     absl::string_view new_extension) {
  auto pos = filename.rfind('.');
  absl::string_view stem = pos == absl::string_view::npos
                               ? filename
                               : absl::string_view(filename.data(), pos);
  return absl::StrCat(stem, ".", new_extension);
}

}  // namespace gpu
}  // namespace xla
