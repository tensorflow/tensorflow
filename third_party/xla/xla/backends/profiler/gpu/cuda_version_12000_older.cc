/* Copyright 2025 The OpenXLA Authors.

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

#include "absl/types/span.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_callbacks.h"
#include "xla/backends/profiler/gpu/cuda_version_variants.h"

namespace xla {
namespace profiler {
namespace cuda_versions {

const CbidCategoryMap& GetExtraCallbackIdCategories12000() {
  return EmptyCallbackIdCategories();
}

absl::Span<const CUpti_CallbackIdResource> GetCudaGraphTracingResourceCbids() {
  return {};
}

}  // namespace cuda_versions

}  // namespace profiler
}  // namespace xla
