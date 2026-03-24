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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUDA_VERSION_VARIANTS_H_
#define XLA_BACKENDS_PROFILER_GPU_CUDA_VERSION_VARIANTS_H_

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_callbacks.h"

namespace xla {
namespace profiler {
namespace cuda_versions {

// Get minimum version of CUDA runtime and CUDA driver for simplicity.
int GetSafeCudaVersion();

// Currently Driver Callback ID (CBID) related versions for the compilation
// CUDA toolkit are (11.0), 12.0, 12.8.
enum CbidCategory {
  kNone,
  kKernel,
  kGraph,
  kGraphNode,
};

using CbidCategoryMap =
    absl::flat_hash_map<CUpti_driver_api_trace_cbid, CbidCategory>;

inline CbidCategory FindCbidCategory(const CbidCategoryMap& cbid_categories,
                                     CUpti_driver_api_trace_cbid cbid) {
  auto it = cbid_categories.find(cbid);
  if (it == cbid_categories.end()) {
    return CbidCategory::kNone;
  }
  return it->second;
}

const CbidCategoryMap& EmptyCallbackIdCategories();

const CbidCategoryMap& GetExtraCallbackIdCategories12000();

const CbidCategoryMap& GetExtraCallbackIdCategories12080();

// Resource CBIDs impacted only before/after 12.0.
absl::Span<const CUpti_CallbackIdResource> GetCudaGraphTracingResourceCbids();

}  // namespace cuda_versions
}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUDA_VERSION_VARIANTS_H_
