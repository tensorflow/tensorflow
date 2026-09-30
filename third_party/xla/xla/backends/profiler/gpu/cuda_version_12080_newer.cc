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

#include "absl/base/no_destructor.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_driver_cbid.h"
#include "xla/backends/profiler/gpu/cuda_version_variants.h"

namespace xla {
namespace profiler {
namespace cuda_versions {

// Previous impacted version is 12.0, CBid supported here are [701, 782)
const CbidCategoryMap& GetExtraCallbackIdCategories12080() {
  if (GetSafeCudaVersion() < 12080) {
    return EmptyCallbackIdCategories();
  }
  static const absl::NoDestructor<CbidCategoryMap> kCbidCategoryMap({
      {CUPTI_DRIVER_TRACE_CBID_cuGraphAddNode /* 712 */,
       CbidCategory::kGraphNode},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphAddNode_v2 /* 723 */,
       CbidCategory::kGraphNode},
  });
  return *kCbidCategoryMap;
}

absl::string_view GetExtraActivityOverheadKindString12080(
    CUpti_ActivityOverheadKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_RUNTIME_TRIGGERED_MODULE_LOADING:
      return "RUNTIME_TRIGGERED_MODULE_LOADING";
    case CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING:
      return "LAZY_FUNCTION_LOADING";
    case CUPTI_ACTIVITY_OVERHEAD_COMMAND_BUFFER_FULL:
      return "COMMAND_BUFFER_FULL";
    case CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST:
      return "ACTIVITY_BUFFER_REQUEST";
    case CUPTI_ACTIVITY_OVERHEAD_UVM_ACTIVITY_INIT:
      return "UVM_ACTIVITY_INIT";
    default:
      return "";
  }
}

}  // namespace cuda_versions

}  // namespace profiler
}  // namespace xla
