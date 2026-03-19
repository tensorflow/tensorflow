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
#include "absl/types/span.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_callbacks.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_driver_cbid.h"
#include "xla/backends/profiler/gpu/cuda_version_variants.h"

namespace xla {
namespace profiler {
namespace cuda_versions {

// Previous impacted version is 11.0, Cbids supported here are [579, 701)
const CbidCategoryMap& GetExtraCallbackIdCategories12000() {
  if (GetSafeCudaVersion() < 12000) {
    return EmptyCallbackIdCategories();
  }
  static const absl::NoDestructor<CbidCategoryMap> kCbidCategoryMap({
      {CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx /* 652 */,
       CbidCategory::kKernel},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithFlags /* 643 */,
       CbidCategory::kGraph},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams /* 656 */,
       CbidCategory::kGraph},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphInstantiateWithParams_ptsz /* 657 */,
       CbidCategory::kGraph},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphAddEventRecordNode /* 589 */,
       CbidCategory::kGraphNode},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphAddEventWaitNode /* 590 */,
       CbidCategory::kGraphNode},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemAllocNode /* 638 */,
       CbidCategory::kGraphNode},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphAddMemFreeNode /* 639 */,
       CbidCategory::kGraphNode},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphAddBatchMemOpNode /* 669 */,
       CbidCategory::kGraphNode},
      {CUPTI_DRIVER_TRACE_CBID_cuGraphAddKernelNode_v2 /* 689 */,
       CbidCategory::kGraphNode},
  });
  return *kCbidCategoryMap;
}

absl::Span<const CUpti_CallbackIdResource> GetCudaGraphTracingResourceCbids() {
  static constexpr CUpti_CallbackIdResource res_cbids[] = {
      CUPTI_CBID_RESOURCE_GRAPH_CREATED /* 9 */,
      CUPTI_CBID_RESOURCE_GRAPH_CLONED /* 11 */,
      CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED /* 13 */,
      CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED /* 18 */,
      CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED /* 20 */};
  if (GetSafeCudaVersion() >= 12000) {
    return absl::MakeSpan(res_cbids);
  }
  return absl::Span<const CUpti_CallbackIdResource>();
}

}  // namespace cuda_versions

}  // namespace profiler
}  // namespace xla
