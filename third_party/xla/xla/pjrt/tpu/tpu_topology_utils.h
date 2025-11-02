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

#ifndef XLA_PJRT_TPU_TPU_TOPOLOGY_UTILS_H_
#define XLA_PJRT_TPU_TPU_TOPOLOGY_UTILS_H_

#ifdef GOOGLE_INTERNAL

#include "learning/brain/research/pjrt/tpu/tpu_topology_utils_internal.h"

namespace xla {

using internal::CreateTpuTopologyDescription;
using internal::TpuTopologyArgsFromPjRtTopologyDescription;
using internal::TpuVersionToString;

}  // namespace xla

#else

#include "xla/pjrt/tpu/tpu_topology_utils_external.h"

namespace xla {

using external::CreateTpuTopologyDescription;
using external::TpuTopologyArgsFromPjRtTopologyDescription;
using external::TpuVersionToString;

}  // namespace xla

#endif

#endif  // XLA_PJRT_TPU_TPU_TOPOLOGY_UTILS_H_
