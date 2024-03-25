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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COST_CONSTANTS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COST_CONSTANTS_H_

namespace tensorflow {

// Types of per-request cost.
inline constexpr char kTpuCostName[] = "tpu";
inline constexpr char kGcuCostName[] = "gcu";
inline constexpr char kNoOpCostName[] = "no_op";

// Each type of per-request cost could have the following versions.
//
// A server may have costs that cannot be directly attributed to a specific
// query. Each request will be assigned a portion of it, and the cost ends with
// '_with_smear" includes this part.
inline constexpr char kWithSmearSuffix[] = "_with_smear";
inline constexpr char kNoSmearSuffix[] = "_no_smear";
inline constexpr char kNonBatchingSuffix[] = "_non_batching";

// Full names of per-request cost.
inline constexpr char kTpuWithSmearCostName[] = "tpu_with_smear";
inline constexpr char kTpuNoSmearCostName[] = "tpu_no_smear";
inline constexpr char kTpuNonBatchingCostName[] = "tpu_non_batching";
inline constexpr char kGcuWithSmearCostName[] = "gcu_with_smear";
inline constexpr char kGcuNoSmearCostName[] = "gcu_no_smear";
inline constexpr char kGcuNonBatchingCostName[] = "gcu_non_batching";

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COST_CONSTANTS_H_
