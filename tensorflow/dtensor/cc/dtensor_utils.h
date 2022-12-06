/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_CC_DTENSOR_UTILS_H_
#define TENSORFLOW_DTENSOR_CC_DTENSOR_UTILS_H_

namespace tensorflow {
namespace dtensor {

// Returns the DTensor client ID of this process, usually equal to the TF task
// ID on this host.
int ClientId();

// Returns the total number of DTensor clients, usually equal to the total
// number of TF tasks.
int NumClients();

// Returns whether to enable logging for passes and layouts on all passes.
bool LogOnAllTasks();

// Returns whether to log op-by-op execution in addition to function execution
// when logging is enabled.
bool LogOpByOp();

// Returns the maximum number of steps to run layout propagation. If the number
// of steps exceeds this amount, layout propagation will fail.
int LayoutPropagationMaxSteps();

// Returns whether to upcast bfloat16 reduction inputs to float32 for
// sufficient reduction group size.
bool EnableMixedPrecisionReduce();

// Returns whether *not* to fuse AllReduce + AllScatter into ReduceScatter op,
// which can be more efficiently implemented.
bool DoNotFuseReduceScatter();

// Returns the maximum reduction group size for bfloat16 reduction. If the
// group size exceeds this, then tensors are upcasted to float32 before the
// reduce op.
int ReduceInBfloat16MaxGroupSize();

// Returns whether to lower DTensorAllGather to CollectiveReduceV2. If false,
// lowers it to CollectiveReduceV2 for GPU and CPU for supported data types.
bool LowerCollectiveGatherToCollectiveGatherV2();

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_CC_DTENSOR_UTILS_H_
