/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TPU_REWRITE_DEVICE_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TPU_REWRITE_DEVICE_UTIL_H_

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
// Finds the TPU compilation device and execution devices from `devices` for a
// replicated TPU computation subgraph. Compilation device is determined from
// looking up all TPU_SYSTEM:0 devices and choosing the CPU device associated
// to the first TPU_SYSTEM device sorted lexicographically by replica and task.
// Execution devices are determined by looking up all TPU devices associated
// with each TPU_SYSTEM:0 device found. A failure will be returned if it is not
// possible (e.g. invalid devices).
//
// For example, with `num_replicas` = 4 and `devices`:
//   /job:localhost/replica:0/task:0/device:CPU:0
//   /job:worker/replica:0/task:0/device:CPU:0
//   /job:worker/replica:0/task:0/device:TPU_SYSTEM:0
//   /job:worker/replica:0/task:0/device:TPU:0
//   /job:worker/replica:0/task:0/device:TPU:1
//   /job:worker/replica:0/task:1/device:CPU:0
//   /job:worker/replica:0/task:1/device:TPU_SYSTEM:0
//   /job:worker/replica:0/task:1/device:TPU:0
//   /job:worker/replica:0/task:1/device:TPU:1
//
// The compilation device will be:
//   /job:worker/replica:0/task:0/device:CPU:0
//
// and the execution devices (sorted) will be:
//   /job:worker/replica:0/task:0/device:TPU:0
//   /job:worker/replica:0/task:0/device:TPU:1
//   /job:worker/replica:0/task:1/device:TPU:0
//   /job:worker/replica:0/task:1/device:TPU:1
Status GetTPUCompilationAndExecutionDevices(
    llvm::ArrayRef<DeviceNameUtils::ParsedName> devices, int num_replicas,
    int num_cores_per_replica, std::string* compilation_device,
    llvm::SmallVectorImpl<std::string>* execution_devices);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_TPU_REWRITE_DEVICE_UTIL_H_
