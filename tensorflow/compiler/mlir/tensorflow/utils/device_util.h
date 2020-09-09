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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DEVICE_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DEVICE_UTIL_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

// Collects all devices known to the system by name and adds them as a
// `tf.devices` dictionary attribute with a full device name as a key, and
// device metadata as a value.
//
// Device names added in full parsed device form:
//   /job:<name>/replica:<replica>/task:<task>/device:<type>:<device_num>
//
// Supported device metadata types:
// (1) GpuDeviceMetadata: GPU device compute capability.
void AddDevicesToOp(mlir::Operation* op, const DeviceSet* device_set);

// Collects devices information from an op `tf.devices` attributes. Returns
// failure if can't parse device metadata from the attribute.
mlir::LogicalResult GetDevicesFromOp(mlir::Operation* op,
                                     mlir::TF::RuntimeDevices* devices);

// Parses a device string and returns its ordinal (id). This will return an
// error if the device string is invalid or has no id.
mlir::LogicalResult GetDeviceOrdinalFromDeviceString(mlir::Location loc,
                                                     llvm::StringRef device,
                                                     int64_t* device_ordinal);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_DEVICE_UTIL_H_
