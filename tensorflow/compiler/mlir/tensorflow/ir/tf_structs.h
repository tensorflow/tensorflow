/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the types used in the standard MLIR TensorFlow dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_STRUCTS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_STRUCTS_H_

#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Diagnostics.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h.inc"

namespace TF {

// Tensorflow devices available at runtime with corresponding metadata if it is
// available. It's completely valid to have a device without any metadata
// attached to it.
class RuntimeDevices {
  using DeviceNameUtils = ::tensorflow::DeviceNameUtils;
  using ParsedName = ::tensorflow::DeviceNameUtils::ParsedName;

 public:
  // Adds a device with and empty metadata. Device can be of any type.
  void AddDevice(const ParsedName& device);

  // Adds a GPU device with GPU specific metadata.
  void AddGpuDevice(const ParsedName& device,
                    const GpuDeviceMetadata& metadata);

  llvm::ArrayRef<ParsedName> device_names() const { return device_names_; }
  size_t NumDevices() const { return device_names_.size(); }

  // Returns GPU device metadata if it is available, otherwise returns None.
  llvm::Optional<GpuDeviceMetadata> GetGpuDeviceMetadata(
      const ParsedName& device) const;

 private:
  llvm::SmallVector<ParsedName, 8> device_names_;
  // TODO(ezhulenev): Add DenseMapInfo<ParsedName> specialization to be able to
  // use ParsedName as a key in a DenseMap.
  llvm::StringMap<GpuDeviceMetadata> gpu_metadata_;
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_STRUCTS_H_
