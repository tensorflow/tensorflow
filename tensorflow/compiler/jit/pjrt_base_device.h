/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_PJRT_BASE_DEVICE_H_
#define TENSORFLOW_COMPILER_JIT_PJRT_BASE_DEVICE_H_

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

// tensorflow::PjRtBaseDevice replaces the deprecated tensorflow::XlaDevice.
// This accelerator agnostic device is mainly used to store metadata.
class PjRtBaseDevice : public LocalDevice {
 public:
  // Stores metadata about the PjRtBaseDevice.
  class Metadata {
   public:
    Metadata(const DeviceType& jit_device_type,
             std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
                 shape_determination_fns)
        : jit_device_type_(jit_device_type),
          shape_determination_fns_(std::move(shape_determination_fns)) {}

    // The index of the device on this host.
    int device_ordinal() const;

    const DeviceType& jit_device_type() const { return jit_device_type_; }
    const XlaShapeLayoutHelpers::ShapeDeterminationFns&
    default_shape_determination_fns() const {
      return shape_determination_fns_.at(0);
    }

    const XlaShapeLayoutHelpers::ShapeDeterminationFns&
    shape_determination_fns_at(int i) const {
      return shape_determination_fns_[i];
    }

   private:
    const DeviceType jit_device_type_;
    std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
        shape_determination_fns_;

    Metadata(const Metadata&) = delete;
    void operator=(const Metadata&) = delete;
  };

  struct Options {
    // The device name's prefix (e.g., "/task:7")
    std::string device_name_prefix;

    // The name of the  device (e.g., "TPU")
    std::string device_name;

    // The index of the device.
    int device_ordinal = -1;

    // The name of the compilation device, also referred to as jit_device_type.
    // (e.g., "XLA_CPU_JIT");
    std::string compilation_device_name;

    // A vector of ShapeDeterminationFn (i.e., a bundle of LayoutSelectionFn,
    // ShapeRepresentationFn). Each bundle describes how the on-host shapes of
    // a) argument and return value, for entry computations b) variables, for
    // all computations, should be represented in XLA. Parameters/return values
    // will be shaped according to the function pair, and reshaped back to/from
    // their declared shapes for computations. Must be non-empty.
    std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
        shape_determination_fns;

    Options(std::string device_name_prefix, std::string device_name,
            int device_ordinal, std::string compilation_device_name,
            std::vector<XlaShapeLayoutHelpers::ShapeDeterminationFns>
                shape_determination_fns)
        : device_name_prefix(device_name_prefix),
          device_name(device_name),
          device_ordinal(device_ordinal),
          compilation_device_name(compilation_device_name),
          shape_determination_fns(shape_determination_fns) {}
  };

  // Creates a new PJRT base device.
  PjRtBaseDevice(const SessionOptions& session_options, const Options& options);

  static absl::StatusOr<const PjRtBaseDevice::Metadata*> GetMetadataFromDevice(
      DeviceBase* device);

 private:
  // The metadata of this PjRtBaseDevice.
  const Metadata metadata_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_PJRT_BASE_DEVICE_H_
