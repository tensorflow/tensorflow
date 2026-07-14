/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_MAP_WRAPPER_H_
#define TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_MAP_WRAPPER_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/core/util/autotune_maps/autotune_map.pb.h"

namespace tensorflow {

// This class is a thin wrapper around `ConvMapProto::Entry`. It is used to
// provide opaque accessors to an entry's key and value without exposing the
// internal structure of the entry.
class ConvMapWrapper {
 public:
  using OpaqueKey = std::string;
  using OpaqueValue = std::string;

  // Creates an `ConvMapWrapper` from a key and value. The provided key and
  // value must be ones that were previously returned by calls to `Key()` and
  // `Value()`.
  static absl::StatusOr<ConvMapWrapper> FromKeyAndValue(OpaqueKey key,
                                                        OpaqueValue value);

  // An opaque string that can be used as a key for this autotuning result.
  // Do not rely on the format of this string.
  OpaqueKey Key() const;

  // An opaque string that encodes the autotuning result.
  // Do not rely on the format of this string.
  OpaqueValue Value() const;

  static std::vector<ConvMapWrapper> ConvMapToWrappers(
      const ConvMapProto& autotune_results);

  // Returns the `ConvMapProto` proto that corresponds to the provided
  // wrappers.
  static absl::StatusOr<ConvMapProto> ConvMapFromWrappers(
      const std::vector<ConvMapWrapper>& wrappers);

 private:
  explicit ConvMapWrapper(const ConvMapProto::Entry& entry)
      : conv_map_entry_(entry) {}

  ConvMapProto::Entry conv_map_entry_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_AUTOTUNE_MAPS_CONV_MAP_WRAPPER_H_
