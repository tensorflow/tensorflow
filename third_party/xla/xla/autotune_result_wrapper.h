/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_AUTOTUNE_RESULT_WRAPPER_H_
#define XLA_AUTOTUNE_RESULT_WRAPPER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/autotune_results.pb.h"

namespace xla {

// This class is a thin wrapper around AutotuneResults::Entry. It is used to
// provide opaque accessors to an entry's key and value without exposing the
// internal structure of the entry.
class AutotuneResultWrapper {
 public:
  using OpaqueKey = std::string;
  using OpaqueValue = std::string;

  // Creates an AutotuneResultWrapper from a key and value. The provided key and
  // value must be ones that were previously returned by calls to Key() and
  // Value().
  static absl::StatusOr<AutotuneResultWrapper> FromKeyAndValue(
      OpaqueKey key, OpaqueValue value);

  // An opaque string that can be used as a key for this Autotuning result.
  // Do not rely on the format of this string.
  OpaqueKey Key() const;

  // An opaque string that encodes the autotuning result.
  // Do not rely on the format of this string.
  OpaqueValue Value() const;

  static std::vector<AutotuneResultWrapper> AutotuneResultsToWrappers(
      const AutotuneResults& autotune_results);

  // Returns the AutotuneResults proto that corresponds to the provided
  // wrappers. This function will return an error if the provided wrappers have
  // inconsistent versions.
  static absl::StatusOr<AutotuneResults> AutotuneResultsFromWrappers(
      const std::vector<AutotuneResultWrapper>& wrappers);

 private:
  AutotuneResultWrapper(const AutotuneResults::Entry& result, int32_t version)
      : autotune_result_(result), version_(version) {}

  AutotuneResults::Entry autotune_result_;
  int32_t version_;
};

}  // namespace xla

#endif  // XLA_AUTOTUNE_RESULT_WRAPPER_H_
