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

#ifndef XLA_STREAM_EXECUTOR_SYCL_ONEAPI_COMPUTE_CAPABILITY_H_
#define XLA_STREAM_EXECUTOR_SYCL_ONEAPI_COMPUTE_CAPABILITY_H_

#include <cassert>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.pb.h"

namespace stream_executor {

#define EMIT_COMPUTE_CAPABILITY_FOR(HW)                    \
  static OneAPIComputeCapability HW() {                    \
    return absl::make_from_tuple<OneAPIComputeCapability>( \
        BaseVersionTupleFor(#HW));                         \
  }

class OneAPIComputeCapability {
 public:
  OneAPIComputeCapability() = default;

  explicit OneAPIComputeCapability(uint32_t generation, uint32_t version)
      : generation_(generation), version_(version) {}

  explicit OneAPIComputeCapability(const OneAPIComputeCapabilityProto& proto)
      : OneAPIComputeCapability(FromProto(proto)) {}

  explicit OneAPIComputeCapability(const uint32_t ip_version)
      : OneAPIComputeCapability((ip_version >> 22) & 0x3ff,
                                (ip_version >> 14) & 0xff) {}

  explicit OneAPIComputeCapability(absl::string_view name)
      : OneAPIComputeCapability(GenericIPVersionFor(name)) {}

  uint32_t generation() const { return generation_; }
  uint32_t version() const { return version_; }

  std::string ToString() const;

  EMIT_COMPUTE_CAPABILITY_FOR(PVC);
  EMIT_COMPUTE_CAPABILITY_FOR(BMG);
  EMIT_COMPUTE_CAPABILITY_FOR(DG2);

  OneAPIComputeCapabilityProto ToProto() const;

  static OneAPIComputeCapability FromProto(
      const OneAPIComputeCapabilityProto& proto);

  bool IsPVC() const { return generation_ == 0xc && (version_ & 0xfe) == 0x3c; }

  bool IsBMG() const { return generation_ == 0x14 && version_ <= 0x2; }

  bool IsDG2() const {
    return generation_ == 0xc &&
           (version_ == 0x37 || (version_ & 0xfe) == 0x38);
  }

  bool operator==(const OneAPIComputeCapability& other) const {
    return generation_ == other.generation_ && version_ == other.version_;
  }

  bool operator!=(const OneAPIComputeCapability& other) const {
    return !this->operator==(other);
  }

 private:
  uint32_t generation_ = 0;
  uint32_t version_ = 0;

  // A utility function that returns the base (generation, version) tuple for
  // the given platform name
  static std::pair<uint32_t, uint32_t> BaseVersionTupleFor(
      absl::string_view name);

  // Return the generic IP version for the given platform name
  static uint32_t GenericIPVersionFor(absl::string_view name);
};

#undef EMIT_COMPUTE_CAPABILITY_FOR

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SYCL_ONEAPI_COMPUTE_CAPABILITY_H_
