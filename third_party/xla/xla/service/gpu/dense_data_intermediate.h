/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_DENSE_DATA_INTERMEDIATE_H_
#define XLA_SERVICE_GPU_DENSE_DATA_INTERMEDIATE_H_

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/gpu/dense_data_intermediate.pb.h"

namespace xla {
namespace gpu {

// This class stores either a non-owning reference or owns data that represents
// a dense array in XLA format. It is used for intermediate storage during IR
// constant emission.
class DenseDataIntermediate {
 public:
  // Creates an instance of DenseDataIntermediate that owns the provided vector.
  static DenseDataIntermediate Own(std::vector<uint8_t> owned) {
    DenseDataIntermediate di;
    di.data_ = std::move(owned);
    return di;
  }

  // Creates an instance of DenseDataIntermediate that aliases the input.
  static DenseDataIntermediate Alias(absl::Span<const uint8_t> aliased) {
    DenseDataIntermediate di;
    di.data_ = aliased;
    return di;
  }

  // Returns a reference to the data this object represents.
  absl::Span<const uint8_t> span() const {
    return data_.index() == 0 ? absl::Span<const uint8_t>(std::get<0>(data_))
                              : std::get<1>(data_);
  }

  // Converts `this` into its protobuf representation.
  // Note that the protobuf message will always contain a copy of the data -
  // also for non-owning instances of DenseDataIntermediate.
  DenseDataIntermediateProto ToProto() const;

  // Constructs a data-owning instance of DenseDataIntermediate from its
  // protobuf representation.
  static DenseDataIntermediate FromProto(
      const DenseDataIntermediateProto& proto);

 private:
  std::variant<std::vector<uint8_t>, absl::Span<const uint8_t>> data_;
};

absl::StatusOr<DenseDataIntermediate> LiteralToXlaFormat(
    const Literal& literal);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_DENSE_DATA_INTERMEDIATE_H_
