/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/sharding_param.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/python/ifrt/ir/sharding_param.pb.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace ifrt {
namespace {

template <typename T>
void PrintDims(llvm::raw_ostream& os, llvm::ArrayRef<T> dims) {
  if (dims.empty()) {
    // A scalar does not have dimensions.
    return;
  }
  os << dims[0];
  for (int i = 1; i < dims.size(); ++i) {
    os << "x" << dims[i];
  }
}

// This function runs recursively to expand `permutation` from major to minor.
// `axis_sizes` is the size of mesh dimensions before the permutation.
// `cum_sizes` is the cumulative product of the element in `sizes`.
// `base` is the start device id of this slice of `permutation`.
void PopulateDevices(llvm::ArrayRef<int> permutation,
                     llvm::ArrayRef<int> axis_sizes,
                     llvm::ArrayRef<int> cum_sizes,
                     llvm::SmallVectorImpl<int>& out_devices, int base = 0) {
  const int expanding_dim = permutation.back();
  const int expanding_dim_size = axis_sizes[expanding_dim];
  const int expanding_cum_dim_size = cum_sizes[expanding_dim];
  for (int i = 0; i < expanding_dim_size; ++i) {
    if (permutation.size() == 1) {
      out_devices.push_back(base + i * expanding_cum_dim_size);
    } else {
      PopulateDevices(permutation.drop_back(), axis_sizes, cum_sizes,
                      out_devices, base + i * expanding_cum_dim_size);
    }
  }
}

void PrintInternalV1(llvm::raw_ostream& os, const ShardingParam& sharding) {
  PrintDims(os, sharding.dim_shards());
  os << " to [";
  llvm::interleaveComma(
      llvm::ArrayRef<int>(sharding.minor_to_major().permutation), os);
  os << "] on ";
  PrintDims<int>(os, sharding.minor_to_major().axis_sizes);
}

}  // namespace

absl::Status ShardingParam::MinorToMajor::verify() const {
  if (permutation.size() != axis_sizes.size() || axis_sizes.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expect same non-zero size for `permutation` and `axis_sizes`. Actual ",
        permutation.size(), " vs ", axis_sizes.size()));
  }
  llvm::DenseSet<int> permutation_set(permutation.begin(), permutation.end());
  if (permutation_set.size() != permutation.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("`permutation` [", absl::StrJoin(permutation, ","),
                     "] has duplicate values"));
  }
  for (const int index : permutation) {
    if (index < 0 || index >= axis_sizes.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Out of range axis ", index, " to the mesh of [",
                       absl::StrJoin(permutation, ","), "] on ",
                       absl::StrJoin(axis_sizes, "x")));
    }
  }
  return absl::OkStatus();
}

mlir::LogicalResult ShardingParam::MinorToMajor::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const {
  auto status = verify();
  if (status.ok()) {
    return mlir::success();
  } else {
    return emit_error() << status.message();
  }
}

void ShardingParam::MinorToMajor::ToDeviceList(
    llvm::SmallVectorImpl<int>& out_devices) const {
  llvm::SmallVector<int, 4> cum_sizes;
  int cum_size = 1;
  cum_sizes.reserve(axis_sizes.size());
  for (auto size : axis_sizes) {
    cum_sizes.push_back(cum_size);
    cum_size *= size;
  }
  PopulateDevices(permutation, axis_sizes, cum_sizes, out_devices);
}

mlir::FailureOr<ShardingParam> ShardingParam::Parse(
    mlir::AsmParser& ods_parser) {
  // V1 is the current ShardingParam format.
  return ParseV1(ods_parser);
}

mlir::FailureOr<ShardingParam> ShardingParam::ParseV1(
    mlir::AsmParser& ods_parser) {
  MinorToMajor minor_to_major;

  auto parseIntoPermutation = [&]() -> mlir::ParseResult {
    int item;
    if (auto result = ods_parser.parseInteger(item)) {
      return result;
    } else {
      minor_to_major.permutation.push_back(item);
    }
    return mlir::ParseResult::success();
  };

  llvm::SmallVector<int64_t, 4> axis_sizes_64;
  llvm::SmallVector<int64_t> dim_shards;
  if (ods_parser.parseDimensionList(dim_shards, false, false) ||
      ods_parser.parseKeyword("to") ||
      ods_parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                         parseIntoPermutation) ||
      ods_parser.parseKeyword("on") ||
      ods_parser.parseDimensionList(axis_sizes_64, false, false)) {
    return mlir::failure();
  }

  minor_to_major.axis_sizes.reserve(axis_sizes_64.size());
  for (int64_t size : axis_sizes_64) {
    minor_to_major.axis_sizes.push_back(size);
  }
  // The copy here is necessary because parseDimensionList expects a
  // llvm::SmallVector<int64_t>, whereas ShardingParam expects a
  // std::vector<int64_t>. ShardingParam has Python bindings, so we do not want
  // its constructor to expose a SmallVector.
  return ShardingParam(std::vector(dim_shards.begin(), dim_shards.end()),
                       std::move(minor_to_major));
}

void ShardingParam::PrintV1(mlir::AsmPrinter& ods_printer,
                            const ShardingParam& sharding) {
  PrintInternalV1(ods_printer.getStream(), sharding);
}

absl::Status ShardingParam::verify() const {
  TF_RETURN_IF_ERROR(minor_to_major().verify());
  int dim_index = 0;
  int cum_size = 1;
  for (const int index : minor_to_major().permutation) {
    while (dim_index < dim_shards().size() && dim_shards()[dim_index] == 1) {
      dim_index++;
    }
    if (dim_index == dim_shards().size()) {
      break;
    }
    cum_size *= minor_to_major().axis_sizes[index];
    while (dim_index < dim_shards().size() &&
           cum_size % dim_shards()[dim_index] == 0) {
      cum_size /= dim_shards()[dim_index];
      dim_index++;
    }
  }
  while (dim_index < dim_shards().size() && dim_shards()[dim_index] == 1) {
    dim_index++;
  }
  if (dim_index != dim_shards().size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Can't shard the dims ", absl::StrJoin(dim_shards(), "x"),
        " to the mesh of [", absl::StrJoin(minor_to_major().permutation, ","),
        "] on ", absl::StrJoin(minor_to_major().axis_sizes, "x")));
  }
  return absl::OkStatus();
}

mlir::LogicalResult ShardingParam::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const {
  auto status = verify();
  if (status.ok()) {
    return mlir::success();
  } else {
    return emit_error() << status.message();
  }
}

std::string ShardingParam::DebugString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << *this;
  return result;
}

mlir::LogicalResult ShardingParam::CanApplyTo(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, llvm::ArrayRef<int> device_ids) const {
  if (mlir::failed(verify(emitError))) {
    return mlir::failure();
  }

  if (shape.getRank() != dim_shards().size()) {
    return emitError() << "Requires dim shards to have the same rank as the "
                          "array. Array rank is "
                       << shape.getRank() << " vs dim shards rank of "
                       << dim_shards().size();
  }

  auto devices_in_mesh = NumDevices();
  if (devices_in_mesh != device_ids.size()) {
    return emitError() << "Requires the same amount of `devices` and from "
                          "`sharding`. Actual: "
                       << device_ids.size() << " vs " << devices_in_mesh;
  }

  return mlir::success();
}

absl::StatusOr<llvm::SmallVector<int64_t>>
ShardingParam::GlobalShapeFromLocalShape(
    llvm::ArrayRef<int64_t> local_shape) const {
  llvm::SmallVector<int64_t> global_shape;
  if (local_shape.size() != dim_shards().size()) {
    return absl::InvalidArgumentError(
        "Rank of local tensor differs from rank of `dim_shards`.");
  }
  for (auto [idx, dim_shard] : llvm::enumerate(dim_shards())) {
    global_shape.push_back(dim_shard * local_shape[idx]);
  }
  return global_shape;
}

absl::StatusOr<llvm::SmallVector<int64_t>>
ShardingParam::LocalShapeFromGlobalShape(
    llvm::ArrayRef<int64_t> global_shape) const {
  auto num_shards = dim_shards();
  llvm::SmallVector<int64_t> local_shape;
  local_shape.reserve(global_shape.size());
  for (int i = 0; i < num_shards.size(); ++i) {
    if (global_shape[i] % num_shards[i] != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Global shape is not divisible by the number of shards in dimension ",
          i, ". Global shape: [", absl::StrJoin(global_shape, ","),
          "], number of shards: ", num_shards[i], "."));
    }
    local_shape.push_back(global_shape[i] / num_shards[i]);
  }
  return local_shape;
}

int ShardingParam::NumDevices() const {
  int devices_in_mesh = 1;
  for (const int axis_size : minor_to_major().axis_sizes) {
    devices_in_mesh *= axis_size;
  }
  return devices_in_mesh;
}

llvm::hash_code hash_value(ShardingParam sharding) {
  return sharding.hash_value();
}

mlir::AsmPrinter& operator<<(mlir::AsmPrinter& os, ShardingParam sharding) {
  // V1 if the current ShardingParam version.
  PrintInternalV1(os.getStream(), sharding);
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ShardingParam sharding) {
  // V1 if the current ShardingParam version.
  PrintInternalV1(os, sharding);
  return os;
}

absl::StatusOr<ShardingParam> ShardingParam::FromProto(
    const ShardingParamProto& proto) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation.append(proto.permutation().begin(),
                                    proto.permutation().end());
  minor_to_major.axis_sizes.append(proto.axis_sizes().begin(),
                                   proto.axis_sizes().end());
  std::vector<int64_t> dim_shards(proto.dim_shards().begin(),
                                  proto.dim_shards().end());
  return ShardingParam(std::move(dim_shards), std::move(minor_to_major));
}

absl::StatusOr<ShardingParamProto> ShardingParam::ToProto() const {
  ShardingParamProto proto;
  proto.mutable_dim_shards()->Add(dim_shards().begin(), dim_shards().end());
  proto.mutable_permutation()->Add(minor_to_major().permutation.begin(),
                                   minor_to_major().permutation.end());
  proto.mutable_axis_sizes()->Add(minor_to_major().axis_sizes.begin(),
                                  minor_to_major().axis_sizes.end());
  return proto;
}

}  // namespace ifrt
}  // namespace xla
