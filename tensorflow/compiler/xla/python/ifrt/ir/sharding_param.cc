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

#include "tensorflow/compiler/xla/python/ifrt/ir/sharding_param.h"

#include <cstdint>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace xla {
namespace ifrt {
namespace {

void PrintDims(llvm::raw_ostream& os, llvm::ArrayRef<int64_t> dims) {
  os << dims[0];
  for (int i = 1; i < dims.size(); ++i) {
    os << "x" << dims[i];
  }
}

// This function runs recursively to expand `permutation` from major to minor.
// `axis_sizes` is the size of mesh dimensions before the permutation.
// `cum_sizes` is the cumulative product of the element in `sizes`.
// `base` is the start device id of this slice of `permutation`.
void PopulateDevices(llvm::ArrayRef<int64_t> permutation,
                     llvm::ArrayRef<int64_t> axis_sizes,
                     llvm::ArrayRef<int64_t> cum_sizes,
                     llvm::SmallVectorImpl<int64_t>& out_devices,
                     int64_t base = 0) {
  const int64_t expanding_dim = permutation.back();
  const int64_t expanding_dim_size = axis_sizes[expanding_dim];
  const int64_t expanding_cum_dim_size = cum_sizes[expanding_dim];
  for (int64_t i = 0; i < expanding_dim_size; ++i) {
    if (permutation.size() == 1) {
      out_devices.push_back(base + i * expanding_cum_dim_size);
    } else {
      PopulateDevices(permutation.drop_back(), axis_sizes, cum_sizes,
                      out_devices, base + i * expanding_cum_dim_size);
    }
  }
}

}  // namespace

mlir::LogicalResult ShardingParam::MinorToMajor::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const {
  if (permutation.size() != axis_sizes.size()) {
    return emit_error()
           << "Expect same size for `permutation` and `axis_sizes`. Actual "
           << permutation.size() << " vs " << axis_sizes.size();
  }
  return mlir::success();
}

void ShardingParam::MinorToMajor::ToDeviceList(
    llvm::SmallVectorImpl<int64_t>& out_devices) const {
  llvm::SmallVector<int64_t, 4> cum_sizes;
  int64_t cum_size = 1;
  cum_sizes.reserve(axis_sizes.size());
  for (auto size : axis_sizes) {
    cum_sizes.push_back(cum_size);
    cum_size *= size;
  }
  PopulateDevices(permutation, axis_sizes, cum_sizes, out_devices);
}

mlir::FailureOr<ShardingParam> ShardingParam::Parse(
    mlir::AsmParser& ods_parser) {
  llvm::SmallVector<int64_t, 4> dim_shards;
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

  if (ods_parser.parseDimensionList(dim_shards, false, false) ||
      ods_parser.parseKeyword("to") ||
      ods_parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                         parseIntoPermutation) ||
      ods_parser.parseKeyword("on") ||
      ods_parser.parseDimensionList(minor_to_major.axis_sizes, false, false)) {
    return mlir::failure();
  }

  return ShardingParam(dim_shards, minor_to_major);
}

mlir::LogicalResult ShardingParam::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const {
  if (mlir::failed(minor_to_major().verify(emit_error))) {
    return mlir::failure();
  }

  int64_t dim_index = 0;
  int64_t cum_size = 1;
  for (const int64_t index : minor_to_major().permutation) {
    while (dim_index < dim_shards().size() && dim_shards()[dim_index] == 1) {
      dim_index++;
    }
    if (dim_index == dim_shards().size()) {
      break;
    }

    cum_size *= minor_to_major().axis_sizes[index];
    if (cum_size > dim_shards()[dim_index]) {
      return emit_error() << "Dimension #" << dim_index << " of "
                          << dim_shards()[dim_index]
                          << " shards can't be assigned to the axes";
    } else if (cum_size == dim_shards()[dim_index]) {
      cum_size = 1;
      dim_index++;
    }
  }
  while (dim_index < dim_shards().size() && dim_shards()[dim_index] == 1) {
    dim_index++;
  }
  if (dim_index != dim_shards().size()) {
    return emit_error() << "Can't shard the dims " << dim_shards()
                        << " to the mesh of " << minor_to_major().permutation
                        << " on " << minor_to_major().axis_sizes;
  }

  return mlir::success();
}

std::string ShardingParam::DebugString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << *this;
  return result;
}

llvm::hash_code hash_value(ShardingParam sharding) {
  return sharding.hash_value();
}

mlir::AsmPrinter& operator<<(mlir::AsmPrinter& os, ShardingParam sharding) {
  os.getStream() << sharding;
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ShardingParam sharding) {
  PrintDims(os, sharding.dim_shards());
  os << " to [";
  llvm::interleaveComma(
      llvm::ArrayRef<int64_t>(sharding.minor_to_major().permutation), os);
  os << "] on ";
  PrintDims(os, sharding.minor_to_major().axis_sizes);
  return os;
}

}  // namespace ifrt
}  // namespace xla
