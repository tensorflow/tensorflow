/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSLATE_MHLO_TO_HLO_ATTRIBUTE_EXPORTER_H_
#define XLA_HLO_TRANSLATE_MHLO_TO_HLO_ATTRIBUTE_EXPORTER_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/dnn.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Converts the conv dimensions attribute to XLA HLO.
ConvolutionDimensionNumbers ConvertConvDimensionNumbers(
    mlir::mhlo::ConvDimensionNumbersAttr input);

// Converts the conv dimensions attribute to XLA HLO.
ConvolutionDimensionNumbers ConvertConvDimensionNumbers(
    mlir::stablehlo::ConvDimensionNumbersAttr input);

// Converts the dot algorithm attribute to XLA HLO.
absl::StatusOr<xla::PrecisionConfig::Algorithm> ConvertDotAlgorithm(
    mlir::mhlo::DotAlgorithmAttr attr);

absl::StatusOr<xla::PrecisionConfig::Algorithm> ConvertDotAlgorithm(
    mlir::stablehlo::DotAlgorithmAttr attr);

absl::StatusOr<std::vector<ReplicaGroup>> ConvertReplicaGroups(
    mlir::DenseIntElementsAttr input);

// Convert a (N, 2) dense attribute to a list of tuples. This is the way padding
// and source-target pairs are defined in HLO.
absl::StatusOr<std::vector<std::pair<int64_t, int64_t>>> ConvertNx2Attribute(
    std::optional<mlir::DenseIntElementsAttr> optional_attr);

absl::StatusOr<TriangularSolveOptions::Transpose> ConvertTranspose(
    llvm::StringRef transpose_string);

absl::StatusOr<xla::CustomCallSchedule> ConvertCustomCallSchedule(
    mlir::mhlo::CustomCallSchedule schedule);

absl::StatusOr<xla::CustomCallApiVersion> ConvertCustomCallApiVersion(
    mlir::mhlo::CustomCallApiVersion api_version);

absl::StatusOr<xla::CustomCallApiVersion> ConvertCustomCallApiVersion(
    mlir::stablehlo::CustomCallApiVersion api_version);

absl::StatusOr<
    std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>>
ConvertOutputOperandAliasing(mlir::ArrayAttr aliasArrayAttr);

// Returns an OpSharding that represents the result of parsing the given string:
// first, as serialized protobuf, and then as prettyprinted representation.
// Will fail if both attempts at parsing failed.
std::optional<xla::OpSharding> ConvertSharding(mlir::StringRef sharding);

// Returns an OpSharding that represents the Shardy sharding contained within
// the frontend attributes of the argument at `arg_num` in `function`. Returns
// std::nullopt if no sharding is found.
std::optional<xla::OpSharding> ExtractShardyArgShardingFromFrontendAttrs(
    mlir::func::FuncOp function, int64_t arg_num,
    std::optional<mlir::DictionaryAttr> sdy_meshes);

// Returns an OpSharding that represents the Shardy sharding contained within
// the frontend attributes of the result at `res_num` in `function`. Returns
// std::nullopt if no sharding is found.
std::optional<xla::OpSharding> ExtractShardyResultShardingFromFrontendAttrs(
    mlir::func::FuncOp function, int64_t res_num,
    std::optional<mlir::DictionaryAttr> sdy_meshes);

// Returns an OriginalValueProto that represents a value in the unoptimized HLO
// graph.
std::optional<xla::OriginalValueProto> ConvertOriginalValue(
    llvm::StringRef original_value, const xla::Shape& shape);

std::optional<xla::HloInputOutputAliasProto> ConvertInputOutputAlias(
    llvm::ArrayRef<mlir::Attribute> aliasing);

template <typename OutputOperandAliasAttrTy>
absl::StatusOr<
    std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>>
ConvertOutputOperandAliasing(mlir::ArrayAttr aliasArrayAttr) {
  std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>> aliasInfo;
  for (auto attr : aliasArrayAttr.getValue()) {
    auto alias = mlir::cast<OutputOperandAliasAttrTy>(attr);
    ShapeIndex outputShapeIndex(alias.getOutputTupleIndices());
    ShapeIndex operandShapeIndex(alias.getOperandTupleIndices());
    aliasInfo.push_back(std::make_pair(
        outputShapeIndex,
        std::make_pair(alias.getOperandIndex(), operandShapeIndex)));
  }
  return aliasInfo;
}

}  // namespace xla
#endif  // XLA_HLO_TRANSLATE_MHLO_TO_HLO_ATTRIBUTE_EXPORTER_H_
