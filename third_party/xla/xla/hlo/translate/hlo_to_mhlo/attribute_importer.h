/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSLATE_HLO_TO_MHLO_ATTRIBUTE_IMPORTER_H_
#define XLA_HLO_TRANSLATE_HLO_TO_MHLO_ATTRIBUTE_IMPORTER_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Converts an XLA PrecisionConfig to the corresponding MLIR attribute.
mlir::ArrayAttr ConvertPrecisionConfig(const PrecisionConfig* config,
                                       mlir::Builder* builder);

// Converts the gather dimensions to attributes.
mlir::mhlo::GatherDimensionNumbersAttr ConvertGatherDimensionNumbers(
    const xla::GatherDimensionNumbers& dnums, mlir::Builder* builder);

// Converts the scatter dimensions to attributes.
mlir::mhlo::ScatterDimensionNumbersAttr ConvertScatterDimensionNumbers(
    const xla::ScatterDimensionNumbers& dnums, mlir::Builder* builder);

// Converts the dot algorithm to attributes.
mlir::mhlo::DotAlgorithmAttr ConvertDotAlgorithm(
    PrecisionConfig::Algorithm algorithm, mlir::Builder* builder);

// Converts the dot dimensions to attributes.
mlir::mhlo::DotDimensionNumbersAttr ConvertDotDimensionNumbers(
    const DotDimensionNumbers& dnums, mlir::Builder* builder);

// Converts ragged dot dimensions to attributes.
mlir::mhlo::RaggedDotDimensionNumbersAttr ConvertRaggedDotDimensionNumbers(
    const RaggedDotDimensionNumbers& dnums, mlir::Builder* builder);

// Converts the conv dimensions to attributes.
mlir::mhlo::ConvDimensionNumbersAttr ConvertConvDimensionNumbers(
    const xla::ConvolutionDimensionNumbers& dnums, mlir::Builder* builder);

// Converts the output operand aliasing to attributes.
mlir::ArrayAttr ConvertOutputOperandAliasing(
    const std::vector<std::pair<xla::ShapeIndex,
                                std::pair<int64_t, xla::ShapeIndex>>>& aliaInfo,
    mlir::Builder* builder);

// Converts the sparsity descriptor to attributes.
absl::StatusOr<mlir::mhlo::SparsityDescriptorAttr> ConvertSparsityDescriptor(
    xla::SparsityDescriptor sparsity_descriptor, mlir::Builder* builder);

absl::StatusOr<mlir::mhlo::FftType> ConvertFftType(FftType type);
absl::StatusOr<mlir::mhlo::Transpose> ConvertTranspose(
    TriangularSolveOptions_Transpose transpose);

absl::StatusOr<mlir::mhlo::CustomCallApiVersion> ConvertCustomCallApiVersion(
    xla::CustomCallApiVersion api_version);

mlir::NamedAttribute ConvertChannelHandle(const ChannelHandle& channel,
                                          mlir::Builder* builder);
mlir::NamedAttribute ConvertChannelHandle(std::optional<int64_t> channel_id,
                                          mlir::Builder* builder);

mlir::NamedAttribute ConvertReplicaGroups(
    absl::Span<const ReplicaGroup> replica_groups, mlir::Builder* builder);

mlir::NamedAttribute ConvertSourceTargetPairs(
    const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
    mlir::Builder* builder);

mlir::NamedAttribute ConvertUseGlobalDeviceIds(mlir::Builder* builder);

// Extracts layouts from shapes and converts it into layout attributes (array of
// rank-1 index tensors). Returns an error if any of the shapes is a tuple.
absl::StatusOr<mlir::ArrayAttr> ExtractLayoutsFromShapes(
    const absl::Span<const Shape> shapes_with_layouts, mlir::Builder* builder);

// Extracts the layouts of each element from a tuple shape and returns them as
// an array of rank-1 index tensors. Returns an error in presence of nested
// tuple shapes.
absl::StatusOr<mlir::ArrayAttr> ExtractLayoutsFromTuple(const xla::Shape shape,
                                                        mlir::Builder* builder);

}  // namespace xla

#endif  // XLA_HLO_TRANSLATE_HLO_TO_MHLO_ATTRIBUTE_IMPORTER_H_
