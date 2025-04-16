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

#include "xla/hlo/translate/mhlo_to_hlo/module_attributes_exporter.h"

#include <cassert>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/xla_data.pb.h"

#define DEBUG_TYPE "hlo-translate"

namespace mlir {
namespace mhlo {
namespace {

constexpr char kMhloNumPartitions[] = "mhlo.num_partitions";
constexpr char kMhloNumReplicas[] = "mhlo.num_replicas";

std::vector<int64_t> ConvertDenseIntAttr(mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

absl::Status AddLayoutToShapeProto(
    const mlir::Attribute& parameter_layout,
    xla::ShapeProto* host_program_shape_parameter,
    xla::ShapeProto* computation_program_shape_parameter) {
  if (auto tuple_parameter_layout =
          mlir::dyn_cast<mlir::ArrayAttr>(parameter_layout)) {
    for (auto [i, tuple_element_parameter_layout] :
         llvm::enumerate(tuple_parameter_layout.getValue())) {
      auto status = AddLayoutToShapeProto(
          tuple_element_parameter_layout,
          host_program_shape_parameter->mutable_tuple_shapes(i),
          computation_program_shape_parameter->mutable_tuple_shapes(i));
      if (!status.ok()) return status;
    }
    return absl::OkStatus();
  }
  auto dense_parameter_layout =
      mlir::dyn_cast<mlir::DenseIntElementsAttr>(parameter_layout);
  if (!dense_parameter_layout)
    return absl::InvalidArgumentError(
        "Only dense elements attr is supported for parameter layout");

  auto layout_dims = ConvertDenseIntAttr(dense_parameter_layout);
  // Empty layout is invalid HLO, so assume default layout.
  if (layout_dims.empty()) return absl::OkStatus();

  host_program_shape_parameter->mutable_layout()
      ->mutable_minor_to_major()
      ->Assign(layout_dims.begin(), layout_dims.end());
  computation_program_shape_parameter->mutable_layout()
      ->mutable_minor_to_major()
      ->Assign(layout_dims.begin(), layout_dims.end());
  return absl::OkStatus();
}

absl::Status AddTileToShapeProto(
    const mlir::Attribute& parameter_tile,
    xla::ShapeProto* host_program_shape_parameter,
    xla::ShapeProto* computation_program_shape_parameter) {
  if (auto tuple_parameter_tile =
          mlir::dyn_cast<mlir::ArrayAttr>(parameter_tile)) {
    for (auto [i, tuple_element_parameter_tile] :
         llvm::enumerate(tuple_parameter_tile.getValue())) {
      // Handle sub-tiles.
      if (auto dense_parameter_tile =
              mlir::dyn_cast<mlir::DenseIntElementsAttr>(
                  tuple_element_parameter_tile)) {
        auto tile_dims = ConvertDenseIntAttr(dense_parameter_tile);
        if (!tile_dims.empty()) {
          xla::TileProto tile;
          tile.mutable_dimensions()->Assign(tile_dims.begin(), tile_dims.end());
          *host_program_shape_parameter->mutable_layout()
               ->mutable_tiles()
               ->Add() = tile;
          *computation_program_shape_parameter->mutable_layout()
               ->mutable_tiles()
               ->Add() = tile;
        }

        // Empty tile is invalid HLO, so assume default tile (no tile).
        continue;
      }
      auto status = AddTileToShapeProto(
          tuple_element_parameter_tile,
          host_program_shape_parameter->mutable_tuple_shapes(i),
          computation_program_shape_parameter->mutable_tuple_shapes(i));
      if (!status.ok()) return status;
    }
    return absl::OkStatus();
  }
  auto dense_parameter_tile =
      mlir::dyn_cast<mlir::DenseIntElementsAttr>(parameter_tile);
  if (!dense_parameter_tile)
    return absl::InvalidArgumentError(
        "Only dense elements attr is supported for parameter tile");

  // Empty tile is valid HLO. E.g. f32[]{:T()}
  xla::TileProto tile;
  auto tile_dims = ConvertDenseIntAttr(dense_parameter_tile);
  tile.mutable_dimensions()->Assign(tile_dims.begin(), tile_dims.end());
  *host_program_shape_parameter->mutable_layout()->mutable_tiles()->Add() =
      tile;
  *computation_program_shape_parameter->mutable_layout()
       ->mutable_tiles()
       ->Add() = tile;
  return absl::OkStatus();
}

// Finds the entry XLA computation.
absl::StatusOr<xla::HloComputationProto*> FindEntryComputation(
    xla::HloModuleProto& hlo) {
  const int id = hlo.entry_computation_id();
  for (auto& c : *hlo.mutable_computations()) {
    if (c.id() == id) {
      return &c;
    }
  }
  return absl::InvalidArgumentError("Missing entry computation");
}

}  // namespace

void ExportHloModuleConfig(xla::HloModuleConfig& config,
                           mlir::ModuleOp module) {
  if (auto num_partitions =
          module->getAttrOfType<mlir::IntegerAttr>(kMhloNumPartitions)) {
    config.set_num_partitions(num_partitions.getInt());
  }
  if (auto num_replicas =
          module->getAttrOfType<mlir::IntegerAttr>(kMhloNumReplicas)) {
    config.set_replica_count(num_replicas.getInt());
  }
}

absl::Status ExportModuleEntryComputationParameterLayouts(
    const mlir::ArrayAttr& xla_entry_computation_parameter_layout,
    xla::HloModuleProto& hlo_module) {
  auto entry_computation = FindEntryComputation(hlo_module);
  if (!entry_computation.ok()) return entry_computation.status();

  auto entry_computation_params =
      entry_computation.value()->mutable_program_shape()->mutable_parameters();
  auto host_program_params =
      hlo_module.mutable_host_program_shape()->mutable_parameters();

  LLVM_DEBUG(llvm::dbgs() << "Setting "
                          << xla_entry_computation_parameter_layout.size()
                          << " parameter layouts for "
                          << entry_computation.value()->name() << "\n");

  for (auto [arg_i, parameter_layout] :
       llvm::enumerate(xla_entry_computation_parameter_layout)) {
    auto status = AddLayoutToShapeProto(
        parameter_layout, host_program_params->Mutable(arg_i),
        entry_computation_params->Mutable(arg_i));
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}

absl::Status ExportModuleEntryComputationParameterTiles(
    const mlir::ArrayAttr& xla_entry_computation_parameter_tiles,
    xla::HloModuleProto& hlo_module) {
  auto entry_computation = FindEntryComputation(hlo_module);
  if (!entry_computation.ok()) return entry_computation.status();

  auto entry_computation_params =
      entry_computation.value()->mutable_program_shape()->mutable_parameters();
  auto host_program_params =
      hlo_module.mutable_host_program_shape()->mutable_parameters();

  LLVM_DEBUG(llvm::dbgs() << "Setting "
                          << xla_entry_computation_parameter_tiles.size()
                          << " parameter tiles for "
                          << entry_computation.value()->name() << "\n");

  for (auto [arg_i, parameter_tile_arg] :
       llvm::enumerate(xla_entry_computation_parameter_tiles)) {
    auto status = AddTileToShapeProto(parameter_tile_arg,
                                      host_program_params->Mutable(arg_i),
                                      entry_computation_params->Mutable(arg_i));
    if (!status.ok()) return status;
  }
  return absl::OkStatus();
}

absl::Status ExportModuleEntryComputationResultLayout(
    const mlir::ArrayAttr& xla_entry_computation_result_layout,
    xla::HloModuleProto& hlo_module) {
  // Assume only one result is allowed.
  return AddLayoutToShapeProto(
      xla_entry_computation_result_layout[0],
      hlo_module.mutable_host_program_shape()->mutable_result(),
      hlo_module.mutable_computations(0)
          ->mutable_program_shape()
          ->mutable_result());
}

absl::Status ExportModuleEntryComputationResultTiles(
    const mlir::ArrayAttr& xla_entry_computation_result_tiles,
    xla::HloModuleProto& hlo_module) {
  // Assume only one result is allowed.
  return AddTileToShapeProto(
      xla_entry_computation_result_tiles[0],
      hlo_module.mutable_host_program_shape()->mutable_result(),
      hlo_module.mutable_computations(0)
          ->mutable_program_shape()
          ->mutable_result());
}

}  // namespace mhlo
}  // namespace mlir
