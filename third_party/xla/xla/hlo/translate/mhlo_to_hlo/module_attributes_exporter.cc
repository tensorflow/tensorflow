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
#include "xla/mlir_hlo/utils/unregistered_attributes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

#define DEBUG_TYPE "hlo-translate"

namespace mlir {
namespace mhlo {
namespace {

// All module level attribute strings must be registered in
//   `xla/mlir_hlo/utils/unregistered_attributes.h`.

std::vector<int64_t> ConvertDenseIntAttr(DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

absl::Status AddLayoutToShapeProto(
    const Attribute& parameter_layout,
    xla::ShapeProto* host_program_shape_parameter,
    xla::ShapeProto* computation_program_shape_parameter) {
  if (auto tuple_parameter_layout = dyn_cast<ArrayAttr>(parameter_layout)) {
    for (auto [i, tuple_element_parameter_layout] :
         llvm::enumerate(tuple_parameter_layout.getValue())) {
      TF_RETURN_IF_ERROR(AddLayoutToShapeProto(
          tuple_element_parameter_layout,
          host_program_shape_parameter->mutable_tuple_shapes(i),
          computation_program_shape_parameter->mutable_tuple_shapes(i)));
    }
    return absl::OkStatus();
  }
  auto dense_parameter_layout =
      dyn_cast<DenseIntElementsAttr>(parameter_layout);
  if (!dense_parameter_layout) {
    return absl::InvalidArgumentError(
        "Only dense elements attr is supported for parameter layout");
  }

  auto layout_dims = ConvertDenseIntAttr(dense_parameter_layout);
  // Empty layout is invalid HLO, so assume default layout.
  if (layout_dims.empty()) {
    return absl::OkStatus();
  }

  host_program_shape_parameter->mutable_layout()
      ->mutable_minor_to_major()
      ->Assign(layout_dims.begin(), layout_dims.end());
  computation_program_shape_parameter->mutable_layout()
      ->mutable_minor_to_major()
      ->Assign(layout_dims.begin(), layout_dims.end());
  return absl::OkStatus();
}

absl::Status AddTileToShapeProto(
    const Attribute& parameter_tile,
    xla::ShapeProto* host_program_shape_parameter,
    xla::ShapeProto* computation_program_shape_parameter) {
  if (auto tuple_parameter_tile = dyn_cast<ArrayAttr>(parameter_tile)) {
    for (auto [i, tuple_element_parameter_tile] :
         llvm::enumerate(tuple_parameter_tile.getValue())) {
      // Handle sub-tiles.
      if (auto dense_parameter_tile =
              dyn_cast<DenseIntElementsAttr>(tuple_element_parameter_tile)) {
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
      TF_RETURN_IF_ERROR(AddTileToShapeProto(
          tuple_element_parameter_tile,
          host_program_shape_parameter->mutable_tuple_shapes(i),
          computation_program_shape_parameter->mutable_tuple_shapes(i)));
    }
    return absl::OkStatus();
  }
  auto dense_parameter_tile = dyn_cast<DenseIntElementsAttr>(parameter_tile);
  if (!dense_parameter_tile) {
    return absl::InvalidArgumentError(
        "Only dense elements attr is supported for parameter tile");
  }

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

void ExportHloModuleConfig(xla::HloModuleConfig& config, ModuleOp module) {
  if (auto num_partitions =
          module->getAttrOfType<IntegerAttr>(xla::kMhloNumPartitions)) {
    config.set_num_partitions(num_partitions.getInt());
  }
  if (auto num_replicas =
          module->getAttrOfType<IntegerAttr>(xla::kMhloNumReplicas)) {
    config.set_replica_count(num_replicas.getInt());
  }
}

absl::Status ExportModuleEntryComputationParameterLayouts(
    const ArrayAttr& xla_entry_computation_parameter_layout,
    xla::HloModuleProto& hlo_module) {
  TF_ASSIGN_OR_RETURN(auto entry_computation, FindEntryComputation(hlo_module));

  LLVM_DEBUG(llvm::dbgs() << "Setting "
                          << xla_entry_computation_parameter_layout.size()
                          << " parameter layouts for "
                          << entry_computation->name() << "\n");

  for (auto [arg_i, parameter_layout] :
       llvm::enumerate(xla_entry_computation_parameter_layout)) {
    TF_RETURN_IF_ERROR(AddLayoutToShapeProto(
        parameter_layout,
        hlo_module.mutable_host_program_shape()->mutable_parameters()->Mutable(
            arg_i),
        entry_computation->mutable_program_shape()
            ->mutable_parameters()
            ->Mutable(arg_i)));
  }
  return absl::OkStatus();
}

absl::Status ExportModuleEntryComputationParameterTiles(
    const ArrayAttr& xla_entry_computation_parameter_tiles,
    xla::HloModuleProto& hlo_module) {
  TF_ASSIGN_OR_RETURN(auto entry_computation, FindEntryComputation(hlo_module));

  LLVM_DEBUG(llvm::dbgs() << "Setting "
                          << xla_entry_computation_parameter_tiles.size()
                          << " parameter tiles for "
                          << entry_computation->name() << "\n");

  for (auto [arg_i, parameter_tile_arg] :
       llvm::enumerate(xla_entry_computation_parameter_tiles)) {
    TF_RETURN_IF_ERROR(AddTileToShapeProto(
        parameter_tile_arg,
        hlo_module.mutable_host_program_shape()->mutable_parameters()->Mutable(
            arg_i),
        entry_computation->mutable_program_shape()
            ->mutable_parameters()
            ->Mutable(arg_i)));
  }
  return absl::OkStatus();
}

absl::Status ExportModuleEntryComputationResultLayout(
    const ArrayAttr& xla_entry_computation_result_layout,
    xla::HloModuleProto& hlo_module) {
  TF_ASSIGN_OR_RETURN(auto entry_computation, FindEntryComputation(hlo_module));
  return AddLayoutToShapeProto(
      (xla_entry_computation_result_layout.size() == 1)
          ? xla_entry_computation_result_layout[0]
          : cast<Attribute>(xla_entry_computation_result_layout),
      hlo_module.mutable_host_program_shape()->mutable_result(),
      entry_computation->mutable_program_shape()->mutable_result());
}

absl::Status ExportModuleEntryComputationResultTiles(
    const ArrayAttr& xla_entry_computation_result_tiles,
    xla::HloModuleProto& hlo_module) {
  TF_ASSIGN_OR_RETURN(auto entry_computation, FindEntryComputation(hlo_module));
  return AddTileToShapeProto(
      (xla_entry_computation_result_tiles.size() == 1)
          ? xla_entry_computation_result_tiles[0]
          : cast<Attribute>(xla_entry_computation_result_tiles),
      hlo_module.mutable_host_program_shape()->mutable_result(),
      entry_computation->mutable_program_shape()->mutable_result());
}

}  // namespace mhlo
}  // namespace mlir
