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

#ifndef XLA_TRANSLATE_MHLO_TO_HLO_MLIR_HLO_TO_HLO_H_
#define XLA_TRANSLATE_MHLO_TO_HLO_MLIR_HLO_TO_HLO_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/client/xla_builder.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/translate/mhlo_to_hlo/layout_util.h"

namespace mlir {

struct MlirToHloConversionOptions {
  // Best-effort propagation of the layouts. These layouts serve as performance
  // hints to the backend.
  //
  // Note that non-array shapes are not carrying layouts, and users have to
  // figure out the proper layouts of them through context. This is one of the
  // reasons why the attribute-based solution is temporary.
  //
  // TODO(timshen): Investigate the necessity of having layouts in MHLO.
  bool propagate_layouts = false;

  // Propagate the source and result layouts from mhlo bitcast op into the
  // backend config for the bitcast. This is required for XLA:GPU backend to
  // use elemental IR emitters for fused bitcasts without propagating layouts.
  bool propagate_bitcast_layouts_to_backend_config = false;

  LayoutPreferenceFn layout_preference_fn;
  ShapeRepresentationFn shape_representation_fn;

  // If use_tuple_args is set, then the entry computations's arguments are
  // converted to a tuple and passed as a single parameter.
  bool use_tuple_args = false;

  // If return tuple is true, then the entry function's return values
  // are converted to a tuple even when there is only a single return value.
  // Multiple return values are always converted to a tuple and returned as a
  // single value.
  bool return_tuple = true;
};

// Prefer `ConvertMlirHloToHloModule` over this method when possible, as it
// preserves more information and abstracts away the proto. This method is
// preserved for legacy reasons.
// TODO (b/345806521): Migrate callsites to ConvertMlirHloToHloModule,
// and delete this method.
//
// Converts a MLIR module in HLO dialect into a HloModuleProto.
//
absl::Status ConvertMlirHloToHlo(mlir::ModuleOp module,
                                 ::xla::HloProto* hlo_proto,
                                 bool use_tuple_args, bool return_tuple,
                                 MlirToHloConversionOptions options = {});

// Converts a MLIR module in HLO dialect into a HloModule with HloModuleConfig.
// This method preserves config data stored in MHLO module attributes.
//
// See `MlirToHloConversionOptions` for details on conversion flags.
absl::StatusOr<std::unique_ptr<xla::HloModule>> ConvertMlirHloToHloModule(
    mlir::ModuleOp module, MlirToHloConversionOptions options = {});

// Transforms a Block into HLO, where the HLO is represented as calls into an
// XlaBuilder. Callee functions are allowed in the Block's ancestor ModuleOp.
// xla_params are inputs to block. returns are the returned XlaOps.
absl::Status BuildHloFromMlirHlo(mlir::Block& block, xla::XlaBuilder& builder,
                                 llvm::ArrayRef<xla::XlaOp> xla_params,
                                 std::vector<xla::XlaOp>& returns,
                                 MlirToHloConversionOptions options = {});

}  // namespace mlir

#endif  // XLA_TRANSLATE_MHLO_TO_HLO_MLIR_HLO_TO_HLO_H_
