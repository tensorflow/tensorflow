/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_MLIR_HLO_TO_HLO_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_MLIR_HLO_TO_HLO_H_

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/framework/tensor_shape.h"

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

  // Legalize names to be compatible with TensorFlow.
  bool legalize_node_names = true;
};

// Converts a MLIR module in HLO dialect into a HloModuleProto. If
// use_tuple_args is set, then the entry computations's arguments are converted
// to a tuple and passed as a single parameter.
// Similarly, if return tuple is true, then the entry function's return values
// are converted to a tuple even when there is only a single return value.
// Multiple return values are always converted to a tuple and returned as a
// single value.
//
// TODO(timshen): move other options into `options`.
Status ConvertMlirHloToHlo(mlir::ModuleOp module, ::xla::HloProto* hlo_proto,
                           bool use_tuple_args, bool return_tuple,
                           const tensorflow::XlaHelpers::ShapeRepresentationFn
                               shape_representation_fn = nullptr,
                           MlirToHloConversionOptions options = {});

// Transforms a Block into HLO, where the HLO is represented as calls into an
// XlaBuilder. Callee functions are allowed in the Block's ancestor ModuleOp.
// xla_params are inputs to block. returns are the returned XlaOps.
Status BuildHloFromMlirHlo(mlir::Block& block, xla::XlaBuilder& builder,
                           llvm::ArrayRef<xla::XlaOp> xla_params,
                           std::vector<xla::XlaOp>& returns,
                           MlirToHloConversionOptions options = {});

// Converts a region to a computation. It returns a standalone module that
// contains the converted region as the entry computation.
Status ConvertRegionToComputation(mlir::Region* region,
                                  ::xla::XlaComputation* func,
                                  MlirToHloConversionOptions options = {});

// Creates XlaOp equivalent of a given MLIR operation using the operand info
// from `value_lowering` map.
llvm::Optional<::xla::XlaOp> CreateXlaOperator(
    mlir::Operation* op,
    llvm::DenseMap<mlir::Value, ::xla::XlaOp>* value_lowering);

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_MLIR_HLO_TO_HLO_H_
