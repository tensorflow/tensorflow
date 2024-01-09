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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_PASSES_H_

#include <memory>
#include <optional>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {

namespace func {
class FuncOp;
}
class ModuleOp;
class Operation;
template <typename T>
class OperationPass;
class Pass;

namespace mhlo {

/// Lowers from TF dialect to HLO dialect. When allow_partial_conversion is
/// false, emits an error if there is any operation that can't be legalized.
/// When `tf2xla_fallback_device_type` is not `None`, also uses legalization
/// patterns from TF2XLA fallback for provided device type (see
/// legalize_tf_with_tf2xla.cc for details). By default, TF2XLA fallback is not
/// used.
/// Note: This is a module pass because when legalizing with TF2XLA fallback,
/// functions are imported into the module. Importing functions into a
/// module is not thread safe.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeTFPass(
    bool legalize_chlo = true,
    std::optional<StringRef> tf2xla_fallback_device_type = std::nullopt,
    bool prefer_tf2xla = false);

/// Converter to be used along with the fallback Tf2Xla patterns below.
class Tf2XlaTypeConverter : public TypeConverter {
 public:
  Tf2XlaTypeConverter();
};

/// Adds the TF to XLA via TF2XLA rewrite patterns to the pattern list.
/// `prefer_tf2xla` means an op will be included iff it is not in
/// `MlirLegalizedUnderPreferTf2XlaSet`. `!prefer_tf2xla` mean an op will be
/// included if there is no native MLIR legalization for the op.
void PopulateLegalizeTfWithTf2XlaPatterns(llvm::StringRef device_type,
                                          RewritePatternSet& patterns,
                                          MLIRContext* ctx,
                                          Tf2XlaTypeConverter& converter,
                                          bool prefer_tf2xla = false);

/// Adds the TF to TF lowerings and TF to XLA rewrite patterns to the pattern
/// list.
void PopulateLegalizeTfPatterns(MLIRContext* context,
                                RewritePatternSet* patterns);

// Populates TF to MHLO legalization for some of the quantization ops.
//
// TODO(hinsu): Remove this once we combine quantized and non quantized op
// legalization in the ODML conversion pipeline.
void PopulateLegalizeTfQuantizationPatterns(MLIRContext* context,
                                            RewritePatternSet* patterns);

/// Converts the provided Operation as well as all nested operations into HLO
/// dialect using the conversion patterns registered by the HLO dialect. When
/// allow_partial_conversion is false, emits an error if there is any operation
/// that can't be legalized.
/// When `tf2xla_fallback_device_type` is not `None`, also uses legalization
/// patterns from TF2XLA fallback for provided device type (see
/// legalize_tf_with_tf2xla.cc for details). By default, TF2XLA fallback is not
/// used.
LogicalResult legalizeTF(
    Operation* op, bool allow_partial_conversion = false,
    bool legalize_chlo = true,
    std::optional<StringRef> tf2xla_fallback_device_type = std::nullopt,
    bool prefer_tf2xla = false);

// Legalizes TF/XLA communication ops (TF dialect) to HLO dialect communication
// ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFCommunicationPass();

// Legalizes TF/XLA collective ops (TF dialect) to HLO dialect collective
// ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFCollectivePass();

// Verifies that the TF/XLA ops have all been lowered to MHLO.
std::unique_ptr<OperationPass<func::FuncOp>> CreateVerifyTFXLALegalizationPass(
    bool legalize_chlo = true);

// Transforms TFXLA Device specific ops into device independent ops.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFXLADeviceSpecificTransformsPass(
    std::optional<StringRef> tf2xla_fallback_device_type = std::nullopt);

// Adjusts XLA layout for Infeed ops.
std::unique_ptr<OperationPass<func::FuncOp>>
CreateInfeedsOpsXlaAdjustLayoutPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_INFEEDSOPSXLAADJUSTLAYOUT
#define GEN_PASS_DECL_LEGALIZETF
#define GEN_PASS_DECL_LEGALIZETFCOLLECTIVE
#define GEN_PASS_DECL_LEGALIZETFMODULEPASS
#define GEN_PASS_DECL_LEGALIZETFTYPESPASS
#define GEN_PASS_DECL_TFXLADEVICESPECIFICTRANSFORMS
#define GEN_PASS_DECL_VERIFYTFXLALEGALIZATION
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_LEGALIZETFCOMMUNICATIONPASS
#define GEN_PASS_DECL_LEGALIZETFWITHTF2XLA
#include "tensorflow/compiler/mlir/tf2xla/transforms/tf_xla_passes.h.inc"
}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_TRANSFORMS_PASSES_H_
