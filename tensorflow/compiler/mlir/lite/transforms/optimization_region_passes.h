/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

// Optimization regions: `optimize = "none"`.
//
// A frontend that authors TFL ops directly (ai_edge_jax `ops.*`, which emits
// them as `stablehlo.custom_call @tfl.*` carriers) can wrap a block of them in
// a `stablehlo.composite "odml.optimization_region"` and ask the converter to
// leave the block alone:
//
//     stablehlo.composite "odml.optimization_region" %x {
//       composite_attributes = {optimize = "none"}, decomposition = @f }
//
// `optimize = "none"` is the only selection this change understands. The
// attribute vocabulary (`optimize`, and per-entry `disable` / `enable` lists)
// is reserved for per-entry selection, a separate change; any other key or
// value is an error so that a stale frontend never silently gets the default.
// A region without `composite_attributes` is an ordinary region: its body is
// converted like any other function and the composite is inlined at the end
// like every region (regions never leave a subgraph boundary in the model).
//
// The contract of `optimize = "none"`, exactly:
//
//  * The `tfl.*` ops the frontend emitted as carriers reach the exported
//    model as emitted: one op per carrier, in order, with the attributes and
//    operands the frontend gave them (an explicit zero bias included). No
//    legalization-time pattern, canonicalization, CSE, Optimize round or
//    quantization pass rewrites them.
//  * Explicit `tfl.quantize` / `tfl.dequantize` carriers are literal too:
//    `quantize(f32 -> qi8)` and `dequantize(qi8 -> f32)` stay two ops with the
//    authored quantized type, the op between them runs in float, and a
//    `dequantize(int8 constant)` stays a dequantize of a quantized constant.
//    No Q/DQ pair elimination, no fusion into a quantized kernel, no
//    propagation of quantization parameters into neighbouring ops. Outside a
//    `none` region, Q/DQ handling is unchanged.
//  * What may still change: type legalization that is needed for a valid
//    TFLite model applies in `none` regions like everywhere else (x64 -> x32
//    downcasting with `enable_x64 = false`, ReduceTypePrecision,
//    CanonicalizeBoundaryValue rewriting +-inf constants), as does the
//    folding of the body's constants. Ops in the body that were not emitted
//    as `tfl.*` carriers (StableHLO ops, constants) are legalized and
//    canonicalized normally.
//
// How it works, in pipeline order:
//
//  * MarkOptimizationRegionsPass (module): right before HLO -> TFL
//    legalization, stamps the decomposition function of every region with
//    its attributes as `odml.optimization = {optimize = "none"}`. HLO -> TFL
//    legalization (IsCustomCallLegal in legalize_hlo_conversions/
//    custom_call.cc) then leaves the `custom_call @tfl.*` carriers of such
//    functions alone, so the frontend's ops travel through the TFL pipeline
//    as opaque ops that no pass has patterns for. `tfl.quantize` /
//    `tfl.dequantize` carriers are the exception: their legalization builds
//    the quantized type from backend_config, so they legalize in place and
//    are protected by the next point.
//  * SkipOptimizationRegionsPass (func): wraps every func-level rewriting pass
//    between legalization and export (canonicalize, CSE, the Optimize rounds,
//    the quantization passes, ...) and skips `optimize = "none"` functions.
//  * MaterializeOptimizationRegionsPass (func): after the last rewriting pass,
//    turns the carriers of `optimize = "none"` functions into the TFL ops they
//    name.
//  * InlineOptimizationRegionsPass (module): inlines every
//    `odml.optimization_region` composite into its caller at the end of the
//    pipeline (innermost first; regions may nest and every decomposition keeps
//    its own stamp) and removes the decomposition functions. It fails if an
//    `optimize = "none"` body still holds carriers.
//
// Known limitation: while the pipeline runs, a region body is a separate
// function, so no rewrite matches across the region boundary, and nothing
// after InlineOptimizationRegionsPass fuses the body with its caller. This is
// by design: the region's protection must not leak into the caller, and the
// caller's optimizations must not reach into the region.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZATION_REGION_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZATION_REGION_PASSES_H_

#include <memory>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_options.h"

namespace mlir {
namespace TFL {

////////////////////////////////////////////////////////////////////////////////
// The region attribute contract
////////////////////////////////////////////////////////////////////////////////

// Name of the composite a frontend region API emits around a region body, and
// of the dictionary attribute MarkOptimizationRegionsPass stamps on the body
// function (`{optimize = "none"}`). Read by anything that wants to know
// whether a function is a frontend-owned `none` region.
inline constexpr llvm::StringRef kOptimizationRegionName =
    "odml.optimization_region";
inline constexpr llvm::StringRef kOptimizationFuncAttr = "odml.optimization";

// The one `composite_attributes` key and value understood here.
inline constexpr llvm::StringRef kRegionOptimizeAttr = "optimize";
inline constexpr llvm::StringRef kRegionOptimizeNone = "none";

// True if `composite` is an `odml.optimization_region`.
bool IsOptimizationRegion(stablehlo::CompositeOp composite);

// True if `func` is the body of an `odml.optimization_region` with
// `optimize = "none"`, i.e. its carriers and explicit Q/DQ ops must be left
// untouched.
bool IsOptimizeNoneRegion(func::FuncOp func);

// Validates the `composite_attributes` of an `odml.optimization_region` (or
// the stamped `odml.optimization` function attribute). Returns true for
// `{optimize = "none"}`, false for a null or empty dictionary, and emits an
// error on `op` and fails for anything else.
FailureOr<bool> ParseOptimizeNone(DictionaryAttr attrs, Operation* op);

// Stamps the decomposition function of every `odml.optimization_region`
// composite (`stablehlo.composite` or `mhlo.composite`, whichever the module
// holds when this runs) with its validated attributes as the
// `odml.optimization` attribute. Idempotent; a decomposition shared by regions
// with different attributes is an error.
LogicalResult StampOptimizationRegions(ModuleOp module);

////////////////////////////////////////////////////////////////////////////////
// MarkOptimizationRegionsPass
////////////////////////////////////////////////////////////////////////////////

class MarkOptimizationRegionsPass
    : public Pass<MarkOptimizationRegionsPass, EmptyPassOptions, ModuleOp> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkOptimizationRegionsPass)

  MarkOptimizationRegionsPass() = default;
  MarkOptimizationRegionsPass(const MarkOptimizationRegionsPass&) {}

  void runOnOperation() override;

  static llvm::StringRef GetName() { return "MarkOptimizationRegionsPass"; }
  static llvm::StringRef GetArgument() {
    return "tfl-mark-optimization-regions";
  }
  static llvm::StringRef GetDescription() {
    return "Stamps decomposition functions of `odml.optimization_region` "
           "composites with their validated attributes as an "
           "`odml.optimization` attribute.";
  }
};

////////////////////////////////////////////////////////////////////////////////
// MaterializeOptimizationRegionsPass
////////////////////////////////////////////////////////////////////////////////

// In `optimize = "none"` functions, turns the `custom_call @tfl.*` carriers
// that HLO -> TFL legalization deliberately left alone into the TFL ops they
// name, exactly as the frontend emitted them. Runs after every rewriting pass
// of the TFL pipeline; anything after it must be gated with
// SkipOptimizationRegionsPass, be a pure check, or be a type-legalizing pass
// that the contract above allows.
class MaterializeOptimizationRegionsPass
    : public Pass<MaterializeOptimizationRegionsPass, EmptyPassOptions,
                  func::FuncOp> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeOptimizationRegionsPass)

  MaterializeOptimizationRegionsPass() = default;
  MaterializeOptimizationRegionsPass(
      const MaterializeOptimizationRegionsPass&) {}

  void runOnOperation() override;

  static llvm::StringRef GetName() {
    return "MaterializeOptimizationRegionsPass";
  }
  static llvm::StringRef GetArgument() {
    return "tfl-materialize-optimization-regions";
  }
  static llvm::StringRef GetDescription() {
    return "Converts `custom_call @tfl.*` carriers in "
           "`odml.optimization = {optimize = \"none\"}` functions into the "
           "named TFL ops.";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
};

////////////////////////////////////////////////////////////////////////////////
// SkipOptimizationRegionsPass
////////////////////////////////////////////////////////////////////////////////

// Runs a func.func pass pipeline on every function except `optimize = "none"`
// region bodies. This is how the rewriting passes that sit between HLO -> TFL
// legalization and export (canonicalize, CSE, Optimize, quantization, ...)
// are kept off frontend-owned regions without each of them learning about
// the attribute.
//
// MLIR's pass manager nests by op name only and cannot filter on attributes,
// so the filter has to be a pass. Build it with
// `CreateSkipOptimizationRegionsPass` in C++; from the command line use
// `-tfl-skip-optimization-regions=pipeline=...` with a textual func-level
// pipeline (e.g. `pipeline=canonicalize`).
class SkipOptimizationRegionsPass
    : public PassWrapper<SkipOptimizationRegionsPass,
                         OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SkipOptimizationRegionsPass)

  SkipOptimizationRegionsPass() = default;
  SkipOptimizationRegionsPass(const SkipOptimizationRegionsPass& other);
  explicit SkipOptimizationRegionsPass(std::unique_ptr<mlir::Pass> inner);

  void runOnOperation() override;

  llvm::StringRef getArgument() const override { return GetArgument(); }
  llvm::StringRef getDescription() const override { return GetDescription(); }
  static llvm::StringRef GetArgument() {
    return "tfl-skip-optimization-regions";
  }
  static llvm::StringRef GetDescription() {
    return "Runs the given func.func pipeline on every function that is not "
           "an `odml.optimization = {optimize = \"none\"}` region body.";
  }

 private:
  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine&)> error_handler) override;
  void getDependentDialects(DialectRegistry& registry) const override;

  Option<std::string> pipeline_{
      *this, "pipeline",
      llvm::cl::desc("Textual func.func pass pipeline to run on functions "
                     "that are not optimize=\"none\" optimization regions."),
      llvm::cl::init("")};
  std::unique_ptr<OpPassManager> pm_;
};

std::unique_ptr<mlir::Pass> CreateSkipOptimizationRegionsPass(
    std::unique_ptr<mlir::Pass> inner);

////////////////////////////////////////////////////////////////////////////////
// InlineOptimizationRegionsPass
////////////////////////////////////////////////////////////////////////////////

// Inlines every `odml.optimization_region` composite into its caller,
// innermost first, and removes the decomposition functions that are left
// without users. Fails if a composite cannot be inlined, if any survives, or
// if an `optimize = "none"` decomposition still holds `custom_call @tfl.*`
// carriers (MaterializeOptimizationRegionsPass must run first; carriers must
// never reach the exporter).
class InlineOptimizationRegionsPass
    : public Pass<InlineOptimizationRegionsPass, EmptyPassOptions, ModuleOp> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlineOptimizationRegionsPass)

  InlineOptimizationRegionsPass() = default;
  InlineOptimizationRegionsPass(const InlineOptimizationRegionsPass&) {}

  void runOnOperation() override;

  static llvm::StringRef GetName() { return "InlineOptimizationRegionsPass"; }
  static llvm::StringRef GetArgument() {
    return "tfl-inline-optimization-regions";
  }
  static llvm::StringRef GetDescription() {
    return "Inlines `odml.optimization_region` composites into their callers "
           "and removes the dead decomposition functions.";
  }

 private:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TFL::TensorFlowLiteDialect>();
  }
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_OPTIMIZATION_REGION_PASSES_H_
