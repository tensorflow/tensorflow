/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/Support/raw_os_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/mlir/xla/xla_mlir_translate.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
namespace {
// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_use_tuple_arg(
    "emit-use-tuple-args",
    llvm::cl::desc(
        "Emit HLO modules using tuples as args for the entry computation"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_return_tuple(
    "emit-return-tuple",
    llvm::cl::desc("Emit HLO modules with entry computations returning tuple"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> optimize_xla_hlo(
    "optimize-xla-hlo",
    llvm::cl::desc("Enable optimizations when translating XLA HLO -> LHLO"),
    llvm::cl::init(true));

// NOLINTNEXTLINE
llvm::cl::opt<bool> legalize_node_names(
    "legalize-node-names",
    llvm::cl::desc("Legalize nodes names when translating MHLO->XLA HLO"),
    llvm::cl::init(true));

// NOLINTNEXTLINE
llvm::cl::opt<bool> with_layouts(
    "with-layouts",
    llvm::cl::desc("Propagate layouts when translating MHLO->XLA HLO"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_layouts(
    "print-layouts", llvm::cl::desc("Print layouts in the generated HLO text"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> via_builder(
    "via-builder", llvm::cl::desc("Translate MHLO->XLA HLO via XLA Builder"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> import_all_computations(
    "hlo-import-all-computations",
    llvm::cl::desc("Enable importing unreachable computations."));
}  // namespace

namespace xla {

static mlir::LogicalResult MlirHloToHloTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
  if (!module) return mlir::failure();

  HloProto hloProto;
  Status status = mlir::ConvertMlirHloToHlo(
      module, &hloProto, emit_use_tuple_arg, emit_return_tuple);
  if (!status.ok()) {
    LOG(ERROR) << "Module conversion failed: " << status;
    return mlir::failure();
  }

  output << hloProto.DebugString();
  return mlir::success();
}

static StatusOr<std::unique_ptr<HloModule>> HloModuleFromProto(
    const HloProto& hlo_proto) {
  const HloModuleProto& module_proto = hlo_proto.hlo_module();
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          module_proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(module_proto, module_config);
}

// Wraps BuildHloFromMlirHlo to output an HloProto that's the same as
// ConvertMlirHloToHlo.
Status ConvertMlirHloToHloViaBuilder(mlir::ModuleOp module,
                                     ::xla::HloProto* hlo_proto,
                                     mlir::MlirToHloConversionOptions options) {
  mlir::func::FuncOp main = module.lookupSymbol<mlir::func::FuncOp>("main");
  mlir::Block& block = main.getRegion().front();
  xla::XlaBuilder builder("main");

  // Create xla_params.
  std::vector<xla::XlaOp> xla_params;
  for (mlir::BlockArgument& arg : block.getArguments()) {
    auto num = arg.getArgNumber();
    xla::Shape shape = xla::TypeToShape(arg.getType());
    XlaOp argop =
        xla::Parameter(&builder, num, shape, absl::StrCat("Arg_", num));
    xla_params.push_back(argop);
  }

  std::vector<xla::XlaOp> returns(1);
  TF_RETURN_IF_ERROR(
      mlir::BuildHloFromMlirHlo(block, builder, xla_params, returns, options));

  xla::XlaOp return_value;
  if (returns.size() == 1)
    return_value = returns[0];
  else if (returns.size() > 1)
    return_value = xla::Tuple(&builder, returns);

  TF_ASSIGN_OR_RETURN(
      xla::XlaComputation computation,
      return_value.valid() ? builder.Build(return_value) : builder.Build());
  auto hlo_module = computation.proto();
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);

  return ::tensorflow::OkStatus();
}

static mlir::LogicalResult MlirHloToHloTextTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
  if (!module) return mlir::failure();

  HloProto hloProto;
  mlir::MlirToHloConversionOptions options;
  options.propagate_layouts = with_layouts;
  options.legalize_node_names = legalize_node_names;
  Status status =
      via_builder
          ? ConvertMlirHloToHloViaBuilder(module, &hloProto, options)
          : mlir::ConvertMlirHloToHlo(module, &hloProto, emit_use_tuple_arg,
                                      emit_return_tuple, options);
  if (!status.ok()) {
    LOG(ERROR) << "Module conversion failed: " << status;
    return mlir::failure();
  }

  auto statusOrHloModule = HloModuleFromProto(hloProto);

  if (!statusOrHloModule.ok()) {
    LOG(ERROR) << "Conversion to HLO module failed: "
               << statusOrHloModule.status();
    return mlir::failure();
  }

  HloModule* hlo_module = statusOrHloModule.value().get();

  output << hlo_module->ToString(
      HloPrintOptions().set_include_layout_in_shapes(print_layouts));

  // Output alias information as comments in the HLO text.
  hlo_module->input_output_alias_config().ForEachAlias(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) {
        output << "// OutputIndex " << output_index.ToString()
               << " aliases with input " << alias.parameter_number << " at "
               << alias.parameter_index.ToString() << "\n";
      });

  return mlir::success();
}

}  // namespace xla

//----------------------------------------------------------------------------//
// Hooks for tf-mlir-translate
//----------------------------------------------------------------------------/

static mlir::OwningOpRef<mlir::ModuleOp> HloToMlirHloTranslate(
    llvm::StringRef input, mlir::MLIRContext* context) {
  return xla::HloToMlirHloTranslateFunction(input, context,
                                            import_all_computations);
}

static mlir::OwningOpRef<mlir::ModuleOp> HloTextToMlirHloTranslate(
    llvm::StringRef input, mlir::MLIRContext* context) {
  return xla::HloTextToMlirHloTranslateFunction(input, context,
                                                import_all_computations);
}

static void RegisterInputDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::mhlo::MhloDialect, mlir::tensor::TensorDialect>();
}

static mlir::TranslateFromMLIRRegistration MlirHloToHloTranslate(
    "mlir-hlo-to-hlo", xla::MlirHloToHloTranslateFunction,
    RegisterInputDialects);

static mlir::TranslateFromMLIRRegistration MlirHloToHloTextTranslate(
    "mlir-hlo-to-hlo-text", xla::MlirHloToHloTextTranslateFunction,
    RegisterInputDialects);

static mlir::TranslateToMLIRRegistration HloToHloMlirTranslate(
    "hlo-to-mlir-hlo", HloToMlirHloTranslate);

static mlir::TranslateToMLIRRegistration HloTextToHloMlirTranslate(
    "hlo-text-to-mlir-hlo", HloTextToMlirHloTranslate);

// MHLO doesn't support explicit layouts, while XLA service does.
// TODO(timshen): remove it once MHLO supports explicit layouts.
static mlir::TranslateToMLIRRegistration HloTextToLhloMlirTranslate(
    "hlo-text-to-lhlo", [](llvm::StringRef input, mlir::MLIRContext* context) {
      return mlir::HloTextToLhloTranslateFunction(input, context,
                                                  optimize_xla_hlo);
    });
