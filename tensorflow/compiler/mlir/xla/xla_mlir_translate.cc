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

#include "tensorflow/compiler/mlir/xla/xla_mlir_translate.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/mlir/xla/transforms/mhlo_to_lhlo_with_xla.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/mlir/xla/xla_mlir_translate_cl.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

namespace {
// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public tensorflow::protobuf::io::ErrorCollector {
 public:
  void AddError(int line, int column, const string& message) override {}
};

bool LoadHloProto(const std::string& contents, HloProto* hlo_proto) {
  tensorflow::protobuf::TextFormat::Parser parser;
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  return hlo_proto->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto) ||
         hlo_proto->mutable_hlo_module()->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto->mutable_hlo_module());
}

}  // namespace

mlir::OwningModuleRef HloToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations) {
  HloProto hlo_proto;
  string content(input.data(), input.size());
  if (!LoadHloProto(content, &hlo_proto)) {
    LOG(ERROR) << "Failed to load proto";
    return nullptr;
  }

  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  auto status = ConvertHloToMlirHlo(
      module.get(), hlo_proto.mutable_hlo_module(), import_all_computations);
  if (!status.ok()) {
    LOG(ERROR) << "Hlo module import failed: " << status;
    return nullptr;
  }

  return module;
}

mlir::OwningModuleRef HloTextToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations) {
  HloProto hlo_proto;
  string content(input.data(), input.size());

  auto hlo_module_error = ParseAndReturnUnverifiedModule(content);
  if (!hlo_module_error.ok()) {
    LOG(ERROR) << "HLO Module loading failed: " << hlo_module_error.status();
    return nullptr;
  }

  auto hlo_module = std::move(hlo_module_error.ValueOrDie());
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  auto status =
      ConvertHloToMlirHlo(*module, hlo_module.get(), import_all_computations);
  if (!status.ok()) {
    LOG(ERROR) << "HLO Module import failed: " << status;
    return nullptr;
  }

  return module;
}

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
  mlir::FuncOp main = module.lookupSymbol<mlir::FuncOp>("main");
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

  return Status::OK();
}

static mlir::LogicalResult MlirHloToHloTextTranslateFunctionImpl(
    mlir::ModuleOp module, llvm::raw_ostream& output, bool with_layouts,
    bool via_builder) {
  if (!module) return mlir::failure();

  HloProto hloProto;
  mlir::MlirToHloConversionOptions options;
  options.propagate_layouts = with_layouts;
  Status status =
      via_builder
          ? ConvertMlirHloToHloViaBuilder(module, &hloProto, options)
          : mlir::ConvertMlirHloToHlo(
                module, &hloProto, emit_use_tuple_arg, emit_return_tuple,
                /*shape_representation_fn=*/nullptr, options);
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

  HloModule* hlo_module = statusOrHloModule.ValueOrDie().get();

  output << hlo_module->ToString(
      HloPrintOptions().set_include_layout_in_shapes(with_layouts));

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

static mlir::LogicalResult MlirHloToHloTextTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
  return MlirHloToHloTextTranslateFunctionImpl(module, output,
                                               /*with_layouts=*/false,
                                               /*via_builder=*/false);
}

static mlir::LogicalResult MlirHloToHloTextWithLayoutsTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
  return MlirHloToHloTextTranslateFunctionImpl(module, output,
                                               /*with_layouts=*/true,
                                               /*via_builder=*/false);
}

// This converts MlirHlo to Hlo by first converting to XlaBuilder.
// This is useful for testing conversion to XlaBuilder.
static mlir::LogicalResult MlirHloToHloTextViaBuilderTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output) {
  return MlirHloToHloTextTranslateFunctionImpl(module, output,
                                               /*with_layouts=*/false,
                                               /*via_builder=*/true);
}

}  // namespace xla

//----------------------------------------------------------------------------//
// Hooks for tf-mlir-translate
//----------------------------------------------------------------------------/

static llvm::cl::opt<bool> import_all_computations(
    "hlo-import-all-computations",
    llvm::cl::desc("Enable importing unreachable computations."));

static mlir::OwningModuleRef HloToMlirHloTranslate(llvm::StringRef input,
                                                   mlir::MLIRContext* context) {
  return xla::HloToMlirHloTranslateFunction(input, context,
                                            import_all_computations);
}

static mlir::OwningModuleRef HloTextToMlirHloTranslate(
    llvm::StringRef input, mlir::MLIRContext* context) {
  return xla::HloTextToMlirHloTranslateFunction(input, context,
                                                import_all_computations);
}

static void RegisterInputDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::StandardOpsDialect, mlir::mhlo::MhloDialect,
                  mlir::tensor::TensorDialect>();
}

static mlir::TranslateFromMLIRRegistration MlirHloToHloTranslate(
    "mlir-hlo-to-hlo", xla::MlirHloToHloTranslateFunction,
    RegisterInputDialects);

static mlir::TranslateFromMLIRRegistration MlirHloToHloTextTranslate(
    "mlir-hlo-to-hlo-text", xla::MlirHloToHloTextTranslateFunction,
    RegisterInputDialects);

static mlir::TranslateFromMLIRRegistration MlirHloToHloTextWithLayoutsTranslate(
    "mlir-hlo-to-hlo-text-with-layouts",
    xla::MlirHloToHloTextWithLayoutsTranslateFunction, RegisterInputDialects);

static mlir::TranslateFromMLIRRegistration MlirHloToHloTextViaBuilderTranslate(
    "mlir-hlo-to-hlo-text-via-builder",
    xla::MlirHloToHloTextViaBuilderTranslateFunction, RegisterInputDialects);

static mlir::TranslateToMLIRRegistration HloToHloMlirTranslate(
    "hlo-to-mlir-hlo", HloToMlirHloTranslate);

static mlir::TranslateToMLIRRegistration HloTextToHloMlirTranslate(
    "hlo-text-to-mlir-hlo", HloTextToMlirHloTranslate);

// MHLO doesn't support explicit layouts, while XLA service does.
// TODO(timshen): remove it once MHLO supports explicit layouts.
static mlir::TranslateToMLIRRegistration HloTextToLhloMlirTranslate(
    "hlo-text-to-lhlo", mlir::HloTextToLhloTranslateFunction);
