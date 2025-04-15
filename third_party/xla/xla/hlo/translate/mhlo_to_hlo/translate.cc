/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/translate/mhlo_to_hlo/translate.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

constexpr char kParameterReplicationAttr[] = "mhlo.parameter_replication";

namespace xla {

mlir::LogicalResult MlirHloToHloTranslateFunction(mlir::ModuleOp module,
                                                  llvm::raw_ostream& output,
                                                  bool emit_return_tuple,
                                                  bool emit_use_tuple_arg) {
  if (!module) return mlir::failure();

  mlir::MlirToHloConversionOptions options;
  options.use_tuple_args = emit_use_tuple_arg;
  options.return_tuple = emit_return_tuple;
  absl::StatusOr<std::unique_ptr<HloModule>> statusOrModule =
      mlir::ConvertMlirHloToHloModule(module, options);

  if (!statusOrModule.ok()) {
    module.emitOpError() << statusOrModule.status().message();
    LOG(ERROR) << "Module conversion failed: " << statusOrModule.status();
    return mlir::failure();
  }

  // Print as HloProto with empty BufferAssignment for legacy compatibility.
  output << MakeHloProto(*statusOrModule.value()).DebugString();
  return mlir::success();
}

absl::StatusOr<std::unique_ptr<HloModule>> HloModuleFromProto(
    const HloProto& hlo_proto) {
  const HloModuleProto& module_proto = hlo_proto.hlo_module();
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          module_proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(module_proto, module_config);
}

// Wraps BuildHloFromMlirHlo to output an HloProto that's the same as
// ConvertMlirHloToHlo.
absl::Status ConvertMlirHloToHloViaBuilder(
    mlir::ModuleOp module, ::xla::HloProto* hlo_proto,
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

  if (auto execution_thread =
          main->getAttrOfType<mlir::StringAttr>("execution_thread")) {
    computation.mutable_proto()->mutable_computations(0)->set_execution_thread(
        execution_thread.str());
  }
  for (int i = 0; i < main.getNumArguments(); ++i)
    if (auto pr = main.getArgAttrOfType<mlir::ArrayAttr>(
            i, kParameterReplicationAttr))
      for (auto b : pr.getValue())
        computation.mutable_proto()
            ->mutable_computations(0)
            ->mutable_instructions(i)
            ->mutable_parameter_replication()
            ->add_replicated_at_leaf_buffers(
                mlir::cast<mlir::BoolAttr>(b).getValue());

  auto hlo_module = computation.proto();
  mlir::StringRef module_name = module.getName() ? *module.getName() : "main";
  hlo_module.set_name(module_name.str());
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);

  return absl::OkStatus();
}

mlir::LogicalResult MlirHloToHloTextTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output, bool emit_return_tuple,
    bool emit_use_tuple_arg, bool print_layouts, bool print_large_constants,
    bool print_sugar, bool via_builder, bool with_layouts,
    bool direct_stablehlo_to_hlo) {
  if (!module) return mlir::failure();

  HloProto hloProto;
  mlir::MlirToHloConversionOptions options;
  options.propagate_layouts = with_layouts;
  options.use_tuple_args = emit_use_tuple_arg;
  options.return_tuple = emit_return_tuple;
  options.direct_stablehlo_to_hlo = direct_stablehlo_to_hlo;
  absl::StatusOr<std::unique_ptr<HloModule>> statusOrHloModule;
  if (via_builder) {
    auto status = ConvertMlirHloToHloViaBuilder(module, &hloProto, options);
    if (!status.ok()) {
      module.emitOpError() << status.message();
      LOG(ERROR) << "Module conversion failed: " << status;
      return mlir::failure();
    }
    statusOrHloModule = HloModuleFromProto(hloProto);
  } else {
    statusOrHloModule = mlir::ConvertMlirHloToHloModule(module, options);
  }

  if (!statusOrHloModule.ok()) {
    module.emitOpError() << statusOrHloModule.status().message();
    LOG(ERROR) << "Conversion to HLO module failed: "
               << statusOrHloModule.status();
    return mlir::failure();
  }

  HloModule* hlo_module = statusOrHloModule.value().get();

  output << hlo_module->ToString(
      HloPrintOptions()
          .set_include_layout_in_shapes(print_layouts)
          .set_syntax_sugar_async_ops(print_sugar)
          .set_print_large_constants(print_large_constants));

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

mlir::LogicalResult MlirHloToHloTextMain(
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    llvm::raw_ostream& output_stream, bool emit_return_tuple,
    bool emit_use_tuple_arg, bool print_layouts, bool print_large_constants,
    bool print_sugar, bool via_builder, bool with_layouts) {
  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  mlir::DialectRegistry registry;
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(*source_mgr, &context);

  if (!module) {
    return mlir::failure();
  }

  return xla::MlirHloToHloTextTranslateFunction(
      *module, output_stream, emit_return_tuple, emit_use_tuple_arg,
      print_layouts, print_large_constants, print_sugar, via_builder,
      with_layouts);
}

}  //  namespace xla
