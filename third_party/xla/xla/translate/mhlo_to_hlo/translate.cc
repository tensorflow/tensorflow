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

#include "xla/translate/mhlo_to_hlo/translate.h"

#include <memory>

#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/service/hlo.pb.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/translate/mhlo_to_hlo/type_to_shape.h"

constexpr char kParameterReplicationAttr[] = "mhlo.parameter_replication";

namespace xla {

mlir::LogicalResult MlirHloToHloTranslateFunction(mlir::ModuleOp module,
                                                  llvm::raw_ostream& output,
                                                  bool emit_return_tuple,
                                                  bool emit_use_tuple_arg) {
  if (!module) return mlir::failure();

  HloProto hloProto;
  Status status = mlir::ConvertMlirHloToHlo(
      module, &hloProto, emit_use_tuple_arg, emit_return_tuple);
  if (!status.ok()) {
    module.emitOpError() << status.message();
    LOG(ERROR) << "Module conversion failed: " << status;
    return mlir::failure();
  }

  output << hloProto.DebugString();
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
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);

  return OkStatus();
}

mlir::LogicalResult MlirHloToHloTextTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output, bool emit_return_tuple,
    bool emit_use_tuple_arg, bool print_layouts, bool print_large_constants,
    bool print_sugar, bool via_builder, bool with_layouts) {
  if (!module) return mlir::failure();

  HloProto hloProto;
  mlir::MlirToHloConversionOptions options;
  options.propagate_layouts = with_layouts;
  Status status =
      via_builder
          ? ConvertMlirHloToHloViaBuilder(module, &hloProto, options)
          : mlir::ConvertMlirHloToHlo(module, &hloProto, emit_use_tuple_arg,
                                      emit_return_tuple, options);
  if (!status.ok()) {
    module.emitOpError() << status.message();
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

}  // namespace xla
