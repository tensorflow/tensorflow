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

#include "xla/translate/hlo_to_mhlo/hlo_module_importer.h"

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/status.h"
#include "xla/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

HloModuleImporter::HloModuleImporter(mlir::ModuleOp module,
                                     bool import_all_computation)
    : import_all_computation_(import_all_computation),
      symbol_table_(module),
      builder_(module.getContext()) {
  module.getContext()->loadDialect<mlir::arith::ArithDialect>();
  module.getContext()->loadDialect<mlir::func::FuncDialect>();
  module.getContext()->loadDialect<mlir::mhlo::MhloDialect>();
  module.getContext()->loadDialect<mlir::quant::QuantizationDialect>();
  module.getContext()->loadDialect<mlir::quant::QuantizationDialect>();
}

namespace {

constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";

mlir::ArrayAttr ConvertCrossProgramPrefetches(
    const absl::Span<const HloModule::CrossProgramPrefetchInfo> prefetches,
    mlir::Builder* builder) {
  llvm::SmallVector<mlir::Attribute, 4> shapes;
  for (auto [parameter, index, alt_memory_offset] : prefetches) {
    llvm::SmallVector<int64_t, 4> dims;
    for (auto dim : index) dims.push_back(dim);
    std::optional<int64_t> offset =
        alt_memory_offset ? std::optional<int64_t>(*alt_memory_offset)
                          : std::nullopt;
    shapes.push_back(mlir::mhlo::CrossProgramPrefetchAttr::get(
        builder->getContext(), parameter, dims, offset));
  }

  return mlir::ArrayAttr::get(builder->getContext(), shapes);
}
}  // namespace

Status HloModuleImporter::Import(const HloModule& hlo_module) {
  auto module = llvm::cast<mlir::ModuleOp>(symbol_table_.getOp());
  module.setName(hlo_module.name());
  module->setAttr("mhlo.cross_program_prefetches",
                  ConvertCrossProgramPrefetches(
                      hlo_module.CrossProgramPrefetches(), &builder_));
  module->setAttr(
      "mhlo.is_dynamic",
      mlir::BoolAttr::get(builder_.getContext(), hlo_module.is_dynamic()));
  ImportFrontendAttributes(hlo_module, module);
  module->setAttr("mhlo.use_auto_spmd_partitioning",
                  mlir::BoolAttr::get(builder_.getContext(),
                                      hlo_module.use_auto_spmd_partitioning()));
  if (hlo_module.has_spmd_output_sharding()) {
    module->setAttr(
        "mhlo.spmd_output_sharding",
        ConvertSharding(hlo_module.spmd_output_sharding(), &builder_));
  }

  if (hlo_module.has_spmd_parameters_shardings()) {
    llvm::SmallVector<mlir::Attribute> parameter_shardings;
    for (const auto& sharding : hlo_module.spmd_parameters_shardings()) {
      parameter_shardings.push_back(ConvertSharding(sharding, &builder_));
    }
    module->setAttr("mhlo.spmd_parameters_shardings",
                    builder_.getArrayAttr(parameter_shardings));
  }

  if (!import_all_computation_)
    // Only import the entry computation, any reachable one will be imported
    // unless turned into a region operation.
    return HloFunctionImporter::ImportAsFunc(*hlo_module.entry_computation(),
                                             symbol_table_, &function_map_,
                                             &builder_,
                                             /*is_main*/ true)
        .status();

  auto* module_entry_computation = hlo_module.entry_computation();
  for (const auto* computation : hlo_module.computations())
    TF_RETURN_IF_ERROR(HloFunctionImporter::ImportAsFunc(
                           *computation, symbol_table_, &function_map_,
                           &builder_,
                           /*is_main*/ computation == module_entry_computation)
                           .status());

  return OkStatus();
}

Status HloModuleImporter::Import(const HloModuleProto& module_proto) {
  DebugOptions debug_options;
  TF_ASSIGN_OR_RETURN(
      auto module_config,
      HloModule::CreateModuleConfigFromProto(module_proto, debug_options));
  TF_ASSIGN_OR_RETURN(auto module,
                      HloModule::CreateFromProto(module_proto, module_config));

  return Import(*module);
}

void HloModuleImporter::ImportFrontendAttributes(const HloModule& hlo_module,
                                                 mlir::ModuleOp module) {
  if (!hlo_module.frontend_attributes().map().empty()) {
    llvm::SmallVector<mlir::NamedAttribute, 4> frontend_attributes;
    for (const auto& [k, v] : hlo_module.frontend_attributes().map()) {
      frontend_attributes.push_back(
          builder_.getNamedAttr(k, builder_.getStringAttr(v)));
    }
    if (!frontend_attributes.empty()) {
      module->setAttr(kFrontendAttributesAttr,
                      builder_.getDictionaryAttr(frontend_attributes));
    }
  }
}

}  // namespace xla
