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

#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_module_importer.h"

#include <iterator>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "tensorflow/compiler/xla/xla.pb.h"

namespace xla {

HloModuleImporter::HloModuleImporter(mlir::ModuleOp module,
                                     bool import_all_computation)
    : import_all_computation_(import_all_computation),
      module_(module),
      builder_(module.getContext()) {
  module.getContext()->loadDialect<mlir::arith::ArithDialect>();
  module.getContext()->loadDialect<mlir::func::FuncDialect>();
  module.getContext()->loadDialect<mlir::mhlo::MhloDialect>();
}

namespace {

// Inserts reshapes and transposes for parameters that have non-default layouts.
Status NormalizeEntryComputationLayout(xla::HloModule* module) {
  auto* computation = module->entry_computation();
  for (int i = 0; i < computation->num_parameters(); ++i) {
    auto* param = computation->parameter_instruction(i);
    const auto& shape =
        module->entry_computation_layout().parameter_layout(i).shape();
    if (!shape.has_layout()) continue;
    const auto& layout = shape.layout();
    // This code can't handle tiles. For default shapes, there's nothing to do.
    if (!layout.tiles().empty() ||
        layout == LayoutUtil::GetDefaultLayoutForShape(shape)) {
      continue;
    }

    // Reshape the parameter into the shape that's actually passed.
    int64_t rank = shape.rank();
    std::vector<int64_t> permutation(rank);
    for (int64_t dim = 0; dim < rank; ++dim) {
      permutation[dim] = rank - 1ll - layout.minor_to_major(dim);
    }

    auto physical_shape = ShapeUtil::MakeShapeWithDescendingLayout(
        shape.element_type(), Permute(shape.dimensions(), permutation));
    auto* reshape = computation->AddInstruction(
        HloInstruction::CreateReshape(physical_shape, param));

    // Transpose the physical shape back to the logical shape.
    auto* transpose =
        computation->AddInstruction(HloInstruction::CreateTranspose(
            shape, reshape, InversePermutation(permutation)));

    std::vector<HloInstruction*> users_to_replace;
    absl::c_remove_copy(param->users(), std::back_inserter(users_to_replace),
                        reshape);

    TF_RETURN_IF_ERROR(param->ReplaceUsesWith(users_to_replace, transpose));
  }
  return ::tsl::OkStatus();
}

}  // namespace

Status HloModuleImporter::Import(const xla::HloModule& module) {
  std::unique_ptr<HloModule> module_copy;
  const HloModule* module_to_convert = &module;
  if (absl::c_any_of(module.entry_computation_layout().parameter_layouts(),
                     [&](const ShapeLayout& layout) {
                       if (!layout.shape().has_layout()) return false;
                       // This code can't handle tiles.
                       return layout.layout().tiles().empty() &&
                              layout.layout() !=
                                  LayoutUtil::GetDefaultLayoutForShape(
                                      layout.shape());
                     })) {
    module_copy = module.Clone(/*suffix=*/"");
    TF_RETURN_IF_ERROR(NormalizeEntryComputationLayout(module_copy.get()));
    module_to_convert = module_copy.get();
  }

  module_.setName(module.name());
  if (!import_all_computation_)
    // Only import the entry computation, any reachable one will be imported
    // unless turned into a region operation.
    return HloFunctionImporter::ImportAsFunc(
        *module_to_convert->entry_computation(), module_, &function_map_,
        &builder_, /*is_main*/ true);

  auto* module_entry_computation = module_to_convert->entry_computation();
  for (const auto* computation : module_to_convert->computations())
    TF_RETURN_IF_ERROR(HloFunctionImporter::ImportAsFunc(
        *computation, module_, &function_map_, &builder_,
        /*is_main*/ computation == module_entry_computation));

  return OkStatus();
}

Status HloModuleImporter::Import(const xla::HloModuleProto& module_proto) {
  xla::DebugOptions debug_options;
  TF_ASSIGN_OR_RETURN(
      auto module_config,
      xla::HloModule::CreateModuleConfigFromProto(module_proto, debug_options));
  TF_ASSIGN_OR_RETURN(auto module, xla::HloModule::CreateFromProto(
                                       module_proto, module_config));

  return Import(*module);
}

}  // namespace xla
