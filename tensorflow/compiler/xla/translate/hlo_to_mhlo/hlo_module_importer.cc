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

// Checks if the shape 1) has a non-default layout 2) without tiles.
bool LayoutRequiresNormalization(const Shape& shape) {
  if (shape.IsTuple()) {
    return absl::c_any_of(shape.tuple_shapes(), LayoutRequiresNormalization);
  }
  if (!shape.has_layout()) return false;

  // This code can't handle tiles.
  if (!shape.layout().tiles().empty()) return false;

  // For default shapes, there's nothing to do.
  return shape.layout() != LayoutUtil::GetDefaultLayoutForShape(shape);
}

bool EntryComputationLayoutCanBeNormalized(const HloModule& module) {
  // This code only supports normalization of entry_computation_layout, not of
  // any other instructions. In the MLIR pipeline, no instructions will ever
  // have layouts.
  for (const auto* instruction : module.entry_computation()->instructions()) {
    if (LayoutRequiresNormalization(instruction->shape())) return false;
  }

  return LayoutRequiresNormalization(
             module.entry_computation_layout().result_layout().shape()) ||
         absl::c_any_of(
             module.entry_computation_layout().parameter_layouts(),
             [](const auto& shape_layout) {
               return LayoutRequiresNormalization(shape_layout.shape());
             });
}

// For a tensor with a non-default layout, inserts a reshape and a transpose to
// convert from the non-default layout to the default.
HloInstruction* NormalizeTensor(HloInstruction* tensor, const Shape& shape,
                                bool is_input) {
  // Reshape the parameter into the shape that's actually passed.
  int64_t rank = shape.rank();
  std::vector<int64_t> permutation(rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    permutation[dim] = rank - 1ll - shape.layout().minor_to_major(dim);
  }
  auto inverse_permutation = InversePermutation(permutation);

  auto physical_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      shape.element_type(),
      Permute(shape.dimensions(),
              is_input ? permutation : inverse_permutation));

  auto* computation = tensor->parent();
  if (is_input) {
    auto* reshape = computation->AddInstruction(
        HloInstruction::CreateReshape(physical_shape, tensor));
    return computation->AddInstruction(
        HloInstruction::CreateTranspose(shape, reshape, inverse_permutation));
  }
  auto* transpose = computation->AddInstruction(HloInstruction::CreateTranspose(
      physical_shape, tensor, inverse_permutation));
  return computation->AddInstruction(
      HloInstruction::CreateReshape(shape, transpose));
}

HloInstruction* NormalizeValue(HloInstruction* value, const Shape& shape,
                               bool is_input) {
  if (!LayoutRequiresNormalization(shape)) return value;

  if (!shape.IsTuple()) {
    return NormalizeTensor(value, shape, is_input);
  }

  std::vector<HloInstruction*> elements;
  for (int64_t i = 0; i < shape.tuple_shapes_size(); ++i) {
    auto* element = value->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(value, i));
    elements.push_back(
        NormalizeValue(element, shape.tuple_shapes(i), is_input));
  }
  return value->parent()->AddInstruction(HloInstruction::CreateTuple(elements));
}

// Inserts reshapes and transposes for entry computation parameters and results
// that have non-default layouts.
Status NormalizeEntryComputationLayout(xla::HloModule* module) {
  auto* computation = module->entry_computation();
  const auto& computation_layout = module->entry_computation_layout();
  for (int i = 0; i < computation->num_parameters(); ++i) {
    auto* param = computation->parameter_instruction(i);
    const auto& shape = computation_layout.parameter_layout(i).shape();
    std::vector<HloInstruction*> users = param->users();
    auto* normalized = NormalizeValue(param, shape, /*is_input=*/true);
    TF_RETURN_IF_ERROR(param->ReplaceUsesWith(users, normalized));
  }

  const auto& result_shape = computation_layout.result_layout().shape();
  auto* normalized_result = NormalizeValue(computation->root_instruction(),
                                           result_shape, /*is_input=*/false);
  computation->set_root_instruction(normalized_result);

  return ::tsl::OkStatus();
}

}  // namespace

Status HloModuleImporter::Import(const xla::HloModule& module) {
  std::unique_ptr<HloModule> module_copy;
  const HloModule* module_to_convert = &module;
  if (EntryComputationLayoutCanBeNormalized(module)) {
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
  for (const auto* computation : module_to_convert->computations()) {
    TF_RETURN_IF_ERROR(HloFunctionImporter::ImportAsFunc(
        *computation, module_, &function_map_, &builder_,
        /*is_main*/ computation == module_entry_computation));
  }
  return ::tsl::OkStatus();
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
