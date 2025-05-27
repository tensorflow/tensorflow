/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/emitters/kernel_api_builder.h"

#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::emitters {

static constexpr absl::string_view kXlaEntryAttr = "xla.entry";
static constexpr absl::string_view kXlaSliceIndexAttr = "xla.slice_index";
static constexpr absl::string_view kXlaInvariantAttr = "xla.invariant";

absl::StatusOr<mlir::func::FuncOp> EmitKernelApi(
    mlir::ModuleOp module, const HloInstruction& hlo_instruction,
    const BufferAssignment* buffer_assignment,
    const KernelArguments::BufferAlignment& buffer_alignment,
    absl::string_view entry_function_name) {
  mlir::ImplicitLocOpBuilder builder(module.getLoc(), module);
  mlir::MLIRContext* context = builder.getContext();

  // Create the entry function.
  llvm::SmallVector<mlir::Type> param_types;
  std::optional<KernelArguments> args;
  if (buffer_assignment != nullptr) {
    TF_ASSIGN_OR_RETURN(
        args, KernelArguments::Create(*buffer_assignment, buffer_alignment,
                                      &hlo_instruction));
  }
  // Annotate tensors with the buffer indices. This way, the buffer propagation
  // pass can clean them up later.
  auto get_arg_attrs = [&](int index) -> mlir::Attribute {
    if (!args) {
      return builder.getDictionaryAttr({builder.getNamedAttr(
          kXlaSliceIndexAttr, builder.getIndexAttr(index))});
    }

    const auto& arg = args->args()[index];
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr(
        kXlaSliceIndexAttr, builder.getIndexAttr(arg.llvm_arg_index())));
    attrs.push_back(
        builder.getNamedAttr(mlir::LLVM::LLVMDialect::getAlignAttrName(),
                             builder.getIndexAttr(arg.alignment())));
    attrs.push_back(builder.getNamedAttr(
        mlir::LLVM::LLVMDialect::getDereferenceableAttrName(),
        builder.getIndexAttr(arg.slice().size())));
    if (!arg.written()) {
      attrs.push_back(
          builder.getNamedAttr(kXlaInvariantAttr, builder.getUnitAttr()));
    }
    return builder.getDictionaryAttr(attrs);
  };

  auto result_types =
      emitters::ShapeToMlirTypes(hlo_instruction.shape(), builder);

  llvm::SmallVector<mlir::Attribute> arg_attrs;
  arg_attrs.reserve(hlo_instruction.operands().size() + result_types.size());

  for (auto [arg_index, param] : llvm::enumerate(hlo_instruction.operands())) {
    param_types.push_back(
        emitters::TensorShapeToMlirType(param->shape(), builder));
    arg_attrs.push_back(get_arg_attrs(arg_index));
  }

  for (auto [result_index, type] : llvm::enumerate(result_types)) {
    param_types.push_back(type);
    arg_attrs.push_back(
        get_arg_attrs(hlo_instruction.operands().size() + result_index));
  }

  builder.setInsertionPointToStart(module.getBody());
  auto entry_func = builder.create<mlir::func::FuncOp>(
      entry_function_name,
      mlir::FunctionType::get(context, param_types, result_types),
      /*sym_visibility=*/mlir::StringAttr{},
      mlir::ArrayAttr::get(context, arg_attrs),
      /*res_attrs=*/mlir::ArrayAttr{});
  entry_func->setAttr(kXlaEntryAttr, mlir::UnitAttr::get(context));

  return entry_func;
}

}  // namespace xla::emitters
