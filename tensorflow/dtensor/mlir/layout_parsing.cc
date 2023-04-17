/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/layout_parsing.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {
namespace {

bool OpUsesV2LayoutAnnotation(mlir::Operation* op) {
  return !op->getUsers().empty() &&
         llvm::all_of(op->getUsers(), [](mlir::Operation* user_op) {
           return llvm::isa<mlir::TF::DTensorLayout>(user_op);
         });
}

}  // namespace

StatusOr<absl::optional<Layout>> ExtractSingleLayoutFromOp(
    mlir::Operation* op, std::string attr_name) {
  absl::optional<Layout> out;

  // If v2 layout propagation algorithm is used, parse layout from DTensorLayout
  // op.
  if (OpUsesV2LayoutAnnotation(op)) {
    // If DTensorLayout is used, then DTensorLayout op is the only consumer for
    // the operation output value.
    auto users = op->getUsers();
    out.emplace(
        llvm::cast<mlir::TF::DTensorLayout>(*users.begin()).getLayout());
  } else {
    TF_ASSIGN_OR_RETURN(auto layouts, ExtractLayoutFromOp(op, attr_name));
    if (layouts.empty()) return out;
    if (layouts.size() != 1) {
      return errors::Internal(
          "Extracting single layout on Op that has multiple layout attached is "
          "ambiguous. op : ",
          op->getName().getStringRef().str());
    }
    out.swap(layouts[0]);
  }
  return out;
}

StatusOr<absl::optional<Layout>> ExtractSingleLayoutFromOp(
    mlir::Operation* op) {
  return ExtractSingleLayoutFromOp(op, kLayoutAttr);
}

StatusOr<Layout> ExtractRequiredSingleLayoutFromOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> layout,
                      ExtractSingleLayoutFromOp(op));
  if (!layout) return errors::Internal("expected layout missing");

  return *layout;
}

StatusOr<std::vector<absl::optional<Layout>>> ExtractLayoutFromOp(
    mlir::Operation* op, std::string attr_name) {
  std::vector<absl::optional<Layout>> outs;
  outs.reserve(op->getNumResults());

  // If v2 layout propagation algorithm is used, parse layout from DTensorLayout
  // op.
  if (OpUsesV2LayoutAnnotation(op)) {
    for (auto op_result : op->getOpResults()) {
      outs.emplace_back(
          llvm::cast<mlir::TF::DTensorLayout>(*op_result.getUsers().begin())
              .getLayout());
    }
  } else {
    auto serialized_layouts = op->getAttrOfType<mlir::ArrayAttr>(attr_name);
    if (!serialized_layouts) return outs;

    for (auto const& attr : serialized_layouts) {
      auto attr_str = attr.cast<mlir::StringAttr>().getValue().str();
      if (!attr_str.empty()) {
        TF_ASSIGN_OR_RETURN(auto layout, Layout::FromString(attr_str));
        outs.emplace_back(std::move(layout));
      } else {
        outs.emplace_back(absl::nullopt);
      }
    }
  }
  return outs;
}

StatusOr<std::vector<absl::optional<Layout>>> ExtractLayoutFromOp(
    mlir::Operation* op) {
  return ExtractLayoutFromOp(op, kLayoutAttr);
}

StatusOr<std::vector<Layout>> ExtractRequiredLayoutFromOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(std::vector<absl::optional<Layout>> optional_layouts,
                      ExtractLayoutFromOp(op));
  std::vector<Layout> layouts;
  for (const absl::optional<Layout>& layout : optional_layouts) {
    if (!layout) return errors::Internal("expected layout missing");
    layouts.emplace_back(*layout);
  }

  return layouts;
}

StatusOr<Mesh> ExtractDeviceMeshEnclosingCluster(mlir::Operation* op) {
  auto enclosing_cluster = op->getParentOfType<mlir::tf_device::ClusterOp>();
  if (!enclosing_cluster)
    return errors::InvalidArgument("op is not inside a device mesh cluster.");

  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshFromOp(enclosing_cluster));
  if (!mesh)
    return errors::InvalidArgument(
        "op's enclosing device cluster does not have mesh defined.");

  return *mesh;
}

StatusOr<absl::optional<Mesh>> ExtractDeviceMeshFromOp(mlir::Operation* op) {
  absl::optional<Mesh> extracted_mesh;
  if (op == nullptr) return extracted_mesh;

  auto mesh_str_attr = op->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  if (!mesh_str_attr) return extracted_mesh;

  TF_ASSIGN_OR_RETURN(Mesh mesh,
                      Mesh::FromString(mesh_str_attr.getValue().str()));

  extracted_mesh.emplace(std::move(mesh));
  return extracted_mesh;
}

StatusOr<absl::optional<Layout>> ExtractLayoutFromOperand(mlir::Value operand) {
  if (auto op_result = operand.dyn_cast<mlir::OpResult>()) {
    mlir::Operation* op = op_result.getDefiningOp();
    absl::optional<Layout> out;
    if (auto layout_op = llvm::dyn_cast<mlir::TF::DTensorLayout>(op)) {
      out.emplace(layout_op.getLayout());
    } else {
      const int result_number = op_result.getResultNumber();
      TF_ASSIGN_OR_RETURN(auto layouts, ExtractLayoutFromOp(op, kLayoutAttr));

      if (layouts.empty()) return out;

      if (result_number >= layouts.size()) {
        return errors::Internal(
            "Expect to extract the ", result_number,
            "-th output's layout, but "
            "only see ",
            layouts.size(), " outputs: ", op->getName().getStringRef().str());
      }
      out.swap(layouts[result_number]);
    }
    return out;
  }

  auto block_arg = operand.dyn_cast<mlir::BlockArgument>();
  if (!block_arg)
    return errors::Internal(
        "Operand is not either a OpResult or a BlockArgument. This should not "
        "happen.");
  auto func_op = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      block_arg.getOwner()->getParentOp());
  if (!func_op) {
    return errors::InvalidArgument("op must be enclosed by a function");
  }

  absl::optional<Layout> extracted_layout;
  auto layout_attr = func_op.getArgAttrOfType<mlir::StringAttr>(
      block_arg.getArgNumber(), kCustomDeviceAttr);
  if (!layout_attr) return extracted_layout;

  TF_ASSIGN_OR_RETURN(auto layout,
                      Layout::FromString(layout_attr.getValue().str()));
  extracted_layout.emplace(std::move(layout));
  return extracted_layout;
}

StatusOr<Layout> ExtractRequiredLayoutFromOperand(mlir::Value operand) {
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> layout,
                      ExtractLayoutFromOperand(operand));
  if (!layout) return errors::Internal("expected layout missing");

  return *layout;
}

StatusOr<std::vector<Layout>> ExtractRequiredLayoutFromOperands(
    mlir::Operation* op) {
  std::vector<Layout> layouts;
  for (const auto& operand : op->getOpOperands()) {
    TF_ASSIGN_OR_RETURN(auto operand_layout,
                        ExtractRequiredLayoutFromOperand(operand.get()));
    layouts.emplace_back(operand_layout);
  }
  return layouts;
}

void SetLayoutOnOp(mlir::Operation* op, mlir::OpBuilder builder,
                   absl::Span<const absl::optional<Layout>> layouts) {
  llvm::SmallVector<std::string, 8> serialized_layouts;
  for (auto const& layout : layouts) {
    serialized_layouts.emplace_back(layout.has_value() ? layout->ToString()
                                                       : "");
  }
  op->setAttr(kLayoutAttr,
              builder.getStrArrayAttr(llvm::SmallVector<llvm::StringRef, 8>(
                  serialized_layouts.begin(), serialized_layouts.end())));
}

void SetLayoutOnOp(mlir::Operation* op,
                   absl::Span<const absl::optional<Layout>> layouts) {
  SetLayoutOnOp(op, mlir::OpBuilder(op), layouts);
}

void SetSingleLayoutOnOp(mlir::Operation* op, const Layout& layout) {
  SetLayoutOnOp(op, mlir::OpBuilder(op), {absl::optional<Layout>(layout)});
}

StatusOr<absl::optional<Layout>> ExtractLayoutFromFunctionReturnAttr(
    mlir::func::ReturnOp return_op, const int return_index) {
  absl::optional<Layout> layout;
  // If value feeds into func op return op, then check to see if layout
  // attribute is set for the return value.
  auto function = return_op->getParentOfType<mlir::func::FuncOp>();
  auto layout_attr_from_func_result =
      function.getResultAttrOfType<mlir::StringAttr>(return_index,
                                                     kCustomDefaultLayoutAttr);
  if (!layout_attr_from_func_result) return layout;

  const std::string layout_string =
      layout_attr_from_func_result.getValue().str();
  auto result_layout_or_status = Layout::FromString(layout_string);
  if (!result_layout_or_status.ok())
    return errors::InvalidArgument(
        llvm::formatv("Malformed default return layout received. {0} Received "
                      "layout : {1}",
                      result_layout_or_status.status().message(), layout_string)
            .str());

  layout.emplace(result_layout_or_status.value());
  return layout;
}

StatusOr<llvm::SmallVector<Layout, 4>> ExtractElementLayoutsFromOperand(
    mlir::OpOperand& input_value) {
  const int operand_index = input_value.getOperandNumber();
  auto defining_op = input_value.get().getDefiningOp();

  if (defining_op) {
    if (mlir::isa<mlir::TF::DTensorLayout,
                  mlir::TF::IteratorGetNextAsOptionalOp>(defining_op)) {
      return ExtractElementLayoutsFromOperand(defining_op->getOpOperand(0));
    }
  }

  // If we reach this point, we're working with a function argument.
  mlir::Operation* op = input_value.getOwner();
  auto enclosing_function = op->getParentOfType<mlir::func::FuncOp>();
  if (!enclosing_function)
    return errors::InvalidArgument(
        llvm::formatv("Could not find iterator at {0}-th input to op: {1}",
                      operand_index, op->getName())
            .str());

  auto block_arg = input_value.get().dyn_cast<mlir::BlockArgument>();
  auto array_attr = enclosing_function.getArgAttrOfType<mlir::ArrayAttr>(
      block_arg.getArgNumber(), kIteratorElementLayouts);
  if (!array_attr)
    return errors::InvalidArgument(
        llvm::formatv(
            "Could not find `{0}` attribute of {1}-th input to op: {2}",
            kIteratorElementLayouts, operand_index, op->getName())
            .str());

  llvm::SmallVector<Layout, 4> layouts(array_attr.size());
  for (int i = 0; i < array_attr.size(); ++i) {
    layouts[i] = Layout::FromString(
                     array_attr[i].cast<mlir::StringAttr>().getValue().str())
                     .value();
  }

  return layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
