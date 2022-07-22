/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/analysis/compatibility_analysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace {

class CompatibilityAnalysis {
 public:
  void AnalyzeOperation(mlir::Operation* op);

  const mlir::tfrt::CompatibilityAnalysisProto& GetResult() const {
    return analysis_;
  }

 private:
  // Return true if some attributes in the op are not supported.
  bool AnalyzeOpAttributes(mlir::Operation* op);
  // Return true if this op has unsupported operation (eg. mutate) on resource
  // variables.
  bool AnalyzeVariable(mlir::Operation* op);

  void UpdateReport(
      const mlir::tfrt::CompatibilityAnalysisReportProto& new_report,
      mlir::tfrt::CompatibilityAnalysisReportProto* old_report);

  mlir::tfrt::CompatibilityAnalysisProto analysis_;
};

void CompatibilityAnalysis::AnalyzeOperation(mlir::Operation* op) {
  // Skip the standard ops that are allowed in tf dialect.
  if (llvm::isa<mlir::func::ReturnOp, mlir::func::FuncOp, mlir::ModuleOp>(op))
    return;

  auto op_name = op->getName();

  std::string name = op_name.getStringRef().str();

  mlir::tfrt::CompatibilityAnalysisReportProto op_report;

  if (op_name.getDialectNamespace() ==
      mlir::TF::TensorFlowDialect::getDialectNamespace()) {
    // Analyze op attributes.
    if (AnalyzeOpAttributes(op)) op_report.set_incompatible_attribute(true);

    // Analyze variable operations.
    if (AnalyzeVariable(op)) op_report.set_incompatible_variable(true);

    // Reference variable is not supported.
    if (op_name.getStringRef() == "tf.VariableV2")
      op_report.set_ref_variable(true);
  } else if (op_name.getDialectNamespace() == "tf_executor") {
    if (llvm::isa<mlir::tf_executor::SwitchOp, mlir::tf_executor::SwitchNOp,
                  mlir::tf_executor::MergeOp, mlir::tf_executor::EnterOp,
                  mlir::tf_executor::NextIterationSourceOp,
                  mlir::tf_executor::NextIterationSinkOp>(op)) {
      op_report.set_control_flow_v1(true);
    } else {
      // Skip the rest of the tf_executor ops as they can be handled.
      //
      // TODO(chky): consider adding allowlist here.
      return;
    }
  } else {
    // Mark unknown dialect in the report.
    op_report.set_unknown_dialect(true);
  }

  auto& op_info = (*analysis_.mutable_ops())[name];
  op_info.set_count(op_info.count() + 1);

  UpdateReport(op_report, op_info.mutable_report());
  UpdateReport(op_report, analysis_.mutable_summary());
}

bool CompatibilityAnalysis::AnalyzeOpAttributes(mlir::Operation* op) {
  // tf.Const gets special handling so it is always compatible.
  if (llvm::isa<mlir::TF::ConstOp>(op)) return false;

  // TODO(chky): Derived attributes should be also analyzed here.
  for (auto attr : op->getAttrs()) {
    if (attr.getName().strref() == "_output_shapes") continue;
    if (attr.getName().strref() == "_class") continue;

    // Symbol attributes (eg. function names) is currently not supported.
    //
    // TODO(chky): CoreRT should ideally support function call operatoins.
    // Remove this condition once that is implemented.
    if (attr.getValue().isa<mlir::FlatSymbolRefAttr>()) return true;

    // Currently only tensors of simple dtypes (i1, i32, i64, f32, f64) are
    // supported.
    if (auto elements_attr = attr.getValue().dyn_cast<mlir::ElementsAttr>()) {
      if (!elements_attr.isa<mlir::DenseElementsAttr>()) return true;
      auto element_type = elements_attr.getType().getElementType();
      if (element_type.isa<mlir::TF::TensorFlowType>()) return true;
    }

    // Currently only arrays of simple element types (i1, i32, i64, f32, f64)
    // are supported.
    if (auto array_attr = attr.getValue().dyn_cast<mlir::ArrayAttr>()) {
      if (!array_attr.empty()) {
        if (array_attr[0].isa<mlir::ElementsAttr>()) return true;

        if (array_attr[0].isa<mlir::StringAttr>()) return true;

        if (array_attr[0].isa<mlir::FlatSymbolRefAttr>()) return true;
      }
    }
  }
  return false;
}

bool CompatibilityAnalysis::AnalyzeVariable(mlir::Operation* op) {
  // Currently only supported variable op is ReadVariableOp.
  if (llvm::isa<mlir::TF::ReadVariableOp>(op)) return false;

  for (auto value : op->getOperands()) {
    auto type = value.getType();
    if (auto tensor_type = type.dyn_cast<mlir::TensorType>()) {
      auto element_type = tensor_type.getElementType();
      if (element_type.isa<mlir::TF::ResourceType>()) return true;
    }
  }

  return false;
}

void CompatibilityAnalysis::UpdateReport(
    const mlir::tfrt::CompatibilityAnalysisReportProto& new_report,
    mlir::tfrt::CompatibilityAnalysisReportProto* old_report) {
  if (new_report.unknown_dialect()) old_report->set_unknown_dialect(true);

  if (new_report.ref_variable()) old_report->set_ref_variable(true);

  if (new_report.incompatible_variable())
    old_report->set_incompatible_variable(true);

  if (new_report.incompatible_attribute())
    old_report->set_incompatible_attribute(true);

  if (new_report.control_flow_v1()) old_report->set_control_flow_v1(true);
}

}  // namespace

mlir::tfrt::CompatibilityAnalysisProto AnalyzeTFCompatibility(
    mlir::ModuleOp op) {
  CompatibilityAnalysis analysis;
  op.walk([&analysis](mlir::Operation* op) { analysis.AnalyzeOperation(op); });
  return analysis.GetResult();
}

static mlir::TranslateFromMLIRRegistration registration(
    "analyze-tf-for-tfrt",
    [](mlir::ModuleOp op, llvm::raw_ostream& output) {
      auto analysis_proto = AnalyzeTFCompatibility(op);
      std::string text_proto;
      if (tensorflow::protobuf::TextFormat::PrintToString(analysis_proto,
                                                          &text_proto)) {
        output << text_proto;
        return mlir::success();
      }

      return mlir::failure();
    },
    [](mlir::DialectRegistry& registry) {
      mlir::RegisterAllTensorFlowDialects(registry);
    });

}  // namespace tensorflow
