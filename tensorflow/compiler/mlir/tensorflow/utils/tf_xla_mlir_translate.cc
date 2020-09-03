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

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate_cl.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/utils/string_container_utils.h"
#include "tensorflow/compiler/mlir/xla/xla_mlir_translate_cl.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

namespace {

mlir::LogicalResult PrintHloModuleText(
    const XlaCompilationResult& compilation_result, llvm::raw_ostream& output) {
  const xla::HloModuleConfig module_config(
      compilation_result.computation->GetProgramShape().ValueOrDie());
  auto status_or_hlo_module = xla::HloModule::CreateFromProto(
      compilation_result.computation->proto(), module_config);
  if (!status_or_hlo_module.ok()) {
    LOG(ERROR) << "Conversion to HLO module failed: "
               << status_or_hlo_module.status().error_message();
    return mlir::failure();
  }

  xla::HloModule* hlo_module = status_or_hlo_module.ValueOrDie().get();

  output << hlo_module->ToString();

  if (!compilation_result.input_mapping.empty())
    output << "// InputMapping {"
           << absl::StrJoin(compilation_result.input_mapping, ", ") << "}\n";

  for (const auto& xla_input_shape : compilation_result.xla_input_shapes)
    output << "// XlaInputShape " << xla_input_shape.ToString() << '\n';

  output << "// XlaOutputShape "
         << compilation_result.xla_output_shape.ToString() << '\n';

  for (const auto& xla_output_description : compilation_result.outputs) {
    output << "// XlaOutputDescription type="
           << DataTypeString(xla_output_description.type) << " shape=("
           << absl::StrJoin(xla_output_description.shape.dim_sizes(), ", ")
           << ')';
    if (xla_output_description.input_index >= 0)
      output << " input_index=" << xla_output_description.input_index;
    if (xla_output_description.is_constant) output << " constant";
    if (xla_output_description.is_tensor_list) output << " tensor_list";
    output << '\n';
  }

  for (const auto& resource_update : compilation_result.resource_updates) {
    output << "// ResourceUpdate input_index=" << resource_update.input_index
           << " type=" << DataTypeString(resource_update.type) << " shape=("
           << absl::StrJoin(resource_update.shape.dim_sizes(), " ") << ')';
    if (resource_update.modified) output << " modified";
    output << '\n';
  }

  return mlir::success();
}

Status ParseArgumentShapes(
    absl::string_view input_shapes_str,
    llvm::SmallVectorImpl<TensorOrResourceShape>& arg_shapes) {
  arg_shapes.clear();
  std::vector<std::vector<int>> input_shapes_vector;
  TF_RETURN_IF_ERROR(ParseNodeShapes(input_shapes_str, input_shapes_vector));
  arg_shapes.resize(input_shapes_vector.size());
  for (const auto& shape : llvm::enumerate(input_shapes_vector))
    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(
        shape.value(), &arg_shapes[shape.index()].shape));

  return Status::OK();
}

}  // anonymous namespace

static mlir::LogicalResult MlirTfToHloTextTranslateFunction(
    mlir::ModuleOp module_op, llvm::raw_ostream& output) {
  if (!module_op) return mlir::failure();

  llvm::SmallVector<TensorOrResourceShape, 4> arg_shapes;
  auto args_status =
      ParseArgumentShapes(mlir::StringRefToView(input_shapes), arg_shapes);
  if (!args_status.ok()) {
    LOG(ERROR) << args_status.error_message();
    return mlir::failure();
  }

  XlaCompilationResult compilation_result;
  auto compilation_status = CompileMlirToXlaHlo(
      module_op, arg_shapes, "XLA_CPU_JIT", emit_use_tuple_arg,
      emit_return_tuple, IdentityShapeRepresentationFn(), &compilation_result,
      /*custom_legalization_passes=*/{});
  if (!compilation_status.ok()) {
    LOG(ERROR) << "TF/XLA compilation failed: "
               << compilation_status.error_message();
    return mlir::failure();
  }

  return PrintHloModuleText(compilation_result, output);
}

}  // namespace tensorflow

static void RegisterInputDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::StandardOpsDialect, mlir::TF::TensorFlowDialect>();
}

static mlir::TranslateFromMLIRRegistration MlirTfXlaToHloTextTranslate(
    "mlir-tf-to-hlo-text", tensorflow::MlirTfToHloTextTranslateFunction,
    RegisterInputDialects);
