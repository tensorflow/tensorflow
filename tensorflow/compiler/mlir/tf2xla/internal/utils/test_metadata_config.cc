/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/utils/test_metadata_config.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/deserialize_mlir_module_utils.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {
namespace {

constexpr char kEntryFuncName[] = "main";

absl::Status SetupArguments(mlir::ModuleOp module,
                            std::vector<TensorShape>& arg_shapes,
                            tpu::TPUCompileMetadataProto& metadata_proto) {
  auto main_fn = module.lookupSymbol<mlir::func::FuncOp>(kEntryFuncName);
  if (!main_fn) {
    return absl::InternalError("Could not find main function in MLIR Module.");
  }

  mlir::FunctionType func_type = main_fn.getFunctionType();
  for (auto input_type : func_type.getInputs()) {
    tensorflow::TensorShape tensor_shape;
    xla::Shape xla_shape = xla::TypeToShape(input_type);
    TF_RETURN_IF_ERROR(tensorflow::TensorShape::BuildTensorShape(
        xla_shape.dimensions(), &tensor_shape));
    arg_shapes.emplace_back(tensor_shape);

    DataType dtype;
    TF_RETURN_IF_ERROR(ConvertToDataType(input_type, &dtype));

    auto metadata_arg = metadata_proto.add_args();
    metadata_arg->set_kind(tpu::TPUCompileMetadataProto::Arg::PARAMETER);
    metadata_arg->set_dtype(dtype);
  }

  return absl::OkStatus();
}

absl::Status SetupReturnValues(mlir::ModuleOp module,
                               tpu::TPUCompileMetadataProto& metadata_proto) {
  auto main_fn = module.lookupSymbol<mlir::func::FuncOp>(kEntryFuncName);
  if (!main_fn) {
    return absl::InternalError("Could not find main function in MLIR Module.");
  }

  int func_results = main_fn.getFunctionType().getNumResults();
  for (int i = 0; i < func_results; i++) {
    metadata_proto.add_retvals();
  }

  return absl::OkStatus();
}

}  // namespace

absl::Status ConfigureMetadata(absl::string_view mlir_module_str,
                               std::vector<TensorShape>& arg_shapes,
                               tpu::TPUCompileMetadataProto& metadata_proto) {
  mlir::DialectRegistry registry;
  mlir::RegisterCommonToolingDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;

  TF_RETURN_IF_ERROR(
      DeserializeMlirModule(mlir_module_str, &context, &mlir_module));
  TF_RETURN_IF_ERROR(SetupReturnValues(*mlir_module, metadata_proto));
  TF_RETURN_IF_ERROR(SetupArguments(*mlir_module, arg_shapes, metadata_proto));

  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
