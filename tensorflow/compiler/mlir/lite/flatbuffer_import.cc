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

#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"

#include <iostream>
#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Diagnostics.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/Support/FileUtilities.h"  // TF:local_config_mlir
#include "mlir/Translation.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

using mlir::Builder;
using mlir::FuncOp;
using mlir::Location;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::OwningModuleRef;
using tflite::TensorT;
using xla::StatusOr;

namespace errors = tensorflow::errors;

namespace {
bool IsScalar(const TensorT& tensor) {
  // TODO(krzysd): We can't distinguish scalars and unranked tensors
  // Work out a way to handle this and stub out the code until then
  return tensor.shape.empty() && false;
}

StatusOr<mlir::Type> GetTensorElementType(const TensorT& tensor,
                                          Builder builder) {
  switch (tensor.type) {
    case tflite::TensorType_FLOAT32:
      return builder.getF32Type();
    case tflite::TensorType_FLOAT16:
      return builder.getF16Type();
    case tflite::TensorType_INT32:
      return builder.getIntegerType(32);
    case tflite::TensorType_UINT8:
      return builder.getIntegerType(8);
    case tflite::TensorType_INT64:
      return builder.getIntegerType(64);
    case tflite::TensorType_STRING:
      return errors::InvalidArgument("String tensors are not supported");
    case tflite::TensorType_BOOL:
      return builder.getI1Type();
    case tflite::TensorType_INT16:
      return builder.getIntegerType(16);
    case tflite::TensorType_COMPLEX64:
      return mlir::ComplexType::get(builder.getF32Type());
    case tflite::TensorType_INT8:
      return builder.getIntegerType(8);
  }
  return errors::OutOfRange("Unknown tensor type");
}

StatusOr<mlir::Type> GetTensorType(const TensorT& tensor, Builder builder) {
  TF_ASSIGN_OR_RETURN(auto elem_type, GetTensorElementType(tensor, builder));
  if (IsScalar(tensor)) {
    return builder.getTensorType({}, elem_type);
  }

  if (!tensor.shape.empty()) {
    llvm::SmallVector<int64_t, 4> shape;
    for (int32_t i : tensor.shape) {
      shape.push_back(int64_t{i});
    }
    return builder.getTensorType(shape, elem_type);
  }

  return builder.getTensorType(elem_type);
}

}  // namespace

OwningModuleRef tflite::FlatBufferToMlir(absl::string_view buffer,
                                         MLIRContext* context,
                                         Location base_loc) {
  auto model_ptr =
      FlatBufferModel::VerifyAndBuildFromBuffer(buffer.data(), buffer.length());
  if (nullptr == model_ptr) {
    return emitError(base_loc, "couldn't parse flatbuffer"), nullptr;
  }

  std::unique_ptr<ModelT> model(model_ptr->GetModel()->UnPack());

  auto builder = Builder(context);
  auto module = mlir::ModuleOp::create(base_loc);

  // TODO(krzysd): Actually account for the FlatBuffer schema version
  module.setAttr("tfl.schema_version",
                 builder.getI32IntegerAttr(model->version));

  for (auto& subgraph : model->subgraphs) {
    llvm::SmallVector<mlir::Type, 2> ret_types;
    llvm::SmallVector<mlir::Type, 4> input_types;

    for (auto input : subgraph->inputs) {
      auto type_or_err = GetTensorType(*subgraph->tensors[input], builder);
      if (!type_or_err.ok()) {
        return emitError(base_loc, type_or_err.status().ToString()), nullptr;
      }
      input_types.push_back(type_or_err.ConsumeValueOrDie());
    }

    auto func_type = builder.getFunctionType(input_types, ret_types);
    auto func_loc = mlir::NameLoc::get(builder.getIdentifier(subgraph->name),
                                       base_loc, context);
    auto func =
        FuncOp::create(func_loc, subgraph->name, func_type, /* attrs= */ {});
    func.addEntryBlock();

    // TODO(krzysd): convert TFLite ops to MLIR ops
    // Note: EnumNamesBuiltinOperator has the names of the builtin ops in
    // uppercase. We will want them in lowercase with a tfl. prefix for MLIR
    OpBuilder op_builder{func.getBody()};
    op_builder.create<mlir::ReturnOp>(base_loc);
    module.push_back(func);
  }

  return OwningModuleRef(module);
}

static OwningModuleRef FlatBufferFileToMlirTrans(llvm::StringRef filename,
                                                 MLIRContext* context) {
  std::string error;
  auto loc = mlir::FileLineColLoc::get(filename, 0, 0, context);
  auto buffer = mlir::openInputFile(filename, &error);
  if (nullptr == buffer) {
    return emitError(loc, error), nullptr;
  }

  return tflite::FlatBufferToMlir(
      absl::string_view(buffer->getBufferStart(), buffer->getBufferSize()),
      context, loc);
}

static mlir::TranslateToMLIRRegistration FlatBufferFileToMlirTransReg(
    "tflite-flatbuffer-to-mlir", FlatBufferFileToMlirTrans);
