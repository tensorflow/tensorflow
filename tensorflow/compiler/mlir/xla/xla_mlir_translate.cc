/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/mlir/xla/xla_mlir_translate.h"

#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/mlir_hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

namespace {
// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public tensorflow::protobuf::io::ErrorCollector {
 public:
  void AddError(int line, int column, const std::string& message) override {}
};

bool LoadHloProto(const std::string& contents, HloProto* hlo_proto) {
  tensorflow::protobuf::TextFormat::Parser parser;
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  return hlo_proto->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto) ||
         hlo_proto->mutable_hlo_module()->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto->mutable_hlo_module());
}

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> HloToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations) {
  HloProto hlo_proto;
  std::string content(input.data(), input.size());
  if (!LoadHloProto(content, &hlo_proto)) {
    LOG(ERROR) << "Failed to load proto";
    return nullptr;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  auto status = ConvertHloToMlirHlo(
      module.get(), hlo_proto.mutable_hlo_module(), import_all_computations);
  if (!status.ok()) {
    LOG(ERROR) << "Hlo module import failed: " << status;
    return nullptr;
  }

  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> HloTextToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations) {
  std::string content(input.data(), input.size());

  auto hlo_module_error = ParseAndReturnUnverifiedModule(content);
  if (!hlo_module_error.ok()) {
    LOG(ERROR) << "HLO Module loading failed: " << hlo_module_error.status();
    return nullptr;
  }

  auto hlo_module = std::move(hlo_module_error.value());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  auto status =
      ConvertHloToMlirHlo(*module, hlo_module.get(), import_all_computations);
  if (!status.ok()) {
    LOG(ERROR) << "HLO Module import failed: " << status;
    return nullptr;
  }

  return module;
}

}  // namespace xla
