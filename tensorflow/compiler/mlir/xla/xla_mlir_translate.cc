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

#include "tensorflow/compiler/mlir/xla/xla_mlir_translate.h"

#include "google/protobuf/text_format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/Translation.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/mlir/xla/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/core/lib/core/errors.h"

using stream_executor::port::Status;
using stream_executor::port::StatusOr;  // NOLINT TODO(b/130822468) fix this

namespace xla {

namespace {
// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public ::proto2::io::ErrorCollector {
 public:
  void AddError(int line, int column, const string& message) override {}
};

bool LoadHloProto(const std::string& contents, HloProto* hlo_proto) {
  ::proto2::TextFormat::Parser parser;
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  return hlo_proto->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto) ||
         hlo_proto->mutable_hlo_module()->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto->mutable_hlo_module());
}

}  // namespace

mlir::OwningModuleRef HloToMlirHloTranslateFunction(
    llvm::StringRef input_filename, mlir::MLIRContext* context) {
  auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_filename.str());
  if (std::error_code error = file_or_err.getError()) {
    LOG(ERROR) << "Failure to read HLO module: " << error;
    return nullptr;
  }

  auto& input_file = *file_or_err;
  HloProto hlo_proto;
  string content(input_file->getBufferStart(), input_file->getBufferSize());
  if (!LoadHloProto(content, &hlo_proto)) {
    LOG(ERROR) << "Failed to load proto: " << input_filename.str();
    return nullptr;
  }

  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  auto status =
      ConvertHloToMlirHlo(module.get(), hlo_proto.mutable_hlo_module());
  if (!status.ok()) {
    LOG(ERROR) << "Hlo module import failed: " << status;
    return nullptr;
  }

  return module;
}

mlir::OwningModuleRef HloTextToMlirHloTranslateFunction(
    llvm::StringRef input_filename, mlir::MLIRContext* context) {
  auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_filename.str());
  if (std::error_code error = file_or_err.getError()) {
    LOG(ERROR) << "Failure to open file: " << error;
    return nullptr;
  }

  auto& input_file = *file_or_err;
  HloProto hlo_proto;
  string content(input_file->getBufferStart(), input_file->getBufferSize());

  auto hlo_module_error = ParseAndReturnUnverifiedModule(content);
  if (!hlo_module_error.ok()) {
    LOG(ERROR) << "HLO Module loading failed: " << hlo_module_error.status();
    return nullptr;
  }

  auto hlo_module = std::move(hlo_module_error.ValueOrDie());
  mlir::OwningModuleRef module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context));
  auto status = ConvertHloToMlirHlo(*module, hlo_module.get());
  if (!status.ok()) {
    LOG(ERROR) << "HLO Module import failed: " << status;
    return nullptr;
  }

  return module;
}

static mlir::LogicalResult MlirHloToHloTranslateFunction(
    mlir::ModuleOp module, llvm::StringRef output_filename) {
  if (!module) return mlir::failure();

  std::error_code error;
  auto result = llvm::make_unique<llvm::ToolOutputFile>(output_filename, error,
                                                        llvm::sys::fs::F_None);
  if (error) {
    LOG(ERROR) << error.message();
    return mlir::failure();
  }

  HloProto hloProto;
  Status status = mlir::ConvertMlirHloToHlo(module, &hloProto);
  if (!status.ok()) {
    LOG(ERROR) << "Module conversion failed: " << status;
    return mlir::failure();
  }

  result->os() << hloProto.DebugString();
  result->keep();
  return mlir::success();
}

static StatusOr<std::unique_ptr<HloModule>> HloModuleFromProto(
    const HloProto& hlo_proto) {
  const HloModuleProto& module_proto = hlo_proto.hlo_module();
  TF_ASSIGN_OR_RETURN(const HloModuleConfig module_config,
                      HloModule::CreateModuleConfigFromProto(
                          module_proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(module_proto, module_config);
}

static mlir::LogicalResult MlirHloToHloTextTranslateFunction(
    mlir::ModuleOp module, llvm::StringRef output_filename) {
  if (!module) return mlir::failure();

  std::error_code error;
  auto result = llvm::make_unique<llvm::ToolOutputFile>(output_filename, error,
                                                        llvm::sys::fs::F_None);
  if (error) {
    LOG(ERROR) << error.message();
    return mlir::failure();
  }

  HloProto hloProto;
  Status status = mlir::ConvertMlirHloToHlo(module, &hloProto);
  if (!status.ok()) {
    LOG(ERROR) << "Module conversion failed: " << status;
    return mlir::failure();
  }

  auto statusOrHloModule = HloModuleFromProto(hloProto);

  if (!statusOrHloModule.ok()) {
    LOG(ERROR) << "Conversion to HLO module failed: "
               << statusOrHloModule.status();
    return mlir::failure();
  }

  result->os() << statusOrHloModule.ValueOrDie()->ToString(
      HloPrintOptions()
          // We don't interpret or use layouts
          .set_include_layout_in_shapes(false));
  result->keep();
  return mlir::success();
}

}  // namespace xla

static mlir::TranslateFromMLIRRegistration MlirHloToHloTranslate(
    "mlir-hlo-to-hlo", xla::MlirHloToHloTranslateFunction);

static mlir::TranslateFromMLIRRegistration MlirHloToHloTextTranslate(
    "mlir-hlo-to-hlo-text", xla::MlirHloToHloTextTranslateFunction);

static mlir::TranslateToMLIRRegistration HloToHloMlirTranslate(
    "hlo-to-mlir-hlo", xla::HloToMlirHloTranslateFunction);

static mlir::TranslateToMLIRRegistration HloTextToHloMlirTranslate(
    "hlo-text-to-mlir-hlo", xla::HloTextToMlirHloTranslateFunction);
