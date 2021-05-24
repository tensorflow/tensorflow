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

#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"

#include <string>
#include <unordered_set>

#include "absl/types/span.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/tools/optimize/quantize_weights.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {
namespace {
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OwningModuleRef;
using stream_executor::port::StatusOr;

bool IsControlFlowV1Op(Operation* op) {
  return mlir::isa<mlir::tf_executor::SwitchOp, mlir::tf_executor::MergeOp,
                   mlir::tf_executor::EnterOp, mlir::tf_executor::ExitOp,
                   mlir::tf_executor::NextIterationSinkOp,
                   mlir::tf_executor::NextIterationSourceOp>(op);
}

mlir::LogicalResult IsValidGraph(mlir::ModuleOp module) {
  auto result = module.walk([&](Operation* op) {
    return IsControlFlowV1Op(op) ? mlir::WalkResult::interrupt()
                                 : mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    module.emitError(
        "The graph has Control Flow V1 ops. TFLite converter doesn't support "
        "Control Flow V1 ops. Consider using Control Flow V2 ops instead. See "
        "https://www.tensorflow.org/api_docs/python/tf/compat/v1/"
        "enable_control_flow_v2.");
    return mlir::failure();
  }
  return mlir::success();
}

// Util that registers 'extra_tf_opdefs' to the TF global registry.
// Return OK on success, failure if registering failed.
Status RegisterExtraTfOpDefs(absl::Span<const std::string> extra_tf_opdefs) {
  for (const auto& tf_opdefs_string : extra_tf_opdefs) {
    tensorflow::OpDef opdef;
    if (!tensorflow::protobuf::TextFormat::ParseFromString(tf_opdefs_string,
                                                           &opdef)) {
      LOG(ERROR) << "OpDef parsing failed for: " << tf_opdefs_string;
      return errors::InvalidArgument("fail to parse extra OpDef");
    }
    // Register extra opdefs.
    // TODO(b/133770952): Support shape functions.
    tensorflow::OpRegistry::Global()->Register(
        [opdef](tensorflow::OpRegistrationData* op_reg_data) -> Status {
          *op_reg_data = tensorflow::OpRegistrationData(opdef);
          return Status::OK();
        });
  }
  return Status::OK();
}
}  // namespace

StatusOr<OwningModuleRef> LoadFromGraphdefOrMlirSource(
    const std::string& input_filename, bool input_mlir,
    bool use_splatted_constant, const std::vector<std::string>& extra_tf_opdefs,
    const GraphImportConfig& specs, absl::string_view debug_info_file,
    absl::string_view input_arrays, absl::string_view input_dtypes,
    absl::string_view input_shapes, absl::string_view output_arrays,
    absl::string_view control_output_arrays, llvm::SourceMgr* source_mgr,
    MLIRContext* context) {
  // Set up the input file.
  std::string error_message;
  auto file = mlir::openInputFile(input_filename, &error_message);
  if (!file) {
    llvm::errs() << error_message << "\n";
    return errors::InvalidArgument("fail to open input file");
  }

  if (input_mlir) {
    source_mgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc());
    return OwningModuleRef(mlir::parseSourceFile(*source_mgr, context));
  }

  // Register extra TF ops passed as OpDef.
  auto extra_opdefs_status = RegisterExtraTfOpDefs(extra_tf_opdefs);
  if (!extra_opdefs_status.ok()) return extra_opdefs_status;

  if (use_splatted_constant) {
    return tensorflow::GraphdefToSplattedMlirTranslateFunction(
        file->getBuffer(), debug_info_file, input_arrays, input_dtypes,
        input_shapes, output_arrays, control_output_arrays,
        specs.prune_unused_nodes, /*convert_legacy_fed_inputs=*/true,
        /*graph_as_function=*/false, specs.upgrade_legacy,
        /*enable_shape_inference=*/false, context);
  }
  return tensorflow::GraphdefToMlirTranslateFunction(
      file->getBuffer(), debug_info_file, input_arrays, input_dtypes,
      input_shapes, output_arrays, control_output_arrays,
      specs.prune_unused_nodes, /*convert_legacy_fed_inputs=*/true,
      /*graph_as_function=*/false, specs.upgrade_legacy,
      /*enable_shape_inference=*/false, context);
}

Status ConvertTFExecutorToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir, bool emit_builtin_tflite_ops,
    bool emit_select_tf_ops, bool emit_custom_ops,
    const std::unordered_set<std::string>& select_user_tf_ops,
    const mlir::TFL::QuantizationSpecs& quant_specs,
    const std::unordered_set<std::string>& saved_model_tags,
    std::string* result, mlir::PassManager* pass_manager) {
  // Explicitly disable dumping Op details on failures.
  module.getContext()->printOpOnDiagnostic(false);

  // Register a warning handler only log to std out.
  mlir::ScopedDiagnosticHandler s(
      module.getContext(), [](mlir::Diagnostic& diag) {
        if (diag.getSeverity() == mlir::DiagnosticSeverity::Warning) {
          for (auto& note : diag.getNotes()) {
            std::cout << note.str() << "\n";
            LOG(WARNING) << note.str() << "\n";
          }
        }
        return mlir::failure();
      });

  mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext(),
                                                    /*propagate=*/true);

  if (failed(IsValidGraph(module)) || failed(pass_manager->run(module))) {
    return statusHandler.ConsumeStatus();
  }

  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return Status::OK();
  }

  // Write MLIR TFLite dialect into FlatBuffer
  OpOrArgLocNameMapper op_or_arg_name_mapper;
  if (!quant_specs.RunWeightQuantization()) {
    tflite::FlatbufferExportOptions options;
    options.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
    options.emit_select_tf_ops = emit_select_tf_ops;
    options.select_user_tf_ops = select_user_tf_ops;
    options.emit_custom_ops = emit_custom_ops;
    options.saved_model_tags = saved_model_tags;
    options.op_or_arg_name_mapper = &op_or_arg_name_mapper;
    if (!tflite::MlirToFlatBufferTranslateFunction(module, options, result)) {
      return statusHandler.ConsumeStatus();
    }
  } else {
    // Post-training weight quantization path. Once MLIR has support for this,
    // we can remove this else statement.
    std::string pre_quantized_result;
    tflite::FlatbufferExportOptions options;
    options.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
    options.emit_select_tf_ops = emit_select_tf_ops;
    options.select_user_tf_ops = select_user_tf_ops;
    options.emit_custom_ops = emit_custom_ops;
    options.saved_model_tags = saved_model_tags;
    options.op_or_arg_name_mapper = &op_or_arg_name_mapper;
    if (!tflite::MlirToFlatBufferTranslateFunction(module, options,
                                                   &pre_quantized_result)) {
      return statusHandler.ConsumeStatus();
    }
    flatbuffers::FlatBufferBuilder q_builder(/*initial_size=*/10240);
    const uint8_t* buffer =
        reinterpret_cast<const uint8_t*>(pre_quantized_result.c_str());
    const ::tflite::Model* input_model = ::tflite::GetModel(buffer);

    ::tflite::optimize::BufferType quantized_type;
    if (quant_specs.inference_type == tensorflow::DT_QINT8) {
      quantized_type = ::tflite::optimize::BufferType::QUANTIZED_INT8;
    } else if (quant_specs.inference_type == tensorflow::DT_HALF) {
      quantized_type = ::tflite::optimize::BufferType::QUANTIZED_FLOAT16;
    } else {
      return errors::InvalidArgument("Quantized type not supported");
    }
    if (::tflite::optimize::QuantizeWeights(&q_builder, input_model,
                                            quantized_type) != kTfLiteOk) {
      return errors::InvalidArgument("Quantize weights transformation failed.");
    }
    const uint8_t* q_buffer = q_builder.GetBufferPointer();
    *result =
        string(reinterpret_cast<const char*>(q_buffer), q_builder.GetSize());
  }

  if (mlir::failed(module.verify())) {
    return tensorflow::errors::Unknown("Final module is invalid");
  }
  return Status::OK();
}

StatusOr<mlir::OwningModuleRef> ImportSavedModel(
    const std::string& input_filename, const int saved_model_version,
    const std::unordered_set<std::string>& tags,
    absl::Span<const std::string> extra_tf_opdefs,
    absl::Span<std::string> exported_names, const GraphImportConfig& specs,
    mlir::MLIRContext* context) {
  // Register extra TF ops passed as OpDef.
  auto extra_opdefs_status = RegisterExtraTfOpDefs(extra_tf_opdefs);
  if (!extra_opdefs_status.ok()) return extra_opdefs_status;

  if (saved_model_version == 2) {
    auto module_or = tensorflow::SavedModelObjectGraphToMlirImport(
        input_filename, tags, exported_names, context);
    if (!module_or.status().ok()) return module_or.status();
    return module_or.ConsumeValueOrDie();
  } else if (saved_model_version == 1) {
    MLIRImportOptions options;
    options.upgrade_legacy = specs.upgrade_legacy;
    auto module_or = tensorflow::SavedModelSignatureDefsToMlirImport(
        input_filename, tags, exported_names, context, options);

    if (!module_or.status().ok()) return module_or.status();
    return module_or.ConsumeValueOrDie();
  } else {
    return tensorflow::errors::InvalidArgument(
        "Should be either saved model v1 or v2");
  }
}

}  // namespace tensorflow
