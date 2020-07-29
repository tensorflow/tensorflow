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

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
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

using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OwningModuleRef;
using stream_executor::port::StatusOr;

StatusOr<OwningModuleRef> LoadFromGraphdefOrMlirSource(
    const std::string& input_filename, bool input_mlir,
    bool use_splatted_constant, const std::vector<std::string>& extra_tf_opdefs,
    absl::string_view debug_info_file, absl::string_view input_arrays,
    absl::string_view input_dtypes, absl::string_view input_shapes,
    absl::string_view output_arrays, bool prune_unused_nodes,
    llvm::SourceMgr* source_mgr, MLIRContext* context) {
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

  if (use_splatted_constant) {
    return tensorflow::GraphdefToSplattedMlirTranslateFunction(
        file->getBuffer(), debug_info_file, input_arrays, input_dtypes,
        input_shapes, output_arrays, /*control_output_arrays=*/"",
        prune_unused_nodes, /*convert_legacy_fed_inputs=*/true,
        /*graph_as_function=*/false, /*upgrade_legacy=*/true,
        /*enable_shape_inference=*/false, context);
  }
  return tensorflow::GraphdefToMlirTranslateFunction(
      file->getBuffer(), debug_info_file, input_arrays, input_dtypes,
      input_shapes, output_arrays, /*control_output_arrays=*/"",
      prune_unused_nodes, /*convert_legacy_fed_inputs=*/true,
      /*graph_as_function=*/false, /*upgrade_legacy=*/true,
      /*enable_shape_inference=*/false, context);
}

Status ConvertTFExecutorToTFLOrFlatbuffer(
    mlir::ModuleOp module, bool export_to_mlir, bool emit_builtin_tflite_ops,
    bool emit_select_tf_ops, bool emit_custom_ops,
    const mlir::TFL::QuantizationSpecs& quant_specs, std::string* result,
    mlir::PassManager* pass_manager) {
  mlir::StatusScopedDiagnosticHandler statusHandler(module.getContext(),
                                                    /*propagate=*/true);
  if (failed(pass_manager->run(module))) {
    return statusHandler.ConsumeStatus();
  }

  if (export_to_mlir) {
    llvm::raw_string_ostream os(*result);
    module.print(os);
    return Status::OK();
  }

  // Write MLIR TFLite dialect into FlatBuffer
  if (!quant_specs.RunWeightQuantization()) {
    if (tflite::MlirToFlatBufferTranslateFunction(
            module, result, emit_builtin_tflite_ops, emit_select_tf_ops,
            emit_custom_ops)) {
      return statusHandler.ConsumeStatus();
    }
  } else {
    // Post-training weight quantization path. Once MLIR has support for this,
    // we can remove this else statement.
    std::string pre_quantized_result;
    if (tflite::MlirToFlatBufferTranslateFunction(
            module, &pre_quantized_result, emit_builtin_tflite_ops,
            emit_select_tf_ops, emit_custom_ops)) {
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

  return Status::OK();
}

StatusOr<mlir::OwningModuleRef> ImportSavedModel(
    const std::string& input_filename, const int saved_model_version,
    const std::unordered_set<std::string>& tags,
    absl::Span<std::string> exported_names, mlir::MLIRContext* context) {
  if (saved_model_version == 2) {
    auto module = tensorflow::SavedModelObjectGraphToMlirImport(
        input_filename, tags, exported_names, context);
    if (!module)
      return tensorflow::errors::InvalidArgument("fail to open input file");

    return module;
  } else if (saved_model_version == 1) {
    auto module = tensorflow::SavedModelSignatureDefsToMlirImport(
        input_filename, tags, exported_names, context);

    if (!module)
      return tensorflow::errors::InvalidArgument("fail to open input file");

    return module;
  } else {
    return tensorflow::errors::InvalidArgument(
        "Should be either saved model v1 or v2");
  }
}

}  // namespace tensorflow
