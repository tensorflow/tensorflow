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
#include "tensorflow/compiler/mlir/lite/python/jax_to_tfl_flatbuffer.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/model_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/python/tf_tfl_flatbuffer_helpers.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/types.pb.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_debug_info.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

namespace tensorflow {
namespace {

// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public tsl::protobuf::io::ErrorCollector {
 public:
  void AddError(int line, int column, const std::string& message) override {}
};

absl::StatusOr<xla::HloProto> LoadHloProto(const std::string& contents) {
  xla::HloProto hlo_proto;
  // NOLINTNEXTLINE: Use tsl::protobuf to be compatible with OSS.
  tsl::protobuf::TextFormat::Parser parser;
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  bool status =
      hlo_proto.ParseFromString(contents) ||
      parser.ParseFromString(contents, &hlo_proto) ||
      hlo_proto.mutable_hlo_module()->ParseFromString(contents) ||
      parser.ParseFromString(contents, hlo_proto.mutable_hlo_module());
  if (!status) {
    return absl::InternalError("Failed to parse HloProto");
  }
  return hlo_proto;
}

}  // namespace

absl::Status ConvertJaxToTFLiteFlatBuffer(
    const std::string& input, const tflite::ModelFlags& model_flags,
    tflite::ConverterFlags& converter_flags, std::string* result) {
  auto context = std::make_unique<mlir::MLIRContext>();
  mlir::quant::QuantizationSpecs quant_specs;

  // Parse input arrays.
  std::vector<std::string> node_names;
  std::vector<std::string> node_dtypes;
  std::vector<std::optional<std::vector<int>>> node_shapes;
  std::vector<std::optional<double>> node_mins;
  std::vector<std::optional<double>> node_maxs;

  // Populate quantization specs.
  TF_RETURN_IF_ERROR(internal::PopulateQuantizationSpecs(
      model_flags, converter_flags, &quant_specs, &node_names, &node_dtypes,
      &node_shapes, &node_mins, &node_maxs));

  internal::WarningUnusedFlags(model_flags, converter_flags);

  // Register all custom ops, including user-specified custom ops.
  TF_RETURN_IF_ERROR(internal::RegisterAllCustomOps(converter_flags));

  mlir::TFL::PassConfig pass_config(quant_specs);
  bool emit_builtin_tflite_ops = !converter_flags.force_select_tf_ops();
  pass_config.emit_builtin_tflite_ops = emit_builtin_tflite_ops;
  pass_config.enable_tflite_variables =
      converter_flags.enable_tflite_resource_variables();
  pass_config.unfold_batch_matmul = converter_flags.unfold_batchmatmul();
  pass_config.lower_tensor_list_ops = converter_flags.lower_tensor_list_ops();
  // Disable the unfolding of the 16x16 TF::BatchMatMulOp to avoid the
  // conversion to an unsupported 16x16 TFL::FullyConnectedOp.
  if (converter_flags.inference_type() == tflite::IODataType::QUANTIZED_INT16) {
    pass_config.unfold_batch_matmul = false;
  }
  pass_config.unfold_large_splat_constant =
      converter_flags.unfold_large_splat_constant();
  pass_config.enable_hlo_to_tf_conversion = true;
  pass_config.enable_stablehlo_conversion =
      converter_flags.convert_to_stablehlo();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::string content(input.data(), input.size());
  if (model_flags.hlo_file_type() == tflite::ModelFlags::HLO_TEXT) {
    TF_ASSIGN_OR_RETURN(auto hlo_module,
                        xla::ParseAndReturnUnverifiedModule(content));
    TF_ASSIGN_OR_RETURN(auto module,
                        xla::ConvertHloToStablehlo(*context, hlo_module.get()));
  } else if (model_flags.hlo_file_type() == tflite::ModelFlags::HLO_PROTO) {
    TF_ASSIGN_OR_RETURN(xla::HloProto hlo_proto, LoadHloProto(content));
    TF_ASSIGN_OR_RETURN(module, xla::ConvertHloToStablehlo(
                                    *context, hlo_proto.mutable_hlo_module()));
  } else {
    return absl::InvalidArgumentError("Unknown hlo format type");
  }

  // Set the input names.
  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func) return errors::Internal("Failed to find the main function.");
  // Retrieve input names from model flags.
  std::vector<std::string> input_names;
  for (const auto& input : model_flags.input_arrays()) {
    input_names.push_back(input.name());
  }

  const auto& inputs = absl::StrJoin(input_names, ",");
  mlir::OpBuilder builder(*module);
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(
      builder.getNamedAttr("inputs", builder.getStringAttr(inputs)));
  // Jax wrapped the output nodes in a tuple, so it's pretty hard to us
  // to tell the output at this point, we will set the output at the export
  // phase.
  main_func->setAttr("tf.entry_function", builder.getDictionaryAttr(attrs));

  // StableHLO Quantizer is not supported for JAX input models, so
  // quantization_py_function_lib is set to nullptr.
  auto status = internal::ConvertMLIRToTFLiteFlatBuffer(
      model_flags, converter_flags, std::move(context), std::move(module),
      pass_config, /*saved_model_tags=*/{}, result,
      /*quantization_py_function_lib=*/nullptr);
  return status;
}

}  // namespace tensorflow
