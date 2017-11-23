/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/toco/toco_tooling.h"

#include <cstdlib>
#include <memory>
#include <set>

#include "absl/strings/str_join.h"
#include "tensorflow/contrib/lite/toco/allocate_transient_arrays.h"
#include "tensorflow/contrib/lite/toco/dump_graphviz.h"
#include "tensorflow/contrib/lite/toco/export_tensorflow.h"
#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/import_tensorflow.h"
#include "tensorflow/contrib/lite/toco/model_flags.pb.h"
#include "tensorflow/contrib/lite/toco/tflite/export.h"
#include "tensorflow/contrib/lite/toco/tflite/import.h"
#include "tensorflow/contrib/lite/toco/toco_flags.pb.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {
namespace {
// CHECK-fails if the model contains a kTensorFlowUnsupported operation.
void CheckUnsupportedOperations(const Model& model) {
  std::set<string> unsupported_ops;
  for (auto& op : model.operators) {
    if (op->type == OperatorType::kTensorFlowUnsupported) {
      unsupported_ops.insert(
          static_cast<const TensorFlowUnsupportedOperator*>(op.get())
              ->tensorflow_op);
    }
  }
  QCHECK(unsupported_ops.empty())
      << "These unsupported ops were not removed by graph transformations: "
      << absl::StrJoin(unsupported_ops, ", ");
}

void MakeGeneralGraphTransformationsSet(
    GraphTransformationsSet* transformations) {
  CHECK(transformations->empty());
  transformations->Add(new ResolveReshapeAttributes);
  transformations->Add(new PropagateArrayDataTypes);
  transformations->Add(new PropagateFixedSizes);
  transformations->Add(new RemoveTensorFlowAssert);
  transformations->Add(new RemoveTensorFlowIdentity);
  transformations->Add(new RemoveTrivialConcatenation);
  transformations->Add(new RemoveTrivialConcatenationInput);
  transformations->Add(new RemoveUnusedOp);
  transformations->Add(new EnsureBiasVectors);
  transformations->Add(new ResolveReorderAxes);
  transformations->Add(new ResolveTensorFlowMatMul);
  transformations->Add(new FuseBinaryIntoPrecedingAffine);
  transformations->Add(new FuseBinaryIntoFollowingAffine);
  transformations->Add(new ResolveBatchNormalization);
  transformations->Add(new ResolveConstantBinaryOperator);
  transformations->Add(new ResolveConstantUnaryOperator);
  transformations->Add(new ResolveTensorFlowMerge);
  transformations->Add(new ResolveTensorFlowSqueeze);
  transformations->Add(new ResolveTensorFlowSwitch);
  transformations->Add(new ResolveTensorFlowTile);
  transformations->Add(new ResolveTensorFlowConcat);
  transformations->Add(new IdentifyL2Normalization);
  transformations->Add(new IdentifyL2Pool);
  transformations->Add(new IdentifyRelu1);
  transformations->Add(new RemoveTrivialBinaryOperator);
  transformations->Add(new ReadFakeQuantMinMax);
  transformations->Add(new ResolvePadAttributes);
  transformations->Add(new ResolveStridedSliceAttributes);
  transformations->Add(new ResolveSliceAttributes);
  transformations->Add(new ResolveMeanAttributes);
  transformations->Add(new ResolveConstantTensorFlowShape);
  transformations->Add(new MakeInitialDequantizeOperator);
}

void SetArrayFinalDataTypes(const TocoFlags& toco_flags, Model* model) {
  const bool output_supports_only_float =
      toco_flags.output_format() == TENSORFLOW_GRAPHDEF;

  ArrayDataType specified_final_data_type = ArrayDataType::kNone;
  if (toco_flags.has_inference_input_type()) {
    specified_final_data_type =
        ConvertIODataTypeToArrayDataType(toco_flags.inference_input_type());
  } else if (toco_flags.has_inference_type()) {
    specified_final_data_type =
        ConvertIODataTypeToArrayDataType(toco_flags.inference_type());
  }
  ArrayDataType final_data_type = ArrayDataType::kNone;
  if (output_supports_only_float) {
    QCHECK(specified_final_data_type == ArrayDataType::kNone ||
           specified_final_data_type == ArrayDataType::kFloat);
    final_data_type = ArrayDataType::kFloat;
  } else {
    final_data_type = specified_final_data_type;
  }
  for (int i = 0; i < model->flags.input_arrays_size(); i++) {
    auto* array = model->arrays[model->flags.input_arrays(i).name()].get();
    // Note that the notion of changing data types only applies to real-numbers
    // arrays (see the documentation for inference_input_type).
    // TODO(benoitjacob) this is assuming that uint8 arrays are quantized,
    // i.e. represent real numbers by means of quantization parameters,
    // and not plain integer uint8 input arrays.
    const bool is_real_numbers = array->data_type == ArrayDataType::kFloat ||
                                 array->data_type == ArrayDataType::kUint8;
    if (is_real_numbers) {
      array->final_data_type = final_data_type;
    }
  }
}

}  // namespace

std::unique_ptr<Model> Import(const TocoFlags& toco_flags,
                              const ModelFlags& model_flags,
                              const string& input_file_contents) {
  std::unique_ptr<Model> model;
  switch (toco_flags.input_format()) {
    case TENSORFLOW_GRAPHDEF:
      model = ImportTensorFlowGraphDef(model_flags, input_file_contents);
      break;
    case TFLITE:
      model = toco::tflite::Import(model_flags, input_file_contents);
      ResolveModelFlags(model_flags, model.get());
      CheckInvariants(*model);
      break;
    default:
      LOG(FATAL) << "Unhandled input_format";
  }

  LogDump(kLogLevelModelChanged, "AT IMPORT", *model);

  return model;
}

void Transform(const TocoFlags& toco_flags, Model* model) {
  const FileFormat output_format = toco_flags.output_format();
  const IODataType inference_type = toco_flags.inference_type();

  const bool output_is_tflite = output_format == TFLITE;

  const bool output_is_tflite_quantized =
      output_is_tflite && inference_type == QUANTIZED_UINT8;

  if (output_is_tflite_quantized) {
    QCHECK_NE(toco_flags.inference_input_type(), FLOAT)
        << "Quantized inference is not allowed with float inputs.";
  }

  SetArrayFinalDataTypes(toco_flags, model);

  GraphTransformationsSet transformations;
  MakeGeneralGraphTransformationsSet(&transformations);
  auto* remove_trivial_reshape = new RemoveTrivialReshape;
  transformations.Add(remove_trivial_reshape);
  if (output_format == TFLITE) {
    transformations.Add(new FuseActivationFunctions);
  } else {
    transformations.Add(new UnfuseActivationFunctions);
  }
  if (output_format != TENSORFLOW_GRAPHDEF) {
    transformations.Add(new ResolveConstantFakeQuant);
  }
  if (toco_flags.drop_fake_quant()) {
    transformations.Add(new DropFakeQuant);
  } else {
    // See the doc for --reorder_across_fake_quant: that flag is needed to
    // support some existing models, e.g. WordLens, that have FakeQuant
    // nodes in the wrong places.
    // We currently unconditionally enable that behavior when the output
    // format is DarwiNN because the DarwiNN test code does not make it
    // easy to pass a new toco flag. Once that is resolved on the DarwiNN
    // tests side, the special-casing of DarwiNN here can go away.
    // TODO(benoitjacob): so drop it when we can.
    if ((output_is_tflite_quantized &&
         toco_flags.reorder_across_fake_quant())) {
      transformations.Add(new DropFakeQuant);
    }
  }
  transformations.Add(new ConvertPureConvToDepthwise);
  // TFLite export does not yet support fused LSTM cell.
  if (output_format == TENSORFLOW_GRAPHDEF) {
    transformations.Add(new IdentifyLstmCell);
  }
  transformations.Add(new ResolveConstantConcatenation);
  RunGraphTransformations(model, "general graph transformations",
                          transformations);
  if (output_is_tflite_quantized) {
    RunGraphTransformations(model, "pre-quantization graph transformations",
                            {new HardcodeMinMax, new DropFakeQuant});
  }

  if (output_is_tflite_quantized) {
    if (toco_flags.has_default_ranges_min() &&
        toco_flags.has_default_ranges_max()) {
      UseDefaultMinMaxRangeValues(model, toco_flags.default_ranges_min(),
                                  toco_flags.default_ranges_max());
    }
    CheckIsReadyForQuantization(*model);
    RunGraphTransformations(
        model, "quantization graph transformations",
        {new Quantize, new RemoveTrivialQuantizedActivationFunc,
         new RemoveFinalDequantizeOp});
  } else {
    GraphTransformationsSet dequantization_transformations{new Dequantize};
    // Dequantize creates FakeQuant nodes. We may want to discard
    // those immediately.
    if (toco_flags.drop_fake_quant()) {
      dequantization_transformations.Add(new DropFakeQuant);
    }

    RunGraphTransformations(model, "dequantization graph transformations",
                            dequantization_transformations);
  }

  LogDump(kLogLevelModelChanged, "AFTER TRANSFORMATIONS", *model);

  if (output_format != GRAPHVIZ_DOT && output_format != TFLITE) {
    // By now there shouldn't be any unsupported ops when exporting to
    // TensorFlow GraphDef.
    CheckUnsupportedOperations(*model);
  }

  if (output_is_tflite) {
    AllocateTransientArrays(model, kDefaultTransientDataAlignment);
    LogDump(kLogLevelModelChanged, "AFTER ALLOCATION", *model);
  }

  CheckModelCounts(*model);
  CheckFinalDataTypesSatisfied(*model);

  int64 ops_count;
  if (EstimateArithmeticOpsCount(*model, &ops_count)) {
    LOG(INFO) << "Estimated count of arithmetic ops: " << 1e-9 * ops_count
              << " billion (note that a multiply-add is counted as 2 ops).";
  }
}

void Export(const TocoFlags& toco_flags, const Model& model,
            bool allow_custom_ops, string* output_file_contents) {
  switch (toco_flags.output_format()) {
    case TENSORFLOW_GRAPHDEF:
      ExportTensorFlowGraphDef(model, output_file_contents);
      break;
    case TFLITE:
      toco::tflite::Export(model, allow_custom_ops, output_file_contents);
      break;
    case GRAPHVIZ_DOT:
      DumpGraphviz(model, output_file_contents);
      break;
    default:
      LOG(FATAL) << "Unhandled output_format";
  }
}

}  // namespace toco
