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
#include "tensorflow/lite/toco/toco_tooling.h"

#include <cstdlib>
#include <memory>
#include <set>

#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/toco/allocate_transient_arrays.h"
#include "tensorflow/lite/toco/dump_graphviz.h"
#include "tensorflow/lite/toco/export_tensorflow.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/import_tensorflow.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/tflite/export.h"
#include "tensorflow/lite/toco/tflite/import.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {
namespace {
// CHECK-fails if the model contains a kUnsupported operation.
void CheckUnsupportedOperations(const Model& model) {
  std::set<string> unsupported_ops;
  for (auto& op : model.operators) {
    if (op->type == OperatorType::kUnsupported) {
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
  transformations->Add(new ConvertExpandDimsToReshape);
  transformations->Add(new ConvertSqueezeToReshape);
  transformations->Add(new ConvertTrivialAddNToAdd);
  transformations->Add(new ConvertTrivialPackToReshape);
  transformations->Add(new ConvertTrivialTileToConcat);
  transformations->Add(new ConvertTrivialTransposeToReshape);
  transformations->Add(new ConvertReorderAxes);
  transformations->Add(new ResolveReshapeAttributes);
  transformations->Add(new ResolveTransposeAttributes);
  transformations->Add(new PropagateActivationFunctionIntoConstants);
  transformations->Add(new PropagateArrayDataTypes);
  transformations->Add(new PropagateFixedSizes);
  transformations->Add(new RemoveTensorFlowAssert);
  transformations->Add(new RemoveTensorFlowIdentity);
  transformations->Add(new RemoveTrivialConcatenation);
  transformations->Add(new RemoveTrivialConcatenationInput);
  transformations->Add(new RemoveTrivialFakeQuant);
  transformations->Add(new RemoveTrivialSlice);
  transformations->Add(new RemoveUnusedOp);
  transformations->Add(new EnsureBiasVectors);
  transformations->Add(new ResolveReorderAxes);
  transformations->Add(new UnrollBatchMatMul);
  transformations->Add(new ResolveTensorFlowMatMul);
  transformations->Add(new FuseBinaryIntoPrecedingAffine);
  transformations->Add(new FuseBinaryIntoFollowingAffine);
  transformations->Add(new FuseBroadcastIntoFollowingBinary);
  transformations->Add(new MergeReshapeIntoPrecedingTranspose);
  transformations->Add(new MoveBinaryOperatorBeforeReshape);
  transformations->Add(new ReorderElementwiseUnary);
  transformations->Add(new ReorderReshapeTranspose);
  transformations->Add(new ResolveBatchNormalization);
  transformations->Add(new ResolveConstantBinaryOperator);
  transformations->Add(new ResolveConstantFill);
  transformations->Add(new ResolveConstantGather);
  transformations->Add(new ResolveConstantPack);
  transformations->Add(new ResolveConstantRandomUniform);
  transformations->Add(new ResolveConstantRange);
  transformations->Add(new ResolveConstantReshape);
  transformations->Add(new ResolveConstantSelect);
  transformations->Add(new ResolveConstantSlice);
  transformations->Add(new ResolveConstantStridedSlice);
  transformations->Add(new ResolveConstantTile);
  transformations->Add(new ResolveConstantTranspose);
  transformations->Add(new ResolveConstantUnaryOperator);
  transformations->Add(new ResolveTensorFlowMerge);
  transformations->Add(new ResolveSqueezeAttributes);
  transformations->Add(new ResolveTensorFlowSwitch);
  transformations->Add(new ResolveTensorFlowConcat);
  transformations->Add(new ResolveMultiplyByZero);
  transformations->Add(new IdentifyL2Normalization);
  transformations->Add(new IdentifyL2Pool);
  transformations->Add(new IdentifyRelu1);
  transformations->Add(new IdentifyPRelu);
  transformations->Add(new RemoveTrivialBinaryOperator);
  transformations->Add(new ResolveFakeQuantArgsFromVars);
  transformations->Add(new ReadArrayMinmaxAndNarrowRangeFromFakeQuant);
  transformations->Add(new ResolveSpaceToBatchNDAttributes);
  transformations->Add(new ResolveBatchToSpaceNDAttributes);
  transformations->Add(new ResolvePadAttributes);
  transformations->Add(new ResolvePadV2Attributes);
  transformations->Add(new ResolveStridedSliceAttributes);
  transformations->Add(new ResolveSliceAttributes);
  transformations->Add(new ResolveReduceAttributes);
  transformations->Add(new ResolveConstantShapeOrRank);
  transformations->Add(new MakeInitialDequantizeOperator);
  transformations->Add(new UnpartitionEmbeddingLookup);
  transformations->Add(new ResolveGatherAttributes);
}

bool SupportsQuantization(FileFormat format) {
  return (format == GRAPHVIZ_DOT || format == TFLITE);
}

bool SupportsFusedActivationFunction(FileFormat format) {
  return (format == GRAPHVIZ_DOT || format == TFLITE);
}

bool SupportsLstmCell(FileFormat format) {
  return (format == TENSORFLOW_GRAPHDEF || format == GRAPHVIZ_DOT ||
          format == TFLITE);
}

bool SupportsPreallocatedWorkspace(FileFormat format) {
  return (format == TFLITE);
}

bool SupportsShuffledFCWeights(FileFormat format) { return format == TFLITE; }

bool IsRealValued(toco::ArrayDataType type) {
  // TODO(benoitjacob) - this is hardcoding that uint8 and int16 are only used
  // for quantized real-number values, and no other integer type is ever used
  // for that. This is dirty, should be resolved as part of a more general push
  // to more explicitly distinguish between true-integers and
  // integers used as quantized values representing real numbers.
  return static_cast<bool>(type == toco::ArrayDataType::kFloat ||
                           type == toco::ArrayDataType::kUint8 ||
                           type == toco::ArrayDataType::kInt16);
}

void SetFinalDataTypeOnInputs(const TocoFlags& toco_flags, Model* model) {
  const FileFormat output_format = toco_flags.output_format();
  ArrayDataType type;
  if (!SupportsQuantization(output_format)) {
    // Data type is implicitly float for non-quantized formats
    type = ArrayDataType::kFloat;
  } else if (toco_flags.has_inference_input_type()) {
    type = ConvertIODataTypeToArrayDataType(toco_flags.inference_input_type());
  } else if (toco_flags.has_inference_type()) {
    type = ConvertIODataTypeToArrayDataType(toco_flags.inference_type());
  } else {
    // Nothing to do. Data types stay as-is.
    return;
  }

  for (int i = 0; i < model->flags.input_arrays_size(); i++) {
    string const& array_name = model->flags.input_arrays(i).name();
    auto* array = &model->GetArray(array_name);
    // Note that the notion of changing data types only applies to real-numbers
    // arrays (see the documentation for inference_input_type).
    // TODO(benoitjacob) this is assuming that uint8 arrays are quantized,
    // i.e. represent real numbers by means of quantization parameters,
    // and not plain integer uint8 input arrays.
    if (!IsRealValued(array->data_type)) {
      // Ignore non-real data types.
      continue;
    }
    // The enum value QUANTIZED_UINT8 for --inference_type and
    // --inference_input_type has long meant just 'QUANTIZED', being used as
    // well in mixed 8-bit / 16-bit quantized models. However,
    // ConvertIODataTypeToArrayDataType still interpretes it as meaning 8bit,
    // and people have run into issues in the situation where they have an
    // already mixed 8-bit / 16-bit quantized model in TFLITE format and
    // want to run it again through toco, without having to re-specify all the
    // extra array info that was used in the (complicated) process of initially
    // quantizing that model. In order to have --inference_type=QUANTIZED_UINT8
    // just work in that case, we implement the logic that when an array is
    // already quantized, if  --inference_type is quantized (so we're not
    // asking to dequantize here), no change of quantized data type is to be
    // recorded.
    if (array->data_type != toco::ArrayDataType::kFloat &&
        type != toco::ArrayDataType::kFloat) {
      continue;
    }

    array->final_data_type = type;
  }
}

}  // namespace

std::unique_ptr<Model> Import(const TocoFlags& toco_flags,
                              const ModelFlags& model_flags,
                              const string& input_file_contents) {
  std::unique_ptr<Model> model;
  switch (toco_flags.input_format()) {
    case TENSORFLOW_GRAPHDEF: {
      TensorFlowImportFlags tf_import_flags;
      tf_import_flags.drop_control_dependency =
          toco_flags.has_drop_control_dependency()
              ? toco_flags.drop_control_dependency()
              : (toco_flags.output_format() != TENSORFLOW_GRAPHDEF);

      tf_import_flags.import_all_ops_as_unsupported =
          toco_flags.force_select_tf_ops();

      model = ImportTensorFlowGraphDef(model_flags, tf_import_flags,
                                       input_file_contents);
      break;
    }
    case TFLITE:
      model = toco::tflite::Import(model_flags, input_file_contents);
      ResolveModelFlags(model_flags, model.get());
      CheckInvariants(*model);
      break;
    default:
      LOG(FATAL) << "Unhandled input_format='"
                 << FileFormat_Name(toco_flags.input_format()) << "'";
  }

  LogDump(kLogLevelModelChanged, "AT IMPORT", *model);

  return model;
}

tensorflow::Status TransformWithStatus(const TocoFlags& toco_flags,
                                       Model* model) {
  const FileFormat output_format = toco_flags.output_format();
  const IODataType inference_type = toco_flags.inference_type();

  const bool quantize_output =
      SupportsQuantization(output_format) &&
      (inference_type == QUANTIZED_UINT8 || inference_type == QUANTIZED_INT16);

  if (quantize_output) {
    QCHECK_NE(toco_flags.inference_input_type(), FLOAT)
        << "Quantized inference is not allowed with float inputs.";
  }

  // Clean up after import.
  SetFinalDataTypeOnInputs(toco_flags, model);
  UseArraysExtraInfo(model, quantize_output);
  FinishBuildingRNNStates(model);

  // Remove unused ops before performing any other optimizations. This is to
  // stop optimizations from crossing the input/output boundaries. For example
  // this will stop BatchNorm fusing if the output node is in between a conv
  // and BatchNorm layers.
  TF_RETURN_IF_ERROR(RunGraphTransformationsWithStatus(
      model, "Removing unused ops", {new toco::RemoveUnusedOp}));

  GraphTransformationsSet transformations;
  MakeGeneralGraphTransformationsSet(&transformations);
  auto* remove_trivial_reshape = new RemoveTrivialReshape;
  transformations.Add(remove_trivial_reshape);
  auto* resolve_constant_fake_quant = new ResolveConstantFakeQuant;
  if (quantize_output) {
    resolve_constant_fake_quant->set_propagate_fake_quant_num_bits(
        toco_flags.propagate_fake_quant_num_bits());
  }
  transformations.Add(resolve_constant_fake_quant);
  if (SupportsFusedActivationFunction(output_format)) {
    transformations.Add(new FuseActivationFunctions);
  } else {
    transformations.Add(new UnfuseActivationFunctions);
  }
  if (toco_flags.drop_fake_quant()) {
    transformations.Add(new DropFakeQuant);
  } else {
    // See the doc for --reorder_across_fake_quant: that flag is needed to
    // support some existing models, e.g. WordLens, that have FakeQuant
    // nodes in the wrong places.
    // TODO(benoitjacob): drop special casing when we can.
    if ((quantize_output && toco_flags.reorder_across_fake_quant())) {
      transformations.Add(new DropFakeQuant);
    }
  }
  transformations.Add(new ConvertPureConvToDepthwise);
  if (SupportsLstmCell(output_format)) {
    if (!toco_flags.debug_disable_recurrent_cell_fusion()) {
      transformations.Add(new IdentifyLstmCell);
    }
    if (output_format == TFLITE && toco_flags.split_tflite_lstm_inputs()) {
      transformations.Add(new toco::SplitLstmCellInputs);
    } else {
      transformations.Add(new toco::MergeLstmCellInputs);
    }
  }
  transformations.Add(new ResolveConstantConcatenation);
  // TODO(b/116063589): TF GraphDef doesn't support dilations on its depthwise
  // conv, so we need to make sure we don't convert to dilated depthwise conv
  // when outputing to TF GraphDef.
  auto* identify_dilated_conv = new IdentifyDilatedConv;
  if (output_format == TENSORFLOW_GRAPHDEF) {
    identify_dilated_conv->set_identify_depthwise_conv(false);
  }
  transformations.Add(identify_dilated_conv);
  TF_RETURN_IF_ERROR(RunGraphTransformationsWithStatus(
      model, "general graph transformations", transformations));

  if (quantize_output) {
    if (toco_flags.propagate_fake_quant_num_bits()) {
      TF_RETURN_IF_ERROR(RunGraphTransformationsWithStatus(
          model, "fake quant propagation graph transformations",
          {new PropagateFakeQuantNumBits}));
    }
    TF_RETURN_IF_ERROR(RunGraphTransformationsWithStatus(
        model, "pre-quantization graph transformations",
        {
            new HardcodeMinMax,
            new DropFakeQuant,
        }));
  }

  // Try to merge bidirectional sequence lstm or rnn if present.
  GraphTransformationsSet bidirectional_transformations;
  bidirectional_transformations.Add(new RemoveUnusedOp);
  bidirectional_transformations.Add(new toco::GroupBidirectionalSequenceLstm);
  bidirectional_transformations.Add(new toco::GroupBidirectionalSequenceRnn);
  TF_RETURN_IF_ERROR(RunGraphTransformationsWithStatus(
      model, "Group bidirectional sequence lstm/rnn",
      bidirectional_transformations));

  // Fix any issues with IO edges. This must happen after any transform that
  // may modify the structure of the edges.
  FixEdgeArrays(model);
  FixOperatorOrdering(model);

  if (quantize_output) {
    // If the user specified default min/max ranges we need to set all arrays
    // that didn't either have a min/max specified or get one set via
    // HardcodeMinMax or PropagateFakeQuantNumBits. This may require running
    // HardcodeMinMax to move changes through the graph as we make changes.
    auto propagate_default_min_max =
        absl::make_unique<PropagateDefaultMinMax>();
    bool has_default_ranges_flag = (toco_flags.has_default_ranges_min() &&
                                    toco_flags.has_default_ranges_max());
    if (has_default_ranges_flag) {
      propagate_default_min_max->DefineTypeRange(
          ArrayDataType::kUint8, toco_flags.default_ranges_min(),
          toco_flags.default_ranges_max());
    }
    if (toco_flags.has_default_int16_ranges_min() &&
        toco_flags.has_default_int16_ranges_max()) {
      propagate_default_min_max->DefineTypeRange(
          ArrayDataType::kInt16, toco_flags.default_int16_ranges_min(),
          toco_flags.default_int16_ranges_max());
    }
    if (propagate_default_min_max->has_any_ranges_defined()) {
      TF_RETURN_IF_ERROR(RunGraphTransformationsWithStatus(
          model, "default min-max range propagation graph transformations",
          {
              propagate_default_min_max.release(),
              new HardcodeMinMax,
          }));
    }

    CheckIsReadyForQuantization(*model);
    auto* ensure_safe_for_int8_kernels =
        new EnsureUint8WeightsSafeForFastInt8Kernels;
    ensure_safe_for_int8_kernels->set_allow_nudging_weights(
        toco_flags.allow_nudging_weights_to_use_fast_gemm_kernel());
    ensure_safe_for_int8_kernels->set_has_default_ranges_flag(
        has_default_ranges_flag);
    TF_RETURN_IF_ERROR(RunGraphTransformationsWithStatus(
        model, "quantization graph transformations",
        {
            new RemoveTrivialQuantizedActivationFunc,
            new RemoveTrivialQuantizedMinMax,
            new Quantize,
            new RemoveFinalDequantizeOp,
            ensure_safe_for_int8_kernels,
        }));
    if (SupportsShuffledFCWeights(output_format)) {
      TF_RETURN_IF_ERROR(RunGraphTransformationsWithStatus(
          model, "shuffling of FC weights", {new ShuffleFCWeights}));
    }
  } else {
    GraphTransformationsSet dequantization_transformations{new Dequantize};
    // Dequantize creates FakeQuant nodes. We may want to discard
    // those immediately.
    if (toco_flags.drop_fake_quant()) {
      dequantization_transformations.Add(new DropFakeQuant);
    }

    TF_RETURN_IF_ERROR(RunGraphTransformationsWithStatus(
        model, "dequantization graph transformations",
        dequantization_transformations));
  }

  if (output_format == TENSORFLOW_GRAPHDEF) {
    EncodeConstantArraysMinMaxByWrappingThemInFakeQuantNodes(model);
  }

  // Deduplicate large constant arrays.
  DedupeConstantArrays(model, toco_flags.dedupe_array_min_size_bytes());

  LogDump(kLogLevelModelChanged, "AFTER TRANSFORMATIONS", *model);

  if (output_format != GRAPHVIZ_DOT && output_format != TFLITE) {
    // By now there shouldn't be any unsupported ops when exporting to
    // TensorFlow GraphDef.
    CheckUnsupportedOperations(*model);
  }

  if (SupportsPreallocatedWorkspace(output_format)) {
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
  model->ops_count = ops_count;
  return tensorflow::Status::OK();
}

tensorflow::Status Export(const TocoFlags& toco_flags, const Model& model,
                          bool allow_custom_ops, string* output_file_contents) {
  switch (toco_flags.output_format()) {
    case TENSORFLOW_GRAPHDEF:
      ExportTensorFlowGraphDef(model, output_file_contents);
      break;
    case TFLITE: {
      toco::tflite::ExportParams params;

      params.enable_select_tf_ops =
          toco_flags.force_select_tf_ops() || toco_flags.enable_select_tf_ops();
      params.allow_custom_ops = allow_custom_ops;
      params.quantize_weights = toco_flags.post_training_quantize();

      auto status = toco::tflite::Export(model, output_file_contents, params);
      if (!status.ok()) {
        LOG(ERROR) << status.error_message();
      }
      return status;
    } break;
    case GRAPHVIZ_DOT:
      DumpGraphviz(model, output_file_contents);
      break;
    default:
      LOG(FATAL) << "Unhandled output_format='"
                 << FileFormat_Name(toco_flags.output_format()) << "'";
  }
  return tensorflow::Status();
}

}  // namespace toco
