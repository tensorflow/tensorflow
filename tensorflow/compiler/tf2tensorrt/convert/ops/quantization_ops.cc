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

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/convert/ops/quantization_ops.h"

#include "absl/strings/str_format.h"
#include "tensorflow/cc/ops//array_ops.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/weights.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

bool IsQuantizeAndDequantizeOp(const Node* node) {
  return absl::c_find(kQuantizationOpNames, node->def().op()) !=
         kQuantizationOpNames.end();
}

namespace {

// Provides quantizing and dequantizing tensor scales for a given dynamic range.
// Borrowed from TF quantization kernel logic.
template <typename T>
QuantizationScales<T, 1> ComputeQuantizationRange(bool signed_input,
                                                  int num_bits,
                                                  bool narrow_range,
                                                  T* min_range, T* max_range) {
  // Calculate the range for the simulated integer quantization:
  // e.g. [-127,127] for signed = true, narrow_range = true, num_bits = 8,
  // or [-128,127] for signed = true, narrow_range = false, num_bits = 8,
  // or [0, 255] for signed = false, num_bits = 8.
  const int64_t min_quantized =
      signed_input ? narrow_range ? -(1ULL << (num_bits - 1)) + 1
                                  : -(1ULL << (num_bits - 1))
                   : 0;
  const int64_t max_quantized =
      signed_input ? (1ULL << (num_bits - 1)) - 1 : (1ULL << num_bits) - 1;
  // Determine the maximum scaling factor that would scale
  // [min_range, max_range] to not exceed [min_quantized, max_quantized],
  // while keeping 0 unchanged.
  const T scale_from_min_side = (min_quantized * *min_range > 0)
                                    ? min_quantized / *min_range
                                    : std::numeric_limits<T>::max();
  const T scale_from_max_side = (max_quantized * *max_range > 0)
                                    ? max_quantized / *max_range
                                    : std::numeric_limits<T>::max();

  QuantizationScales<T, 1> scales;
  // Note: Avoids changing the side of the range that determines scale.
  if (scale_from_min_side < scale_from_max_side) {
    scales.quantize_scale[0] = scale_from_min_side;
    scales.dequantize_scale[0] = *min_range / min_quantized;
    *max_range = max_quantized * scales.dequantize_scale[0];
  } else {
    scales.quantize_scale[0] = scale_from_max_side;
    scales.dequantize_scale[0] = *max_range / max_quantized;
    *min_range = min_quantized * scales.dequantize_scale[0];
  }
  return scales;
}

// Prepares the input for a QDQ node in explicit precision mode, returning a
// ITensor pointer. If the input is weights, we convert it to a ITensor by
// adding a constant layer.
StatusOr<nvinfer1::ITensor*> ExlicitQDQInputToTensor(
    TRTNetworkBuilder* builder, const OpConverterParams* params,
    const TRT_TensorOrWeights& input) {
  if (input.is_tensor()) {
    return input.tensor()->trt_tensor();
  }
  if (!IS_TRT_VERSION_GE(8, 0, 0, 0) && input.weights().count() > 1) {
    LOG(WARNING) << absl::StrCat(
        "QDQ per-channel for weights not "
        "implemented, assuming uniform scaling");
  }
  TRT_ShapedWeights trt_weights = input.weights();
  StatusOr<nvinfer1::IConstantLayer*> weights_const =
      builder->WeightsToConstant(trt_weights.GetTrtWeights(),
                                 trt_weights.Shape());
  TRT_ENSURE_PTR_OK(weights_const);
  params->converter->SetLayerName(*weights_const, params->node_def, "const");
  nvinfer1::ITensor* qdq_input = (*weights_const)->getOutput(0);
  std::string name = absl::StrCat((*weights_const)->getName(), "_output");
  qdq_input->setName(name.c_str());
  return qdq_input;
}

}  // namespace

// Carries traits for each specific quantization op type for conversion.
// Specialization for template parameter T should be given for each TF C++
// quantization op.
template <typename T>
struct QDQOpSpec {};

template <>
struct QDQOpSpec<ops::QuantizeAndDequantizeV2> {
  static constexpr std::array<InputArgSpec, 3> InputSpec() {
    return {
        InputArgSpec::Create("input", TrtInputArg::kBoth),
        InputArgSpec::Create("input_min", TrtInputArg::kWeight),
        InputArgSpec::Create("input_max", TrtInputArg::kWeight),
    };
  }

  struct Attrs {
    float min_range;
    float max_range;
    bool narrow_range;
    std::string round_mode;
    UniformQuantizationScales scales;
  };

  static Status ValidateQDQForExplicitPrecision(
      const std::vector<TRT_TensorOrWeights>& inputs, const NodeDef& node_def,
      Attrs* args) {
    AttrSlice attrs(node_def);
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "round_mode", &args->round_mode));
    if (args->round_mode != "HALF_TO_EVEN") {
      LOG(WARNING) << node_def.op() << ": " << node_def.name()
                   << " has round_mode=" << args->round_mode
                   << ", but for TensorRT conversion, "
                      "round_mode=HALF_TO_EVEN is recommended.";
    }
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "narrow_range", &args->narrow_range));
    if (args->narrow_range) {
      LOG(WARNING) << node_def.op() << ": " << node_def.name()
                   << " has narrow_range=true, but for TensorRT conversion, "
                      "narrow_range=false is recommended.";
    }
    args->min_range = inputs.at(1).weights().template GetPointer<float>()[0];
    args->max_range = inputs.at(2).weights().template GetPointer<float>()[0];
    const int num_bits = 8;
    args->scales = ComputeQuantizationRange<float>(
        /*signed_input=*/true, num_bits, args->narrow_range, &args->min_range,
        &args->max_range);
    TRT_ENSURE(args->scales.dequantize_scale[0] != 0);
    TRT_ENSURE(args->scales.quantize_scale[0] != 0);
    return OkStatus();
  }

  // Converts in explicit precision mode. In this mode, QDQ operations are
  // directly converted into TensorRT quantizing and dequantizing scale
  // operations.
  static Status ConvertExplicit(const OpConverterParams* params,
                                const Attrs& args) {
    const auto& node_def = params->node_def;

    StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
        params->converter->network(), params->weight_store);

    StatusOr<nvinfer1::ITensor*> qdq_input =
        ExlicitQDQInputToTensor(&*builder, params, params->inputs.at(0));
    TRT_ENSURE_PTR_OK(qdq_input);

    // TODO(cbate): check this condition exists for TRT8? Outline this block to
    // a "reshape policy".
    const int required_dims = params->use_implicit_batch ? 3 : 4;
    const nvinfer1::Dims idims = (*qdq_input)->getDimensions();
    nvinfer1::Dims intermediate_dims = idims;
    TRT_ENSURE(idims.nbDims > 0);
    if (idims.nbDims < required_dims) {
      const int nb_extra_dims = required_dims - idims.nbDims;
      intermediate_dims.nbDims = required_dims;
      std::vector<int> ones(nb_extra_dims, 1);
      TRT_ENSURE(ones.size() == nb_extra_dims && nb_extra_dims > 0);

      if (!params->use_implicit_batch) {
        intermediate_dims.d[0] = idims.d[0];
        std::copy(ones.begin(), ones.end(), intermediate_dims.d + 1);
        std::copy_n(idims.d + 1, idims.nbDims - 1,
                    intermediate_dims.d + ones.size() + 1);
      } else {
        std::copy(ones.begin(), ones.end(), intermediate_dims.d);
        std::copy_n(idims.d, idims.nbDims, intermediate_dims.d + ones.size());
      }

      LOG(WARNING) << absl::StrCat(
          node_def.name(), ":", node_def.op(), ": tensor ",
          (*qdq_input)->getName(), " has shape ", DebugString(idims),
          " but TRT scale layer requires at least 3 dims excluding batch dim, "
          "trying to recover by inserting 1's to create shape ",
          DebugString(intermediate_dims));
      StatusOr<nvinfer1::IShuffleLayer*> reshape =
          builder->Reshape(*qdq_input, intermediate_dims);
      TRT_ENSURE_PTR_OK(reshape);
      *qdq_input = (*reshape)->getOutput(0);
    }

    VLOG(1) << "[ExplicitPrecision]" << node_def.op() << ": " << node_def.name()
            << " computed scales: " << args.scales << " from min/max ranges "
            << args.min_range << "/" << args.max_range;

    StatusOr<nvinfer1::ILayer*> qdq =
        builder->UniformQuantizeDequantizeExplicit(
            *qdq_input, args.scales.quantize_scale[0],
            args.scales.dequantize_scale[0], node_def.name());
    TRT_ENSURE_PTR_OK(qdq);
    ITensorProxyPtr final_output = (*qdq)->getOutput(0);
    if (idims.nbDims != intermediate_dims.nbDims) {
      StatusOr<nvinfer1::IShuffleLayer*> undo_reshape =
          builder->Reshape(*qdq_input, idims);
      TRT_ENSURE_PTR_OK(undo_reshape);
      final_output = (*undo_reshape)->getOutput(0);
    }
    params->outputs->push_back(final_output);
    return OkStatus();
  }
};

template <>

struct QDQOpSpec<ops::QuantizeAndDequantizeV3> {
  static constexpr std::array<InputArgSpec, 4> InputSpec() {
    return {
        InputArgSpec::Create("input", TrtInputArg::kBoth),
        InputArgSpec::Create("min", TrtInputArg::kWeight),
        InputArgSpec::Create("max", TrtInputArg::kWeight),
        InputArgSpec::Create("num_bits", TrtInputArg::kWeight),
    };
  }
  // Use same attributes and conversion functions as QDQV2.
  using Attrs = QDQOpSpec<ops::QuantizeAndDequantizeV2>::Attrs;

  static Status ValidateQDQForExplicitPrecision(
      const std::vector<TRT_TensorOrWeights>& inputs, const NodeDef& node_def,
      Attrs* args) {
    return QDQOpSpec<
        ops::QuantizeAndDequantizeV2>::ValidateQDQForExplicitPrecision(inputs,
                                                                       node_def,
                                                                       args);
  }

  static Status ConvertExplicit(const OpConverterParams* params,
                                const Attrs& args) {
    return QDQOpSpec<ops::QuantizeAndDequantizeV2>::ConvertExplicit(params,
                                                                    args);
  }
};

template <>

struct QDQOpSpec<ops::FakeQuantWithMinMaxVars> {
  static constexpr std::array<InputArgSpec, 3> InputSpec() {
    return {
        InputArgSpec::Create("input", TrtInputArg::kBoth),
        InputArgSpec::Create("min", TrtInputArg::kWeight),
        InputArgSpec::Create("max", TrtInputArg::kWeight),
    };
  }
  struct Attrs {
    int num_bits;
    bool narrow_range;
    float min_range;
    float max_range;
    std::string round_mode;
    UniformQuantizationScales scales;
  };

  static Status ValidateQDQForExplicitPrecision(
      const std::vector<TRT_TensorOrWeights>& inputs, const NodeDef& node_def,
      Attrs* args) {
    AttrSlice attrs(node_def);
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "narrow_range", &args->narrow_range));
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "num_bits", &args->num_bits));
    if (args->narrow_range) {
      return errors::Unimplemented("FakeQuantWithMinMaxVars is not supported ",
                                    "with `narrow_range=True`.");
    }
    // FakeQuantWithMinMaxVars does not take round_mode as an input, hard code
    // it to TRT's reccomended round mode.
    args->round_mode = "HALF_TO_EVEN";
    const float min_range = inputs.at(1).weights().template GetPointer<float>()[0];
    const float max_range = inputs.at(2).weights().template GetPointer<float>()[0];
    TRT_ENSURE(min_range < 0);
    TRT_ENSURE(max_range > 0);
    // Adjust the min & max range be perfectly symmetric.
    const float range = std::max(abs(min_range), abs(max_range));
    args->min_range = range * -1.0f;
    args->max_range = range;
    args->scales = ComputeQuantizationRange<float>(
        /*signed_input=*/true, args->num_bits, args->narrow_range, &args->min_range,
        &args->max_range);
    TRT_ENSURE(args->scales.dequantize_scale[0] != 0);
    TRT_ENSURE(args->scales.quantize_scale[0] != 0);
    return OkStatus();
  }

  static Status ConvertExplicit(const OpConverterParams* params,
                                const Attrs& args) {
    const auto& node_def = params->node_def;
    StatusOr<TRTNetworkBuilder> builder = TRTNetworkBuilder::Create(
        params->converter->network(), params->weight_store);

    StatusOr<nvinfer1::ITensor*> qdq_input =
        ExlicitQDQInputToTensor(&*builder, params, params->inputs.at(0));
    TRT_ENSURE_PTR_OK(qdq_input);

    const int required_dims = params->use_implicit_batch ? 3 : 4;
    const nvinfer1::Dims idims = (*qdq_input)->getDimensions();
    nvinfer1::Dims intermediate_dims = idims;

    auto actual_dims = params->inputs.at(0).GetTrtDims();
    TRT_ENSURE(idims.nbDims > 0);
    if (idims.nbDims < required_dims) {
      const int nb_extra_dims = required_dims - idims.nbDims;
      intermediate_dims.nbDims = required_dims;
      std::vector<int> ones(nb_extra_dims, 1);
      TRT_ENSURE(ones.size() == nb_extra_dims && nb_extra_dims > 0);

      if (!params->use_implicit_batch) {
        intermediate_dims.d[0] = idims.d[0];
        std::copy(ones.begin(), ones.end(), intermediate_dims.d + 1);
        std::copy_n(idims.d + 1, idims.nbDims - 1,
                    intermediate_dims.d + ones.size() + 1);
      } else {
        std::copy(ones.begin(), ones.end(), intermediate_dims.d);
        std::copy_n(idims.d, idims.nbDims, intermediate_dims.d + ones.size());
      }

      LOG(WARNING) << absl::StrCat(
          node_def.name(), ":", node_def.op(), ": tensor ",
          (*qdq_input)->getName(), " has shape ", DebugString(idims),
          " but TRT scale layer requires at least 3 dims excluding batch dim, "
          "trying to recover by inserting 1's to create shape ",
          DebugString(intermediate_dims));  
      StatusOr<nvinfer1::IShuffleLayer*> reshape =
          builder->Reshape(*qdq_input, intermediate_dims);
      TRT_ENSURE_PTR_OK(reshape);
      *qdq_input = (*reshape)->getOutput(0);
    }

    VLOG(1) << "[ExplicitPrecision]" << node_def.op() << ": " << node_def.name()
            << " computed scales: " << args.scales << " from min/max ranges "
            << args.min_range << "/" << args.max_range;

    StatusOr<nvinfer1::ILayer*> qdq =
        builder->UniformQuantizeDequantizeExplicit(
            *qdq_input, args.scales.quantize_scale[0],
            args.scales.dequantize_scale[0], node_def.name());
    TRT_ENSURE_PTR_OK(qdq);
    ITensorProxyPtr final_output = (*qdq)->getOutput(0);
    if (idims.nbDims != intermediate_dims.nbDims) {
      StatusOr<nvinfer1::IShuffleLayer*> undo_reshape =
          builder->Reshape(*qdq_input, idims);
      TRT_ENSURE_PTR_OK(undo_reshape);
      final_output = (*undo_reshape)->getOutput(0);
    }
    params->outputs->push_back(final_output);
    return OkStatus();
  }
};

template <>

struct QDQOpSpec<ops::FakeQuantWithMinMaxArgs> {
  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return {
        InputArgSpec::Create("input", TrtInputArg::kBoth),
    };
  }

  struct Attrs {
    float min;
    float max;
    int num_bits;
    bool narrow_range;
  };

  static Status ValidateQDQForExplicitPrecision(
      const std::vector<TRT_TensorOrWeights>& inputs, const NodeDef& node_def,
      Attrs* args) {
    return errors::Unimplemented("");
  }

  static Status ConvertExplicit(const OpConverterParams* params,
                                const Attrs& args) {
    return errors::Unimplemented("");
  }
};

// Converts QDQ operations in non-explicit precision mode. This is the original
// "ConvertQuantize" function. In this mode, Q/DQ operations are no-ops and are
// instead used to set the dynamic range of the input tensor.
Status ConvertDynamicRangeMode(const OpConverterParams* params) {
  const auto& inputs = params->inputs;
  const auto& node_def = params->node_def;
  float min_range = 0.0f;
  float max_range = 0.0f;
  const auto& op_name = node_def.op();
  if (op_name == "FakeQuantWithMinMaxArgs") {
    AttrSlice attrs(node_def);
    // Get ranges via node attributes.
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "min", &min_range));
    TF_RETURN_IF_ERROR(GetNodeAttr(attrs, "max", &max_range));
  } else if (op_name == "FakeQuantWithMinMaxVars" ||
             op_name == "QuantizeAndDequantizeV2" ||
             op_name == "QuantizeAndDequantizeV3") {
    // Get ranges via inputs.
    auto get_weights_value = [&inputs](int index) {
      const auto* raw_weights = inputs.at(index).weights().GetPointer<float>();
      return raw_weights[0];
    };
    min_range = get_weights_value(1);
    max_range = get_weights_value(2);
  } else {
    return errors::InvalidArgument("Unknown quantization op ", op_name, ", at ",
                                   node_def.name());
  }
  if (params->validation_only) {
    return OkStatus();
  }

  // Store ranges for tensor
  ITensorProxyPtr input0 = inputs.at(0).tensor();
  params->converter->ProvideQuantizationRange(&input0, min_range, max_range);
  // Sometimes, TRT may not quantize a tensor, either because it chooses to
  // execute a higher precision kernel or because of op fusion. In these
  // cases, accuracy will suffer if the model was trained to expect
  // quantization at that tensor. We should consider adding a clip(tensor,
  // min_range, max_range) operation here to ensure that any arbitrarily
  // placed quantize node will execute as expected. However, this will
  // negatively affect performance. If users train their models in a way which
  // models inference as close as possible (i.e. not quantizing in place where
  // fusion will occur), then there is no problem with the current
  // implementation.
  params->outputs->push_back(inputs.at(0));
  return OkStatus();
}

template <typename TFOpType>
class ConvertQDQ : public OpConverterBase<ConvertQDQ<TFOpType>> {
 public:
  explicit ConvertQDQ(const OpConverterParams* params)
      : OpConverterBase<ConvertQDQ<TFOpType>>(params) {}

  static constexpr auto InputSpec() { return QDQOpSpec<TFOpType>::InputSpec(); }

  // Disable the non-applicable data type check by providing empty string.
  static constexpr const char* NodeDefDataTypeAttributeName() { return ""; }

  Status ValidateDynamicRangeINT8Mode() {
    // The condition ensures we only call the conversion once. We should break
    // this function up into validation and conversion.
    if (this->params_->validation_only) {
      return ConvertDynamicRangeMode(this->params_);
    }
    return OkStatus();
  }

  Status Validate() {
    if (!this->params_->use_explicit_precision) {
      return ValidateDynamicRangeINT8Mode();
    }
    return OpSpec::ValidateQDQForExplicitPrecision(
        this->params_->inputs, this->params_->node_def, &attrs_);
  }

  Status Convert() {
    if (!this->params_->use_explicit_precision) {
      return ConvertDynamicRangeMode(this->params_);
    }
    return OpSpec::ConvertExplicit(this->params_, attrs_);
  }

  using OpSpec = QDQOpSpec<TFOpType>;
  using OpSpecAttrs = typename QDQOpSpec<TFOpType>::Attrs;
  OpSpecAttrs attrs_;
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertQDQ<ops::QuantizeAndDequantizeV2>>(),
    "QuantizeAndDequantizeV2");
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertQDQ<ops::QuantizeAndDequantizeV3>>(),
    "QuantizeAndDequantizeV3");
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertQDQ<ops::FakeQuantWithMinMaxVars>>(),
    "FakeQuantWithMinMaxVars");
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertQDQ<ops::FakeQuantWithMinMaxArgs>>(),
    "FakeQuantWithMinMaxArgs");

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
