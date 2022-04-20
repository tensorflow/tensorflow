/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

#if IS_TRT_VERSION_GE(8, 2, 0, 0)

template <typename Impl>
class ConvertFillBase : public OpConverterBase<Impl> {
 public:
  explicit ConvertFillBase(OpConverterParams* params)
      : OpConverterBase<Impl>(params) {}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32};
  }

  Status ValidateFillBase(const OpConverterParams& params) {
    if (params.use_implicit_batch) {
      return errors::Unimplemented("Conversion for ", params.node_def.op(),
                                   " is not implemented in"
                                   " implicit batch mode");
    }
    return Status::OK();
  }

  Status AddFill(const TRT_TensorOrWeights& value_input,
                 const TRT_TensorOrWeights& dims_input, bool is_value_static,
                 bool is_dims_static, int nbDims,
                 const nvinfer1::Dims& trt_dims,
                 TRTNetworkBuilder* pBuilder = nullptr,
                 nvinfer1::ITensor* size_tensor = nullptr,
                 const float delta = 0, bool cast_to_integer = false) {
    const auto& params = *this->params_;
    auto* converter = params.converter;
    auto* network = converter->network();

    // TensorRT IFillLayer requires a rank 0 scalar.
    ITensorProxyPtr scalar_tensor, beta_tensor;
    nvinfer1::Dims scalar_dims;
    scalar_dims.nbDims = 0;
    if (is_value_static) {
      scalar_tensor =
          converter->CreateConstantLayer(value_input.weights(), scalar_dims);
    } else {
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          converter, size_tensor ? params.inputs.at(0) : value_input,
          scalar_dims, false, &scalar_tensor, params.node_def));
    }

    if (!pBuilder) {
      auto builder = TRTNetworkBuilder::Create(network, params.weight_store);
      pBuilder = &*builder;
    }

    nvinfer1::IFillLayer* layer =
        network->addFill(trt_dims, nvinfer1::FillOperation::kLINSPACE);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, params.node_def.name());

    if (!size_tensor) {
      nvinfer1::Dims beta_shape{1, {nbDims}};
      StatusOr<nvinfer1::IConstantLayer*> const_layer =
          pBuilder->Constant(delta, beta_shape, value_input.TrtDType());
      TF_RETURN_IF_ERROR(const_layer.status());
      beta_tensor = (*const_layer)->getOutput(0);
    } else {
      beta_tensor = params.inputs.at(2).tensor();
    }

    if (!is_dims_static) {
      layer->setInput(
          0, size_tensor ? *size_tensor : *dims_input.tensor()->trt_tensor());
    }

    layer->setInput(1, *scalar_tensor->trt_tensor());
    layer->setInput(2, *beta_tensor->trt_tensor());
    converter->SetLayerName(layer, params.node_def, "fill");

    ITensorProxyPtr output_tensor = layer->getOutput(0);
    if (cast_to_integer) {
      output_tensor->setType(nvinfer1::DataType::kINT32);
    }

    this->AddOutput(TRT_TensorOrWeights(output_tensor));
    return Status::OK();
  }
};

class ConvertFill : public ConvertFillBase<ConvertFill> {
 public:
  explicit ConvertFill(OpConverterParams* params)
      : ConvertFillBase<ConvertFill>(params) {}

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("dims", TrtInputArg::kBoth),
        InputArgSpec::Create("value", TrtInputArg::kBoth)};
  }

  Status Validate() {
    const auto& params = *this->params_;
    TF_RETURN_IF_ERROR(this->ValidateFillBase(params));

    const auto& inputs = params.inputs;
    const auto& node_def = params.node_def;
    const TRT_TensorOrWeights& dims_input = inputs.at(0);

    const auto dims_type = dims_input.TrtDType();
    if (dims_type != nvinfer1::DataType::kINT32) {
      return errors::InvalidArgument("The dims parameter of ", node_def.op(),
                                     " operation in ", node_def.name(),
                                     " is expected to be of type ",
                                     DebugString(nvinfer1::DataType::kINT32),
                                     " type, got ", DebugString(dims_type));
    }

    const auto nbDims = dims_input.GetTrtDims().nbDims;
    if (nbDims < 0) {
      return errors::InvalidArgument("The shape of parameter ", node_def.op(),
                                     " operation in ", node_def.name(),
                                     " cannot be partial.");
    }
    return Status::OK();
  }

  Status Convert() {
    const auto& params = *this->params_;
    const auto& inputs = params.inputs;

    const bool is_dims_static = inputs[0].is_weights();
    const bool is_value_static = inputs[1].is_weights();

    const TRT_TensorOrWeights& dims_input = inputs.at(0);
    const TRT_TensorOrWeights& value_input = inputs.at(1);

    nvinfer1::Dims trt_dims{0};
    if (is_dims_static) {
      const auto dims_weights = dims_input.weights();
      DimsAdapter dims_adapter(dims_weights.GetSpan<int32>());
      dims_adapter.TrtDims(&trt_dims);
    }

    return this->AddFill(value_input, dims_input, is_value_static,
                         is_dims_static, dims_input.GetTrtDims().d[0],
                         trt_dims);
  }
};

template <int V>
class ConvertLikeOps : public ConvertFillBase<ConvertLikeOps<V>> {
 public:
  explicit ConvertLikeOps(OpConverterParams* params)
      : ConvertFillBase<ConvertLikeOps<V>>(params) {}

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return std::array<InputArgSpec, 1>{
        InputArgSpec::Create("input", TrtInputArg::kBoth),
    };
  }

  Status Validate() { return this->ValidateFillBase(*this->params_); }

  Status Convert() {
    const auto& params = *this->params_;
    const auto& inputs = params.inputs;
    const TRT_TensorOrWeights& input = inputs.at(0);

    DimsAdapter value_input_dims(std::vector<int>{1});
    StatusOr<TRT_ShapedWeights> value_weights =
        params.weight_store->GetTempWeights(input.TrtDType(), value_input_dims);
    TF_RETURN_IF_ERROR(value_weights.status());
    TF_RETURN_IF_ERROR(value_weights->SetValues(V));
    TRT_TensorOrWeights value_input(value_weights.ValueOrDie());

    ITensorProxyPtr dims_input_tensor;
    nvinfer1::Dims dims = input.GetTrtDims();
    const bool is_dims_static = HasStaticShape(dims);
    TRTNetworkBuilder* pBuilder = nullptr;
    if (!is_dims_static) {
      auto builder = TRTNetworkBuilder::Create(params.converter->network(),
                                               params.weight_store);
      pBuilder = &*builder;
      StatusOr<nvinfer1::IShapeLayer*> shape_layer =
          builder->Shape(input.tensor()->trt_tensor());
      TF_RETURN_IF_ERROR(shape_layer.status());
      dims_input_tensor = (*shape_layer)->getOutput(0);
      dims = {0};
    }

    TRT_TensorOrWeights dims_input(dims_input_tensor);
    return this->AddFill(value_input, dims_input, true, is_dims_static,
                         input.GetTrtDims().nbDims, dims, pBuilder);
  }
};

class ConvertRange : public ConvertFillBase<ConvertRange> {
 public:
  explicit ConvertRange(OpConverterParams* params)
      : ConvertFillBase<ConvertRange>(params) {}

  static constexpr std::array<InputArgSpec, 3> InputSpec() {
    return std::array<InputArgSpec, 3>{
        InputArgSpec::Create("start", TrtInputArg::kBoth),
        InputArgSpec::Create("limit", TrtInputArg::kBoth),
        InputArgSpec::Create("delta", TrtInputArg::kBoth)};
  }

  static constexpr const char* NodeDefDataTypeAttributeName() { return ""; }
  Status Validate() {
    const auto& params = *this->params_;
    TF_RETURN_IF_ERROR(this->ValidateFillBase(params));

    const auto& inputs = params.inputs;
    const auto& node_def = params.node_def;

    if (!all_same_types(inputs)) {
      return errors::InvalidArgument(
          "All parameter (start, limit, delta) of ", node_def.op(),
          " operation in ", node_def.name(),
          " are expected to be passed as weights OR tensors");
    }

    if (!all_weights_) {
      return Status::OK();
    }

    float param[3];
    for (int i = 0; i < 3; i++) {
      const auto& input = inputs.at(i);
      switch (input.TrtDType()) {
        case nvinfer1::DataType::kFLOAT:
          param[i] = get_input_param<float>(input);
          break;
        case nvinfer1::DataType::kHALF:
          param[i] = get_input_param<Eigen::half>(input);
          break;
        default:  // nvinfer1::DataType::kINT32:
          param[i] = get_input_param<int>(input);
      }
    }

    if (param[2] == 0) {
      return errors::InvalidArgument("The delta parameter of ", node_def.op(),
                                     " operation cannot be equal to 0");
    }
    start_ = param[0];
    delta_ = param[2];
    const auto num_intervals_float = (param[1] - start_) / delta_;
    if (num_intervals_float < 0) {
      const auto error = convert_range_error_message(start_, param[1], delta_);
      return errors::InvalidArgument(error);
    }

    num_values_ = static_cast<int>(num_intervals_float);
    if (start_ + delta_ * num_values_ != param[1]) {
      num_values_++;
    }
    return Status::OK();
  }

  Status Convert() {
    const auto& params = *this->params_;
    const auto& inputs = params.inputs;
    const TRT_TensorOrWeights& input = inputs.at(0);

    DimsAdapter value_input_dims(std::vector<int>{1});
    StatusOr<TRT_ShapedWeights> value_weights =
        params.weight_store->GetTempWeights(input.TrtDType(), value_input_dims);

    TF_RETURN_IF_ERROR(value_weights.status());
    TF_RETURN_IF_ERROR(value_weights->SetValues(start_));
    TRT_TensorOrWeights value_input(value_weights.ValueOrDie());

    nvinfer1::Dims trt_dims{1};
    auto builder = TRTNetworkBuilder::Create(params.converter->network(),
                                             params.weight_store);
    TRT_ENSURE_OK(builder);
    nvinfer1::ITensor* size_tensor = nullptr;
    ITensorProxyPtr dims_input_tensor = nullptr;
    if (!all_weights_) {
      StatusOr<nvinfer1::IElementWiseLayer*> num =
          builder->Sub(/*limit*/ inputs.at(1).tensor()->trt_tensor(),
                       /*start*/ inputs.at(0).tensor()->trt_tensor());
      TRT_ENSURE_PTR_OK(num);
      StatusOr<nvinfer1::IElementWiseLayer*> ceil_div = builder->AbsCeilDivInt(
          (*num)->getOutput(0), inputs.at(2).tensor()->trt_tensor() /*delta*/);
      TRT_ENSURE_PTR_OK(ceil_div);
      size_tensor = (*ceil_div)->getOutput(0);
    } else {
      trt_dims.nbDims = 1;
      trt_dims.d[0] = num_values_;
      StatusOr<nvinfer1::IConstantLayer*> const_layer =
          builder->ConstantShape(value_input_dims);
      TRT_ENSURE_PTR_OK(const_layer);
      dims_input_tensor = (*const_layer)->getOutput(0);
    }

    const bool is_value_static = true;
    const bool is_dims_static = all_weights_;
    TRT_TensorOrWeights dims_input(dims_input_tensor);
    return this->AddFill(value_input, dims_input, is_value_static,
                         is_dims_static, 1, trt_dims, &*builder, size_tensor,
                         delta_, all_integers(inputs));
  }

 private:
  template <typename T>
  float get_input_param(const TRT_TensorOrWeights& input) {
    return static_cast<float>(*input.weights().GetPointer<T>());
  }

  bool all_integers(const std::vector<TRT_TensorOrWeights>& inputs) const {
    for (int i = 0; i < 3; i++) {
      if (inputs.at(i).TrtDType() != nvinfer1::DataType::kINT32) {
        return false;
      }
    }
    return true;
  }

  bool all_same_types(const std::vector<TRT_TensorOrWeights>& inputs) {
    auto i = inputs.size();
    const bool is_weight = inputs.at(--i).is_weights();
    while (i--) {
      if (inputs.at(i).is_weights() != is_weight) {
        return all_weights_ = false;
      }
    }
    all_weights_ = is_weight;
    return true;
  }

  float start_;
  float delta_;
  int num_values_;
  bool all_weights_;
};

std::string convert_range_error_message(float start, float limit, float delta) {
  const char* format_string = "For parameters (start, limit) = (%.2f, %.2f) "
      "of the Range operation delta cannot be %s, got %.2f";
  return absl::StrFormat(format_string, start, limit,
                         start < limit ? "negative" : "positive", delta);
}

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertFill>(), "Fill");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertLikeOps<0>>(),
                                  "zeros_like");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertLikeOps<1>>(),
                                  "ones_like");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertLikeOps<0>>(),
                                  "ZerosLike");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertLikeOps<1>>(),
                                  "OnesLike");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertRange>(),
                                  "Range");

#endif  // IS_TRT_VERSION_GE(8, 2, 0, 0)

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
