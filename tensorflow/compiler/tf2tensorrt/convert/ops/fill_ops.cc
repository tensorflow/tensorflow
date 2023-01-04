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
  explicit ConvertFillBase(const OpConverterParams* params)
      : OpConverterBase<Impl>(params, {DataType::DT_FLOAT, DataType::DT_HALF,
                                       DataType::DT_INT32}) {}
};

class ConvertFill : public ConvertFillBase<ConvertFill> {
 public:
  explicit ConvertFill(const OpConverterParams* params)
      : ConvertFillBase<ConvertFill>(params) {}

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("dims", TrtInputArg::kBoth),
        InputArgSpec::Create("value", TrtInputArg::kBoth)};
  }

  Status Validate() {
    const auto& params = *this->params_;
    TF_RETURN_IF_ERROR(NotSupportedInImplicitBatch());

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
    return OkStatus();
  }

  Status Convert() {
    const auto& params = *this->params_;
    auto* network = params.converter->network();
    const auto& inputs = params.inputs;

    const bool is_dims_static = inputs[0].is_weights();
    const bool is_value_static = inputs[1].is_weights();

    const TRT_TensorOrWeights& dims_input = inputs.at(0);
    const TRT_TensorOrWeights& value_input = inputs.at(1);

    int nbDims = dims_input.GetTrtDims().d[0];

    nvinfer1::Dims trt_dims{0};
    if (is_dims_static) {
      const auto dims_weights = dims_input.weights();
      DimsAdapter dims_adapter(dims_weights.GetSpan<int32>());
      dims_adapter.TrtDims(&trt_dims);
    }

    auto builder = TRTNetworkBuilder::Create(network, params.weight_store);
    StatusOr<nvinfer1::ILayer*> layer =
        builder->AddFill(value_input, dims_input, is_value_static,
                         is_dims_static, nbDims, trt_dims);
    ITensorProxyPtr output_tensor = (*layer)->getOutput(0);
    this->AddOutput(TRT_TensorOrWeights(output_tensor));
    return OkStatus();
  }
};

class ConvertRange : public ConvertFillBase<ConvertRange> {
 public:
  explicit ConvertRange(const OpConverterParams* params)
      : ConvertFillBase<ConvertRange>(params) {}

  static constexpr std::array<InputArgSpec, 3> InputSpec() {
    return std::array<InputArgSpec, 3>{
        InputArgSpec::Create("start", TrtInputArg::kBoth),
        InputArgSpec::Create("limit", TrtInputArg::kBoth),
        InputArgSpec::Create("delta", TrtInputArg::kBoth)};
  }

  static constexpr const char* NodeDefDataTypeAttributeName() {
    /*
    node {
      name: "..."
      op: "Range"
      ...
      attr {
        key: "Tidx"
        value {
          type: DT_INT32
        }
      }
    }
    */
    return "Tidx";
  }
  Status Validate() {
    TF_RETURN_IF_ERROR(NotSupportedInImplicitBatch());
    const auto& params = *this->params_;
    const auto& inputs = params.inputs;
    const auto& node_def = params.node_def;

    float param[3];
    all_weights_ = all_integers_ = true;
    for (int i = 0; i < 3; i++) {
      const auto& input = inputs.at(i);
      all_integers_ &= input.TrtDType() == nvinfer1::DataType::kINT32;
      if (input.is_weights()) {
        switch (input.TrtDType()) {
          case nvinfer1::DataType::kFLOAT:
            param[i] = get_input_param<float>(input);
            break;
          case nvinfer1::DataType::kHALF:
            param[i] = get_input_param<Eigen::half>(input);
            break;
          case nvinfer1::DataType::kINT32:
            param[i] = get_input_param<int>(input);
            break;
          default:
            return errors::InvalidArgument(
                "Unsupported data type ", DebugString(input.TrtDType()),
                " used for '", InputSpec()[i].name, "'");
        }
      } else {
        all_weights_ = false;
      }
    }

    if (!(all_weights_ || all_integers_)) {
      // As of 06/03/2022, when at least one of the (start, limit, delta)
      // is passed as a tensor, they must all be of type kINT32
      return errors::Unimplemented(convert_range_expected_msg(node_def));
    }

    if (inputs.at(2).is_weights()) {
      if ((delta_ = param[2]) == 0) {
        return errors::InvalidArgument("The delta parameter of ", node_def.op(),
                                       " operation cannot be equal to 0");
      }

      if (!all_weights_ && delta_ < 0) {
        return errors::InvalidArgument(
            "The delta parameter of Range operation "
            "cannot be negative, when one of (start, limit) is passed as "
            "a tensor, but got ",
            delta_);
      }
    }

    for (int i = 0; i < 3; i++) {
      const auto& input = inputs.at(i);
      const auto& dims = input.GetTrtDims();
      if (dims.nbDims != 1 || dims.d[0] != 1) {
        return errors::InvalidArgument("Dimension for '", InputSpec()[i].name,
                                       "' of ", node_def.op(), " operator ",
                                       "should be equal to 1");
      }
    }

    if (all_weights_) {
      const auto num_intervals_float =
          (param[1] - (start_ = param[0])) / delta_;
      if (num_intervals_float < 0) {
        const auto error = convert_range_error_msg(start_, param[1], delta_);
        return errors::InvalidArgument(error);
      }

      num_values_ = static_cast<int>(num_intervals_float);
      if (start_ + delta_ * num_values_ != param[1]) {
        num_values_++;
      }
    }

    return OkStatus();
  }

  Status Convert() {
    const auto& params = *this->params_;
    const auto& inputs = params.inputs;
    const TRT_TensorOrWeights& input = inputs.at(0);
    TRT_TensorOrWeights value_input;
    nvinfer1::Dims trt_dims{1};
    auto builder = TRTNetworkBuilder::Create(params.converter->network(),
                                             params.weight_store);
    TRT_ENSURE_OK(builder);
    ITensorProxyPtr dims_input_tensor = nullptr;
    ITensorProxyPtr beta_tensor = nullptr;
    ITensorProxyPtr scalar_tensor = nullptr;
    if (!all_weights_) {
      ITensorProxyPtr tensors[3];
      for (int i = 0; i < 3; i++) {
        TF_RETURN_IF_ERROR(
            builder->get_tensor4TensorOrWeights(inputs.at(i), tensors + i));
      }

      StatusOr<nvinfer1::IElementWiseLayer*> num =
          builder->Sub(/*limit*/ tensors[1]->trt_tensor(),
                       /*start*/ tensors[0]->trt_tensor());

      TRT_ENSURE_PTR_OK(num);
      StatusOr<nvinfer1::IElementWiseLayer*> ceil_div = builder->FloorDiv(
          (*num)->getOutput(0), (beta_tensor = tensors[2])->trt_tensor());
      TRT_ENSURE_PTR_OK(ceil_div);
      dims_input_tensor = (*ceil_div)->getOutput(0);
      dims_input_tensor->setType(nvinfer1::DataType::kINT32);

      nvinfer1::Dims scalar_dims{0};
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params.converter, params.inputs.at(0), scalar_dims, false,
          &scalar_tensor, params.node_def));
    } else {
      DimsAdapter value_input_dims(std::vector<int>{1});
      StatusOr<TRT_ShapedWeights> value_weights =
          params.weight_store->GetTempWeights(input.TrtDType(),
                                              value_input_dims);

      TF_RETURN_IF_ERROR(value_weights.status());
      TF_RETURN_IF_ERROR(value_weights->SetValues(start_));
      value_input = TRT_TensorOrWeights(value_weights.value());

      trt_dims.d[0] = num_values_;
      StatusOr<nvinfer1::IConstantLayer*> const_layer =
          builder->ConstantShape(value_input_dims);
      TRT_ENSURE_PTR_OK(const_layer);
      dims_input_tensor = (*const_layer)->getOutput(0);
    }

    TRT_TensorOrWeights dims_input(dims_input_tensor);

    StatusOr<nvinfer1::ILayer*> layer =
        builder->AddFill(value_input, dims_input, all_weights_, all_weights_, 1,
                         trt_dims, scalar_tensor, beta_tensor, delta_);

    ITensorProxyPtr output_tensor = (*layer)->getOutput(0);
    if (all_integers_) {
      output_tensor->setType(nvinfer1::DataType::kINT32);
    }

    this->AddOutput(TRT_TensorOrWeights(output_tensor));
    return OkStatus();
  }

 private:
  template <typename T>
  float get_input_param(const TRT_TensorOrWeights& input) {
    return static_cast<float>(*input.weights().GetPointer<T>());
  }

  float start_;
  float delta_;
  int num_values_;
  bool all_weights_;
  bool all_integers_;
};

std::string convert_range_error_msg(float start, float limit, float delta) {
  constexpr char* format_string =
      "For parameters (start, limit) = (%.2f, %.2f) "
      "of the Range operation delta cannot be %s, got %.2f";
  return absl::StrFormat(format_string, start, limit,
                         start < limit ? "negative" : "positive", delta);
}

std::string convert_range_expected_msg(const NodeDef& node_def) {
  return "When at least one of parameters (start, limit, delta) of " +
         node_def.op() + " operation in " + node_def.name() +
         " is passed as a tensor, they must all be of type kINT32";
}

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertFill>(), "Fill");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertRange>(),
                                  "Range");

#endif  // IS_TRT_VERSION_GE(8, 2, 0, 0)

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
