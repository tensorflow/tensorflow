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

#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

const UnaryOperationMapType* UnaryOperationMap() {
  static auto* const m = new UnaryOperationMapType({
    {"Exp", nvinfer1::UnaryOperation::kEXP},
        {"Log", nvinfer1::UnaryOperation::kLOG},
        {"Sqrt", nvinfer1::UnaryOperation::kSQRT},
        {"Rsqrt", nvinfer1::UnaryOperation::kSQRT},
        {"Reciprocal", nvinfer1::UnaryOperation::kRECIP},
        {"Abs", nvinfer1::UnaryOperation::kABS},
        {"Neg", nvinfer1::UnaryOperation::kNEG},
        {"Sin", nvinfer1::UnaryOperation::kSIN},
        {"Cos", nvinfer1::UnaryOperation::kCOS},
        {"Tan", nvinfer1::UnaryOperation::kTAN},
        {"Sinh", nvinfer1::UnaryOperation::kSINH},
        {"Cosh", nvinfer1::UnaryOperation::kCOSH},
        {"Asin", nvinfer1::UnaryOperation::kASIN},
        {"Acos", nvinfer1::UnaryOperation::kACOS},
        {"Atan", nvinfer1::UnaryOperation::kATAN},
        {"Asinh", nvinfer1::UnaryOperation::kASINH},
        {"Acosh", nvinfer1::UnaryOperation::kACOSH},
        {"Atanh", nvinfer1::UnaryOperation::kATANH},
        {"Ceil", nvinfer1::UnaryOperation::kCEIL},
        {"Floor", nvinfer1::UnaryOperation::kFLOOR},
        {"Erf", nvinfer1::UnaryOperation::kERF},
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
        {"Round", nvinfer1::UnaryOperation::kROUND},
        {"Sign", nvinfer1::UnaryOperation::kSIGN},
#endif
  });
  return m;
}

const UnaryOperationMapType* UnaryBooleanOperationMap() {
  static auto* const m = new UnaryOperationMapType({
      {"LogicalNot", nvinfer1::UnaryOperation::kNOT},
  });
  return m;
}

const ActivationTypeMapType* ActivationTypeMap() {
  static auto* const m = new ActivationTypeMapType({
      {"LeakyRelu", nvinfer1::ActivationType::kLEAKY_RELU},
      {"Relu", nvinfer1::ActivationType::kRELU},
      {"Relu6", nvinfer1::ActivationType::kCLIP},
      {"Sigmoid", nvinfer1::ActivationType::kSIGMOID},
      {"Tanh", nvinfer1::ActivationType::kTANH},
      {"Elu", nvinfer1::ActivationType::kELU},
      {"Selu", nvinfer1::ActivationType::kSELU},
      {"Softsign", nvinfer1::ActivationType::kSOFTSIGN},
      {"Softplus", nvinfer1::ActivationType::kSOFTPLUS},
  });
  return m;
}

template <typename T>
class ConvertUnaryImpl {
 protected:
  ConvertUnaryImpl(const OperationMap<T>* pOperMap) : pOperMap_(pOperMap) {}

  Status ValidateImpl(const OpConverterParams& params,
                      const std::vector<string>& not_supported_ops = {}) {
    const auto& node = params.node_def;
    const auto& op = node.op();
    if (pOperMap_->find(op) == pOperMap_->end()) {
      return errors::Unimplemented("Unary op: ", op, " not supported");
    }
    DimsAdapter input_dims(params.inputs.at(0).GetTrtDims());
    if (!input_dims.NumDims()) {
      return errors::InvalidArgument(
          "At least 1 dimension is required for UNARY operation '", op, "'");
    }

    if (!not_supported_ops.empty() && params.use_implicit_batch) {
      const auto& end = not_supported_ops.end();
      if (std::find(not_supported_ops.begin(), end, op) != end) {
        const auto& err =
            convert_not_supported_implicit(op, node.name(), "Unary");
        return errors::Unimplemented(err);
      }
    }

    return OkStatus();
  }

  Status ConvertImpl(const OpConverterParams& params) {
    const auto& node_def = params.node_def;
    auto* converter = params.converter;
    const auto op_pair = pOperMap_->find(node_def.op());
    ITensorProxyPtr tensor = params.inputs.at(0).tensor();
    nvinfer1::IUnaryLayer* layer =
        converter->network()->addUnary(*tensor->trt_tensor(), op_pair->second);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    converter->SetLayerName(layer, node_def);
    if (node_def.op() == "Rsqrt") {
      layer = converter->network()->addUnary(*layer->getOutput(0),
                                             nvinfer1::UnaryOperation::kRECIP);
      TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
      converter->SetLayerName(layer, node_def, "recip");
    }
    params.outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
    return OkStatus();
  }
  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return std::array<InputArgSpec, 1>{
        InputArgSpec::Create("x", TrtInputArg::kTensor)};
  }

 protected:
  const OperationMap<T>* pOperMap_;
};

class ConvertUnary : public OpConverterBase<ConvertUnary>,
                     protected ConvertUnaryImpl<nvinfer1::UnaryOperation> {
 public:
  explicit ConvertUnary(const OpConverterParams* params)
      : OpConverterBase<ConvertUnary>(
            params,
            params->node_def.op() == "Sign"
                ? std::vector<DataType>{DataType::DT_FLOAT, DataType::DT_HALF,
                                        DataType::DT_INT8, DT_INT32}
                : std::vector<DataType>{DataType::DT_FLOAT, DataType::DT_HALF,
                                        DataType::DT_INT8}),
        ConvertUnaryImpl(UnaryOperationMap()) {}

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return ConvertUnaryImpl::InputSpec();
  }

  Status Validate() { return ValidateImpl(*params_, {"Sign", "Round"}); }
  Status Convert() { return ConvertImpl(*params_); }
};

class ConvertBooleanUnary : public OpConverterBase<ConvertBooleanUnary>,
                            public ConvertUnaryImpl<nvinfer1::UnaryOperation> {
 public:
  explicit ConvertBooleanUnary(const OpConverterParams* params)
      : OpConverterBase<ConvertBooleanUnary>(params, {DataType::DT_BOOL}),
        ConvertUnaryImpl(UnaryBooleanOperationMap()) {}

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return ConvertUnaryImpl::InputSpec();
  }

  static constexpr const char* NodeDefDataTypeAttributeName() {
    /*
    node {
      name: "..."
      op: "LogicalNot"
      input: "..."
    }
    */
    return "";
  }
  Status Validate() {
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    return ValidateImpl(*params_, {"LogicalNot"});
#else
    return errors::Unimplemented("Boolean op: ", params_->node_def.op(),
                                 " is not supported in TRT version < 8.2");
#endif
  }
  Status Convert() { return ConvertImpl(*params_); }
};

class ConvertActivation : public OpConverterBase<ConvertActivation>,
                          protected ConvertUnaryImpl<nvinfer1::ActivationType> {
 public:
  explicit ConvertActivation(const OpConverterParams* params)
      : OpConverterBase<ConvertActivation>(params),
        ConvertUnaryImpl(ActivationTypeMap()) {}

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return std::array<InputArgSpec, 1>{
        InputArgSpec::Create("input", TrtInputArg::kTensor)};
  }

  Status Validate() {
    TF_RETURN_IF_ERROR(ValidateImpl(*params_));
    const auto& node_def = params_->node_def;
    if (node_def.op() == "LeakyRelu") {
      return GetNodeAttr(AttrSlice(node_def), "alpha", &alpha_);
    }
    alpha_ = 1.0f;
    return OkStatus();
  }
  Status Convert() {
    auto* converter = params_->converter;
    const auto& inputs = params_->inputs;
    const auto& node_def = params_->node_def;
    const auto& op = node_def.op();
    const auto op_pair = pOperMap_->find(op);
    nvinfer1::IActivationLayer* layer = converter->network()->addActivation(
        *inputs.at(0).tensor()->trt_tensor(), op_pair->second);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    converter->SetLayerName(layer, node_def, "activation");
    ITensorProxyPtr output_tensor = layer->getOutput(0);
    // Set parameters.
    if (op == "Selu") {
      // From tensorflow/core/kernels/relu_op_functor.h
      alpha_ = 1.7580993408473768599402175208123f;
      layer->setBeta(1.0507009873554804934193349852946f);
    } else if (op == "Softplus") {
      layer->setBeta(1.0f);
    } else if (op == "Relu6") {
      layer->setBeta(6.0f);
      converter->ProvideQuantizationRange(&output_tensor, alpha_ = 0.0f, 6.0f);
    }
    layer->setAlpha(alpha_);
    params_->outputs->push_back(TRT_TensorOrWeights(output_tensor));
    return OkStatus();
  }

 private:
  float alpha_ = 0.f;
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertUnary>(),
                                  GetOperationNames(*UnaryOperationMap()));
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertBooleanUnary>(),
    GetOperationNames(*UnaryBooleanOperationMap()));

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertActivation>(),
                                  GetOperationNames(*ActivationTypeMap()));
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
