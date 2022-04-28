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

const unaryOperationMap* UnaryOperationMap() {
  static auto* const m =
      new std::unordered_map<string, nvinfer1::UnaryOperation>({
        {"Neg", nvinfer1::UnaryOperation::kNEG},
            {"Exp", nvinfer1::UnaryOperation::kEXP},
            {"Log", nvinfer1::UnaryOperation::kLOG},
            {"Rsqrt", nvinfer1::UnaryOperation::kSQRT},
            {"Sqrt", nvinfer1::UnaryOperation::kSQRT},
            {"Abs", nvinfer1::UnaryOperation::kABS},
            {"Reciprocal", nvinfer1::UnaryOperation::kRECIP},
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

const unaryOperationMap* UnaryBooleanOperationMap() {
  static auto* const m = new unaryOperationMap({
      {"LogicalNot", nvinfer1::UnaryOperation::kNOT},
  });
  return m;
}

class ConvertUnaryImpl {
 protected:
  ConvertUnaryImpl(const unaryOperationMap* pOperMap) : pOperMap_(pOperMap) {}

  Status ImplValidate(const OpConverterParams& params,
                      const std::vector<string>& not_supported_ops = {}) {
    const auto& op = params.node_def.op();
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
        return errors::Unimplemented(
            "Unary op: '", op, "' is not supported in implicit batch mode");
      }
    }

    return Status::OK();
  }

  Status ImplConvert(const OpConverterParams& params) {
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
    return Status::OK();
  }
  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return std::array<InputArgSpec, 1>{
        InputArgSpec::Create("x", TrtInputArg::kTensor)};
  }

 private:
  const unaryOperationMap* pOperMap_;
};

class ConvertUnary : public OpConverterBase<ConvertUnary>,
                     protected ConvertUnaryImpl {
 public:
  explicit ConvertUnary(OpConverterParams* params)
      : OpConverterBase<ConvertUnary>(params),
        ConvertUnaryImpl(UnaryOperationMap()) {}

  static constexpr std::array<DataType, 2> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return ConvertUnaryImpl::InputSpec();
  }

  static constexpr const char* NodeDefDataTypeAttributeName() { return ""; }
  Status Validate() { return ImplValidate(*params_, {"Sign", "Round"}); }
  Status Convert() { return ImplConvert(*params_); }
};

class ConvertBooleanUnary : public OpConverterBase<ConvertBooleanUnary>,
                            public ConvertUnaryImpl {
 public:
  explicit ConvertBooleanUnary(OpConverterParams* params)
      : OpConverterBase<ConvertBooleanUnary>(params),
        ConvertUnaryImpl(UnaryBooleanOperationMap()) {}

  static constexpr std::array<DataType, 1> AllowedDataTypes() {
    return {DataType::DT_BOOL};
  }

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return ConvertUnaryImpl::InputSpec();
  }

  static constexpr const char* NodeDefDataTypeAttributeName() { return ""; }
  Status Validate() {
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    return ImplValidate(*params_, {"LogicalNot"});
#else
    return errors::Unimplemented("Boolean op: ", params_->node_def.op(),
                                 " is not supported in TRT version < 8.2");
#endif
  }
  Status Convert() { return ImplConvert(*params_); }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertUnary>(),
                                  GetOperationNames(*UnaryOperationMap()));
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertBooleanUnary>(),
    GetOperationNames(*UnaryBooleanOperationMap()));

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
