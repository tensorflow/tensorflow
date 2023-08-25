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

const BinaryOperationMapType* BinaryOperationMap() {
  static const auto* map = new BinaryOperationMapType({
    {"Add", nvinfer1::ElementWiseOperation::kSUM},
        {"AddV2", nvinfer1::ElementWiseOperation::kSUM},
        {"Mul", nvinfer1::ElementWiseOperation::kPROD},
        {"Sub", nvinfer1::ElementWiseOperation::kSUB},
        {"Div", nvinfer1::ElementWiseOperation::kDIV},
        {"FloorDiv", nvinfer1::ElementWiseOperation::kFLOOR_DIV},
        {"RealDiv", nvinfer1::ElementWiseOperation::kDIV},
        {"Minimum", nvinfer1::ElementWiseOperation::kMIN},
        {"Maximum", nvinfer1::ElementWiseOperation::kMAX},
        {"Pow", nvinfer1::ElementWiseOperation::kPOW},
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
        {"Greater", nvinfer1::ElementWiseOperation::kGREATER},
        {"Less", nvinfer1::ElementWiseOperation::kLESS},
        {"Equal", nvinfer1::ElementWiseOperation::kEQUAL},
        // Operators are implemented as NOT Less and NOT Greater, respectively.
        {"GreaterEqual", nvinfer1::ElementWiseOperation::kLESS},
        {"LessEqual", nvinfer1::ElementWiseOperation::kGREATER},
#endif
  });
  return map;
}

const BinaryOperationMapType* BinaryBooleanOperationMap() {
  static const auto* map = new BinaryOperationMapType({
      {"LogicalOr", nvinfer1::ElementWiseOperation::kOR},
      {"LogicalAnd", nvinfer1::ElementWiseOperation::kAND},
  });
  return map;
}

namespace {
class ConvertBinaryImpl {
 protected:
  ConvertBinaryImpl(const BinaryOperationMapType* pOperMap)
      : pOperMap_(pOperMap) {}

  Status ValidateImpl(
      const OpConverterParams& params,
      const std::vector<string>& implicit_batch_not_supported_ops = {},
      bool both_tensors = false) {
    const auto& node_def = params.node_def;
    const auto& op = node_def.op();
    const auto op_pair = pOperMap_->find(op);
    if (op_pair == pOperMap_->end()) {
      return errors::Unimplemented("Binary op: ", op, " not supported");
    }

    // Constant folding should have been done by TensorFlow.
    const auto& inputs = params.inputs;
    if (inputs.at(0).is_weights() && inputs.at(1).is_weights()) {
      return errors::Unimplemented(
          "Constant folding is falled back to TensorFlow, binary op '", op,
          "' received both input as constant");
    }

    if ((convertToBool_ = find_name(op, implicit_batch_not_supported_ops))) {
      if (params.use_implicit_batch) {
        return errors::Unimplemented(
            convert_not_supported_implicit(op, node_def.name(), "Binary"));
      }
    }

    if (both_tensors) {
      if (inputs.at(0).is_weights() || inputs.at(1).is_weights()) {
        return errors::InvalidArgument("Both inputs  of '", op,
                                       "' are expected to be tensors");
      }
      // No need to convert the output of "LogicalOr" and "LogicalAnd"
      convertToBool_ = false;
    }

    nvinfer1::Dims broadcasted_dims[2];
    TF_RETURN_IF_ERROR(GetTrtBroadcastShape(
        inputs.at(0), inputs.at(1), true, params.use_implicit_batch,
        broadcasted_dims, broadcasted_dims + 1));

    for (int i = 0; i < tensor_.size(); i++) {
      // This will also convert constants to tensors.
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params.converter, inputs.at(i), broadcasted_dims[i],
          params.validation_only, &tensor_[i], node_def, i));
    }
    operation_ = op_pair->second;
    return OkStatus();
  }

  Status ConvertImpl(const OpConverterParams& params,
                     const std::vector<string>& revert_bool_ops = {}) {
    const auto& node_def = params.node_def;
    // Add ElementWise layer.
    auto* network = params.converter->network();
    nvinfer1::ILayer* layer = network->addElementWise(
        *tensor_[0]->trt_tensor(), *tensor_[1]->trt_tensor(), operation_);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());

    if (params.use_explicit_precision) {
      layer->setPrecision(nvinfer1::DataType::kFLOAT);
    }

    params.converter->SetLayerName(layer, node_def);
    const auto& output = layer->getOutput(0);
    if (convertToBool_) {
      output->setType(nvinfer1::DataType::kBOOL);
      if (find_name(node_def.op(), revert_bool_ops)) {
        nvinfer1::IUnaryLayer* unary_layer =
            network->addUnary(*output, nvinfer1::UnaryOperation::kNOT);
        TFTRT_RETURN_ERROR_IF_NULLPTR(unary_layer, node_def.name());
        params.outputs->push_back(
            TRT_TensorOrWeights(unary_layer->getOutput(0)));
        return OkStatus();
      }
    }

    params.outputs->push_back(TRT_TensorOrWeights(output));
    return OkStatus();
  }

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("x", TrtInputArg::kBoth),
        InputArgSpec::Create("y", TrtInputArg::kBoth)};
  }

 private:
  const BinaryOperationMapType* pOperMap_;
  std::array<ITensorProxyPtr, 2> tensor_{nullptr, nullptr};
  nvinfer1::ElementWiseOperation operation_;
  bool convertToBool_;
};

class ConvertBinary : public OpConverterBase<ConvertBinary>,
                      protected ConvertBinaryImpl {
 public:
  explicit ConvertBinary(const OpConverterParams* params)
      : OpConverterBase<ConvertBinary>(
            params,
            {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}),
        ConvertBinaryImpl(BinaryOperationMap()) {}

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return ConvertBinaryImpl::InputSpec();
  }

  Status Validate() {
    const std::vector<string> implicit_batch_not_supported_ops {
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
      "Greater", "Less", "Equal", "GreaterEqual", "LessEqual"
#endif
    };
    return ValidateImpl(*params_, implicit_batch_not_supported_ops);
  }
  Status Convert() {
    const std::vector<string> implemented_with_reverted_ops {
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
      "GreaterEqual", "LessEqual"
#endif
    };
    return ConvertImpl(*params_, implemented_with_reverted_ops);
  }
};

class ConvertBooleanBinary : public OpConverterBase<ConvertBooleanBinary>,
                             public ConvertBinaryImpl {
 public:
  explicit ConvertBooleanBinary(const OpConverterParams* params)
      : OpConverterBase<ConvertBooleanBinary>(params, {DataType::DT_BOOL}),
        ConvertBinaryImpl(BinaryBooleanOperationMap()) {}

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return ConvertBinaryImpl::InputSpec();
  }

  static constexpr const char* NodeDefDataTypeAttributeName() {
    /*
    node {
      name: "..."
      op: "LogicalOr"
      input: "..."
      input: "..."
      attr {
        key: "_output_shapes"
        ...
      }
    }
    */
    return "";
  }
  Status Validate() {
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    return ValidateImpl(*params_, {"LogicalOr", "LogicalAnd"}, true);
#else
    return errors::Unimplemented("Boolean op: ", params_->node_def.op(),
                                 " is not supported in TRT version < 8.2");
#endif
  }
  Status Convert() { return ConvertImpl(*params_); }
};
}  // namespace

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertBinary>(),
                                  GetOperationNames(*BinaryOperationMap()));
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertBooleanBinary>(),
    GetOperationNames(*BinaryBooleanOperationMap()));

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
