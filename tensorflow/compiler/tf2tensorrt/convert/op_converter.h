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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_H_

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <memory>
#include <vector>

#include "absl/strings/str_format.h"
#include "tensorflow/compiler/tf2tensorrt/convert/trt_parameters.h"
#include "tensorflow/compiler/tf2tensorrt/convert/weights.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

class Converter;

// Specifies the expected type taken by a TRT_TensorOrWeights input during op
// conversion.
// kResource is only used for resource variable ops. For an operation like
// Add(tensor, ReadVariableOp(...)), the second operand of Add is the result of
// the ReadVariableOp, which is a kWeight.
enum class TrtInputArg { kTensor = 1, kWeight = 2, kBoth = 3, kResource = 4 };

// Parameters for each op converter.
struct OpConverterParams {
  // Constructor used for validation only.
  OpConverterParams(const NodeDef& node_def,
                    const std::vector<TRT_TensorOrWeights>& inputs,
                    std::vector<TRT_TensorOrWeights>* outputs,
                    TrtWeightStore* weight_store,
                    TrtPrecisionMode precision_mode, bool use_calibration,
                    bool use_implicit_batch, bool use_explicit_precision);

  // Constructor used for conversion.
  OpConverterParams(Converter* converter, const NodeDef& node_def,
                    const std::vector<TRT_TensorOrWeights>& inputs,
                    std::vector<TRT_TensorOrWeights>* outputs,
                    TrtWeightStore* weight_store);

  Converter* converter = nullptr;
  const NodeDef& node_def;
  const std::vector<TRT_TensorOrWeights>& inputs;
  std::vector<TRT_TensorOrWeights>* outputs;
  const bool validation_only;
  TrtWeightStore* weight_store;
  const TrtPrecisionMode precision_mode;
  const bool use_calibration;
  const bool use_implicit_batch;
  const bool use_explicit_precision;
};

// Operation converter function specification.
using OpConverter = std::function<Status(const OpConverterParams*)>;

struct InputArgSpec {
  absl::string_view name;
  TrtInputArg allowed_roles;

  static constexpr InputArgSpec Create(absl::string_view n, TrtInputArg role) {
    return InputArgSpec{n, role};
  }
};

template <typename T>
std::string convert_not_supported_dtype_msg(const T& allowed_types,
                                            DataType tf_type,
                                            const NodeDef& node) {
  string allowed_types_string =
      absl::StrJoin(allowed_types, ", ", [](string* out, const DataType& type) {
        absl::StrAppendFormat(out, "%s", DataTypeString(type));
      });

  return absl::StrCat("Data type ", DataTypeString(tf_type),
                      " is not supported for ", node.op(), ", must be one of [",
                      allowed_types_string, "]");
}

std::string convert_not_supported_implicit(const std::string& pOpName,
                                           const std::string& pNodeName,
                                           const char* pOpType = NULL);

// A Curiously recurring template pattern (CRTP) template class for operation
// converters.
template <typename Impl>
class OpConverterBase {
 public:
  explicit OpConverterBase(const OpConverterParams* params,
                           const std::vector<DataType>& data_types =
                               {DataType::DT_FLOAT, DataType::DT_HALF})
      : params_(params),
        node_def_attrs_(params->node_def),
        allowed_dtypes_(data_types) {}

  // Default NodeDef attribute name to inspect in order to determine node data
  // type. The Impl class can override this by implementing the same function.
  static constexpr const char* NodeDefDataTypeAttributeName() { return "T"; }

  // Validate data type of the given NodeDef against allowed types.
  Status ValidateNodeDefDataType() {
    // If the attribute name is empty, we should skip this check.
    if (absl::string_view(Impl::NodeDefDataTypeAttributeName()).empty()) {
      return Status::OK();
    }

    // Get the NodeDef data type.
    auto dtype = GetAttrValue<DataType>(Impl::NodeDefDataTypeAttributeName());
    if (!dtype.ok()) {
      return errors::InvalidArgument("Attribute with name ",
                                     Impl::NodeDefDataTypeAttributeName(),
                                     " not found.");
    }

    // Check allowed data types.;
    if (std::find(allowed_dtypes_.begin(), allowed_dtypes_.end(), *dtype) ==
        allowed_dtypes_.end()) {
      return errors::Unimplemented(convert_not_supported_dtype_msg(
          allowed_dtypes_, *dtype, params_->node_def));
    }
    return Status::OK();
  }

  static constexpr bool HasFixNumberOfInputs() { return true; }

  // Validates input argument roles and data types.
  Status ValidateInputs() {
    const NodeDef& node_def = params_->node_def;
    const auto& inputs = params_->inputs;
    if (Impl::HasFixNumberOfInputs()) {
      TRT_ENSURE(inputs.size() == Impl::InputSpec().size());
    } else {
      TRT_ENSURE(inputs.size() <= Impl::InputSpec().size());
    }
    for (int i = 0; i < inputs.size(); i++) {
      const InputArgSpec arg_spec = Impl::InputSpec()[i];
      if (arg_spec.allowed_roles == TrtInputArg::kWeight &&
          inputs.at(i).is_tensor()) {
        return errors::Unimplemented("The input \"", arg_spec.name, "\" for ",
                                     node_def.op(), " must be a constant, at ",
                                     node_def.name());
      }
      if (arg_spec.allowed_roles == TrtInputArg::kTensor &&
          inputs.at(i).is_weights()) {
        return errors::Unimplemented("The input \"", arg_spec.name, "\" for ",
                                     node_def.op(), " must be a tensor, at ",
                                     node_def.name());
      }
    }
    return Status::OK();
  }

  Status operator()() {
    // Validate data type and inputs.
    TF_RETURN_IF_ERROR(this->ValidateNodeDefDataType());
    TF_RETURN_IF_ERROR(this->ValidateInputs());

    // Perform op-level validation.
    TF_RETURN_IF_ERROR(reinterpret_cast<Impl*>(this)->Validate());
    if (params_->validation_only) {
      return Status::OK();
    }

    // Perform conversion.
    return reinterpret_cast<Impl*>(this)->Convert();
  }

 protected:
  Status NotSupportedInImplicitBatch(const char* pOpType = nullptr) {
    if (params_->use_implicit_batch) {
      const auto& op = params_->node_def.op();
      const auto& nodeName = params_->node_def.name();
      const auto& error = convert_not_supported_implicit(op, nodeName, pOpType);
      return errors::Unimplemented(error);
    }
    return Status::OK();
  }

  void AddOutput(const TRT_TensorOrWeights& out) {
    params_->outputs->push_back(out);
  }

  template <typename T>
  StatusOr<T> GetAttrValue(absl::string_view key) const {
    T result;
    TF_RETURN_IF_ERROR(GetNodeAttr(node_def_attrs_, key, &result));
    return result;
  }

  const OpConverterParams* const params_;
  const AttrSlice node_def_attrs_;
  const std::vector<DataType> allowed_dtypes_;
};

// Constructs and returns a converter function for a given operation converter
// class T. This requires T to be a derived class of StructuredOpConverter.
template <typename T>
OpConverter MakeConverterFunction() {
  return [](const OpConverterParams* params) -> Status {
    T converter(params);
    return converter();
  };
}

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OP_CONVERTER_H_
