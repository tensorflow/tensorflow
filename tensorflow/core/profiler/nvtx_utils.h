/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_NVTX_H_
#define TENSORFLOW_CORE_PLATFORM_NVTX_H_

#include "third_party/nvtx3/nvToolsExt.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

namespace nvtx {

#define NVTX_RANGE_INJECTION_STATUS_kDisabled 0
#define NVTX_RANGE_INJECTION_STATUS_kBasic 1
#define NVTX_RANGE_INJECTION_STATUS_kDetailed 2

extern const int8 NVTX_RANGE_INJECTION_STATUS;

// A helper function to decide whether to enable CUDA NVTX profiling ranges.
inline bool IsNvtxRangesEnabled(){
  // return NVTX_RANGE_INJECTION_STATUS == NVTX_RANGE_INJECTION_STATUS_kBasic;
  return NVTX_RANGE_INJECTION_STATUS != NVTX_RANGE_INJECTION_STATUS_kDisabled;
};

// A helper function to decide whether to enable CUDA NVTX profiling ranges
// with detailed node information.
inline bool IsNvtxRangesDetailedEnabled(){
  return NVTX_RANGE_INJECTION_STATUS == NVTX_RANGE_INJECTION_STATUS_kDetailed;
};

class NvtxDomain {
 public:
  explicit NvtxDomain(const char* name) : handle_(nvtxDomainCreateA(name)) {}
  ~NvtxDomain() { nvtxDomainDestroy(handle_); }
  operator nvtxDomainHandle_t() const { return handle_; }

 private:
  nvtxDomainHandle_t handle_;
  TF_DISALLOW_COPY_AND_ASSIGN(NvtxDomain);
};

string DataTypeToNumpyString(DataType dtype);

// TODO(benbarsdell): This is a bit crude and hacky (and inefficient).
string AttrValueToJson(const AttrValue& attr_value);

/*
string MaybeGetNvtxDomainRangeMessage(
    const OpKernel* op_kernel, const int num_inputs,
    std::vector<const TensorShape*> input_shape_array);
*/
namespace{
static const Tensor* const kEmptyTensor = new Tensor;

template <typename T1>
const Tensor* GetTensorValueForDump(const T1& input) {
  if (!input.has_value) {
    return kEmptyTensor;
  } else if (input.ref == nullptr) {
    return input.val.get();
  } else {
    return input.ref;
  }
}
}

template <typename T1, typename T2>
string MaybeGetNvtxDomainRangeMessage(const T1& item, T2* first_input) {

  if (!IsNvtxRangesEnabled()) {
    return string();
  } else {

    const OpKernel* kernel = item.kernel;
    const int num_inputs = item.num_inputs;

    if (IsNvtxRangesDetailedEnabled()) {

      std::vector<string> args_pieces;
      std::vector<string> attrs_pieces;

      std::vector<const TensorShape*> input_shape_array;

      for (int i = 0; i < item.num_inputs; ++i) {
        input_shape_array.push_back(
            &GetTensorValueForDump(first_input[i])->shape());
      }

      for (int i = 0; i < num_inputs; ++i) {
        if (i == 10) {
          // Truncate long arg lists and indicate with an ending null value.
          args_pieces.push_back("null");
          break;
        }
        const TensorShape& shape = *(input_shape_array[i]);
        string shape_str = shape.unknown_rank() ? "null" : shape.DebugString();
        args_pieces.push_back(strings::StrCat("{\"name\":\"",
                                              kernel->def().input(i),
                                              "\",\"shape\":", shape_str, "}"));
      }

      const auto& attrs = kernel->def().attr();

      for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        const string& key = it->first;
        const AttrValue& value = it->second;
        // Exclude types that aren't useful for profiling.
        if (value.value_case() == AttrValue::kFunc ||
            value.value_case() == AttrValue::kPlaceholder ||
            value.value_case() == AttrValue::VALUE_NOT_SET) {
          continue;
        }
        string value_str = AttrValueToJson(value);
        attrs_pieces.push_back(strings::StrCat("\"", key, "\":", value_str));
      }

      return strings::StrCat("{\"op\":\"", kernel->def().op(), "\",\"name\":\"",
                            kernel->name(), "\",\"args\":[",
                            str_util::Join(args_pieces, ","), "],\"attrs\":{",
                            str_util::Join(attrs_pieces, ","), "}}");
    }
    else {
      return kernel->def().op() + ": " + kernel->name();
    }
  }
}

nvtxRangeId_t MaybeNvtxDomainRangeStart(string node_op, string node_name);

nvtxRangeId_t MaybeNvtxDomainRangeStartMsg(string msg, string node_op);

void MaybeNvtxDomainRangeEnd(nvtxRangeId_t nvtx_range);

namespace hlo{

// Returns the op name for the node associated with this HLO, for use with
// NVTX range annotations.
string NvtxNodeNameString(string cluster_name, string op_name);

}  // namespace HLO

namespace eager{

// Returns the op name for the node associated with this HLO, for use with
// NVTX range annotations.
template <typename T1, typename T2>
string GetNvtxRangeMessage(const T1& inputs, T2* kernel_)
{
   if (IsNvtxRangesDetailedEnabled()) {
      std::vector<string> args_pieces;
      for (int i = 0; i < inputs.GetTensorValues()->size(); i++) {
        if (i == 10) {
          // Truncate long arg lists and indicate with an ending null value.
          args_pieces.push_back("null");
          break;
        }
        const auto& shape = inputs.GetTensorValues()->at(i).tensor->shape();
        string shape_str = shape.unknown_rank() ? "null" : shape.DebugString();
        args_pieces.push_back(strings::StrCat("{\"name\":\"",
                                              kernel_->def().input(i),
                                              "\",\"shape\":", shape_str, "}"));
      }
      std::vector<string> attrs_pieces;
      const auto& attrs = kernel_->def().attr();
      for (auto it = attrs.begin(); it != attrs.end(); ++it) {
        const string& key = it->first;
        const AttrValue& value = it->second;
        // Exclude types that aren't useful for profiling.
        if (value.value_case() == AttrValue::kFunc ||
            value.value_case() == AttrValue::kPlaceholder ||
            value.value_case() == AttrValue::VALUE_NOT_SET) {
          continue;
        }
        string value_str = AttrValueToJson(value);
        attrs_pieces.push_back(strings::StrCat("\"", key, "\":", value_str));
      }
      return strings::StrCat("{\"op\":\"", kernel_->def().op(), "\",\"name\":\"",
                            kernel_->name(), "\",\"args\":[",
                            str_util::Join(args_pieces, ","), "],\"attrs\":{",
                            str_util::Join(attrs_pieces, ","), "}}");
    } else {
      return kernel_->def().op() + ": " + kernel_->name();
    }

}

}  // namespace eager

}  // namespace nvtx
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_NVTX_H_
