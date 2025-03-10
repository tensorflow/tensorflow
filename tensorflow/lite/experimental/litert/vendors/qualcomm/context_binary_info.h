// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CONTEXT_BINARY_INFO_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CONTEXT_BINARY_INFO_H_

#include <cstddef>
#include <string>
#include <vector>
#include <list>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"
#include "third_party/qairt/latest/include/QNN/QnnInterface.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"

namespace litert {
namespace qnn {

class GraphInfo {
 public:
  static Expected<GraphInfo> Create(
      const QnnSystemContext_GraphInfo_t& graph_info);
  const std::string& Name() const { return name_; }
  const std::vector<::qnn::TensorWrapperRef>& Inputs() const { return inputs_; }
  const std::vector<::qnn::TensorWrapperRef>& Outputs() const { return outputs_; }

 private:
  GraphInfo() = default;
  Expected<void> Init(const QnnSystemContext_GraphInfo_t& graph_info);
  std::string name_;
  std::vector<::qnn::TensorWrapperRef> inputs_;
  std::vector<::qnn::TensorWrapperRef> outputs_;
  std::list<::qnn::TensorWrapper> owned_tensors_;
};

class ContextBinaryInfo {
 public:
  static Expected<ContextBinaryInfo> Create(QnnManager& qnn,
                                            const void* exec_bytecode_ptr,
                                            size_t exec_bytecode_size);
  const std::vector<::qnn::TensorWrapperRef>& ContextTensors() const {
    return context_tensors_;
  }
  const std::vector<GraphInfo>& Graphs() const { return graphs_; }

 private:
  ContextBinaryInfo() = default;
  Expected<void> Init(const QnnSystemContext_BinaryInfo_t& binary_info);
  std::vector<GraphInfo> graphs_;
  std::vector<::qnn::TensorWrapperRef> context_tensors_;
  std::list<::qnn::TensorWrapper> owned_tensors_;
};

}  // namespace qnn
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CONTEXT_BINARY_INFO_H_
