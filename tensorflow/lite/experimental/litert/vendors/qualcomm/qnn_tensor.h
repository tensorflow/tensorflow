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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_QNN_TENSOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_QNN_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnInterface.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert::qnn {

class QnnTensor {
 public:
  static Expected<QnnTensor> Create(const Qnn_Tensor_t& tensor);

  QnnTensor(const QnnTensor& other);
  QnnTensor(QnnTensor&& other);

  QnnTensor& operator=(const QnnTensor&) = delete;
  QnnTensor& operator=(QnnTensor&&) = delete;

  Qnn_Tensor_t& Tensor() { return tensor_; }
  const Qnn_Tensor_t& Tensor() const { return tensor_; }

  size_t Rank() const { return dimensions_.size(); }
  const uint32_t* Dimensions() const { return dimensions_.data(); }

 private:
  explicit QnnTensor(const Qnn_Tensor_t& tensor) : tensor_(tensor) {}
  Expected<void> DeepCopy();

  Qnn_Tensor_t tensor_;
  std::string name_;
  std::vector<uint32_t> dimensions_;
  std::vector<uint8_t> is_dynamic_dimensions_;
};

}  // namespace litert::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_QNN_TENSOR_H_
