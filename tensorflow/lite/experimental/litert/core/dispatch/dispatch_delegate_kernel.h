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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DISPATCH_DISPATCH_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DISPATCH_DISPATCH_DELEGATE_KERNEL_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"
#include "tensorflow/lite/experimental/litert/c/litert_dispatch_delegate.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

namespace litert {
namespace internal {

// A TFL kernel that the interpreter calls to dispatch execution through the
// Dispatch API.
class DispatchDelegateKernel
    : public tflite::SimpleOpaqueDelegateKernelInterface {
 public:
  using Ptr = std::unique_ptr<tflite::SimpleOpaqueDelegateKernelInterface>;

  ~DispatchDelegateKernel() override;

  static absl::StatusOr<Ptr> Create(
      std::string&& graph_name, const LiteRtDispatchDelegateOptions& options);

  TfLiteStatus Init(TfLiteOpaqueContext* context,
                    const TfLiteOpaqueDelegateParams* params) override;

  TfLiteStatus Prepare(TfLiteOpaqueContext* context,
                       TfLiteOpaqueNode* node) override;

  TfLiteStatus Eval(TfLiteOpaqueContext* context,
                    TfLiteOpaqueNode* node) override;

 private:
  DispatchDelegateKernel(const LiteRtDispatchDelegateOptions& options,
                         std::string&& graph_name,
                         LiteRtDispatchDeviceContext device_context)
      : options_(options),
        graph_name_(std::move(graph_name)),
        device_context_(device_context) {}

  absl::StatusOr<TensorBufferRequirements> GetBufferRequirements(
      const RankedTensorType& tensor_type, int io_tensor_index,
      bool is_input) const;
  TfLiteStatus SetBuffer(const TfLiteOpaqueTensor* tfl_opaque_tensor,
                         int buffer_index, bool is_input);

  const LiteRtDispatchDelegateOptions& options_;
  std::string graph_name_;
  LiteRtDispatchDeviceContext device_context_;
  LiteRtDispatchInvocationContext invocation_context_ = nullptr;

  std::vector<TensorBuffer> input_tensor_buffers_;
  std::vector<LiteRtTensorBufferHandle> input_tensor_buffer_handles_;
  std::vector<size_t> input_tensor_buffer_used_size_;

  std::vector<TensorBuffer> output_tensor_buffers_;
  std::vector<LiteRtTensorBufferHandle> output_tensor_buffer_handles_;
  std::vector<size_t> output_tensor_buffer_used_size_;
};

}  // namespace internal
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CORE_DISPATCH_DISPATCH_DELEGATE_KERNEL_H_
