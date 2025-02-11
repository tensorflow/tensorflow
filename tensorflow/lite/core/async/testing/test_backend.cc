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

#include "tensorflow/lite/core/async/testing/test_backend.h"

#include <cstddef>
#include <string>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/utils.h"

namespace tflite {
namespace async {
namespace testing {

namespace {

TfLiteStatus DelegatePrepare(TfLiteContext* context,
                             TfLiteDelegate* tflite_delegate) {
  auto* backend = reinterpret_cast<TestBackend*>(tflite_delegate->data_);

  // Can delegate all nodes.
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool { return true; };

  delegates::GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition(nullptr));

  auto supported_nodes = helper.GetNodesOfFirstNLargestPartitions(
      backend->NumPartitions(), backend->MinPartitionedNodes());

  // Create TfLiteRegistration with the provided async kernel.
  TfLiteRegistration reg{};
  reg.init = [](TfLiteContext* context, const char* buffer,
                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    auto* backend = reinterpret_cast<TestBackend*>(params->delegate->data_);
    // AsyncSubgraph requires TfLiteNode.user_data to be of TfLiteAsyncKernel
    // type.
    return backend->get_kernel();
  };
  reg.free = [](TfLiteContext*, void*) -> void {};
  reg.prepare = [](TfLiteContext*, TfLiteNode*) -> TfLiteStatus {
    return kTfLiteOk;
  };
  reg.invoke = [](TfLiteContext*, TfLiteNode*) -> TfLiteStatus {
    return kTfLiteOk;
  };
  reg.profiling_string = nullptr;
  reg.builtin_code = kTfLiteBuiltinDelegate;
  reg.custom_name = "TestBackend";
  reg.version = 1;
  reg.async_kernel = [](TfLiteContext*,
                        TfLiteNode* node) -> TfLiteAsyncKernel* {
    return reinterpret_cast<TfLiteAsyncKernel*>(node->user_data);
  };

  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, reg, BuildTfLiteArray(supported_nodes).get(), tflite_delegate);
}

}  // namespace

TestBackend::TestBackend(TfLiteAsyncKernel* kernel)
    : kernel_(kernel), delegate_(TfLiteDelegateCreate()) {
  delegate_.Prepare = &DelegatePrepare;
  delegate_.CopyFromBufferHandle = nullptr;
  delegate_.CopyToBufferHandle = nullptr;
  delegate_.FreeBufferHandle = nullptr;
  delegate_.data_ = this;
}

}  // namespace testing
}  // namespace async
}  // namespace tflite
