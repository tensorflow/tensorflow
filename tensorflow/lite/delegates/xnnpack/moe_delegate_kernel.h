/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_MOE_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_MOE_DELEGATE_KERNEL_H_

#include <memory>

#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace xnnpack {

// Executor for the opt-in custom "moe" XNNPACK delegate path.
//
// This class handles only singleton MoE delegate partitions. Ordinary XNNPACK
// nodes are still lowered through the regular XNNPACK subgraph path.
class MoeExpertsDelegateKernel {
 public:
  ~MoeExpertsDelegateKernel();

  static bool IsMoeExpertsNode(const TfLiteRegistration* registration,
                               const TfLiteNode* node);

  static TfLiteStatus IsSupported(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  const TfLiteRegistration* registration,
                                  int node_index);

  static std::unique_ptr<MoeExpertsDelegateKernel> Create(
      TfLiteContext* context, const TfLiteDelegateParams* params,
      pthreadpool_t threadpool);

  TfLiteStatus Prepare(TfLiteContext* context);
  TfLiteStatus Invoke(TfLiteContext* context);

 private:
  class Impl;

  explicit MoeExpertsDelegateKernel(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_MOE_DELEGATE_KERNEL_H_
