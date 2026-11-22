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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QSA_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QSA_DELEGATE_KERNEL_H_

#include <memory>

#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace xnnpack {

// Executor for the QSA sparse-attention StableHLO composites:
//
//   * `odml.sparse_sdpa_transposed`: GQA attention restricted to the KV
//     micro-blocks selected by `topk_block_indices`, plus partially visible
//     blocks and the trailing tokens that do not form a complete block.
//   * `odml.qsa_indexer_update`: updates the indexer ring buffer and the
//     compressed block-key pool, scores the complete blocks and selects the
//     top-k block indices.
//
// XNNPACK has no top-k or dynamic gather, so both composites are executed by
// hand-written C++ kernels parallelized with the delegate's pthreadpool. Each
// QSA node is delegated as a singleton partition. The numerics follow the
// reference decompositions that the exporter embeds in each composite.
class QsaDelegateKernel {
 public:
  class Impl;

  ~QsaDelegateKernel();

  // Returns true if `node` is a StableHLO composite implemented by this class.
  static bool IsQsaNode(const TfLiteRegistration* registration,
                        const TfLiteNode* node);

  // Checks the attributes, tensor types and shapes of a QSA node.
  static TfLiteStatus IsSupported(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  const TfLiteRegistration* registration,
                                  int node_index);

  // Returns nullptr unless `params` describes a singleton supported QSA node.
  static std::unique_ptr<QsaDelegateKernel> Create(
      TfLiteContext* context, const TfLiteDelegateParams* params,
      pthreadpool_t threadpool);

  TfLiteStatus Prepare(TfLiteContext* context);
  TfLiteStatus Invoke(TfLiteContext* context);

 private:
  explicit QsaDelegateKernel(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QSA_DELEGATE_KERNEL_H_
