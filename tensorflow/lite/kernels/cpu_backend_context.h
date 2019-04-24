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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_

#include "public/gemmlowp.h"
#include "tensorflow/lite/c/c_api_internal.h"

namespace tflite {

class CpuBackendContext final {
 public:
  explicit CpuBackendContext(TfLiteContext* tflite_context);
  ~CpuBackendContext();

  gemmlowp::GemmContext* gemmlowp_context() const { return gemmlowp_context_; }

  void set_max_num_threads(int max_num_threads);

 private:
  TfLiteContext* const tflite_context_;
  // gemmlowp context used to implement this CpuBackendContext.
  // Not owned: currently this is shared with other direct usage of this
  // gemmlowp context by other users of :gemmlowp_support.
  // TODO(benoitjacob): factor all gemmlowp context usage through
  // CpuBackendContext, then make this owned and delete :gemmlowp_support.
  gemmlowp::GemmContext* const gemmlowp_context_;

  CpuBackendContext() = delete;
  CpuBackendContext(const CpuBackendContext&) = delete;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_
