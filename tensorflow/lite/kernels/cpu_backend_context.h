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

#include <memory>

#include "public/gemmlowp.h"
#include "tensorflow/lite/experimental/ruy/context.h"

namespace tflite {

class CpuBackendContext final {
 public:
  CpuBackendContext();
  ~CpuBackendContext();

  ruy::Context* ruy_context() const { return ruy_context_.get(); }

  gemmlowp::GemmContext* gemmlowp_context() const {
    return gemmlowp_context_.get();
  }

  void set_max_num_threads(int max_num_threads);

 private:
  // To enable a smooth transition from the current direct usage
  // of the underlying gemmlowp context to going through abstractions
  // (see :cpu_backend_gemm), for now a CpuBackendContext always
  // stores both a gemmlowp context and a ruy context.
  // TODO(b/131416458): Once call sites all go through abstractions,
  // elide what can be elided based on TFLITE_WITH_RUY.
  const std::unique_ptr<ruy::Context> ruy_context_;
  const std::unique_ptr<gemmlowp::GemmContext> gemmlowp_context_;

  CpuBackendContext(const CpuBackendContext&) = delete;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_CONTEXT_H_
