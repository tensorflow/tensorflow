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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_SHAPE_VERIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_SHAPE_VERIFIER_H_

#include <functional>
#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_verifier.h"

namespace xla {

// Verifies that HLO Shapes are supported by the XLA-GPU compiler.
class GpuShapeVerifier : public ShapeVerifier {
 public:
  explicit GpuShapeVerifier(const HloVerifierOpts& opts)
      : ShapeVerifier(opts) {}

  Status Preprocess(HloInstruction* hlo) override;
};

// A verifier metadata class that uses the GpuShapeVerifier.
class GpuVerifierMetadata : public TargetVerifierMetadata {
 public:
  explicit GpuVerifierMetadata(HloVerifierOpts&& opts)
      : TargetVerifierMetadata(std::move(opts)) {}

  std::unique_ptr<ShapeVerifier> GetVerifier() const override {
    return std::make_unique<GpuShapeVerifier>(GetVerifierOpts());
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_SHAPE_VERIFIER_H_
