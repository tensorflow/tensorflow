/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_jitrt_stub.h"

#include <memory>
#include <string>
#include <utility>

namespace tensorflow {
namespace {

class TfrtJitRtStubRegistry {
 public:
  TfrtJitRtStubRegistry() : stub_(std::make_unique<TfrtJitRtStub>()) {}

  void Register(std::unique_ptr<TfrtJitRtStub> stub) {
    stub_ = std::move(stub);
  }

  TfrtJitRtStub &Get() { return *stub_; }

 private:
  std::unique_ptr<TfrtJitRtStub> stub_;
};

TfrtJitRtStubRegistry &GetGlobalTfrtJitRtStubRegistry() {
  static auto *const stub = new TfrtJitRtStubRegistry;
  return *stub;
}

}  // namespace

void RegisterTfrtJitRtStub(std::unique_ptr<TfrtJitRtStub> stub) {
  GetGlobalTfrtJitRtStubRegistry().Register(std::move(stub));
}

void RegisterJitRtDialects(mlir::DialectRegistry &registry) {
  GetGlobalTfrtJitRtStubRegistry().Get().RegisterJitRtDialects(registry);
}

// Helper function for inserting TFRT JitRt dialect conversions.
void PopulateJitRtConversionPatterns(mlir::ConversionTarget *target,
                                     mlir::MLIRContext *context,
                                     mlir::RewritePatternSet *patterns,
                                     CoreRTConverter *corert_converter) {
  GetGlobalTfrtJitRtStubRegistry().Get().PopulateJitRtConversionPatterns(
      target, context, patterns, corert_converter);
}

mlir::Value CreateJitRtFallbackCompileKernel(mlir::OpBuilder &builder,
                                             mlir::ModuleOp module,
                                             mlir::Value chain_value) {
  return GetGlobalTfrtJitRtStubRegistry()
      .Get()
      .CreateJitRtFallbackCompileKernel(builder, module, chain_value);
}

void AddTfrtJitRtPasses(const TfrtPipelineOptions &options,
                        mlir::OpPassManager &pm) {
  GetGlobalTfrtJitRtStubRegistry().Get().AddTfrtJitRtPasses(options, pm);
}

}  // namespace tensorflow
