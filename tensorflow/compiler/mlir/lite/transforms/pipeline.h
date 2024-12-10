/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PIPELINE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PIPELINE_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/pass.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_options_setter.h"

namespace mlir {
namespace TFL {

/// Pipeline is a base class for pipelines of passes.
///
/// A pipeline is a collection of passes that are run in a specific order. The
/// pipeline can be configured with options that control which passes are
/// enabled and how they are run.
///
/// To create a new pipeline, derive from this class and implement the
/// `AddPasses` method. This method should add passes to the pipeline using the
/// `AddPass` method.
///
/// Example:
///
/// ```cpp
/// class MyPipeline : public Pipeline<MyPipeline> {
///  public:
///   void AddPasses() override {
///     AddPass<Pass1>();
///     AddPass<Pass2>();
///   }
/// };
/// ```
template <typename DerivedPipeline,
          typename DerivedPipelineOptions = EmptyPassOptions>
class Pipeline {
 public:
  struct PipelineEntry {
    std::unique_ptr<mlir::Pass> pass;
    std::function<bool(const DerivedPipelineOptions &)> enable_condition;
  };

  Pipeline() = default;
  virtual ~Pipeline() = default;
  virtual void AddPasses() = 0;

  /// Function to force the derived pipeline to implement the metadata
  // method.
  llvm::StringRef getArgument() const { return DerivedPipeline::GetArgument(); }

  llvm::StringRef getDescription() const {
    return DerivedPipeline::GetDescription();
  }

  llvm::StringRef getName() const { return DerivedPipeline::GetName(); }

  void GetPipeline(mlir::OpPassManager &pm,
                   const DerivedPipelineOptions &options) {
    for (auto &&entry : passes_) {
      if (entry.enable_condition(options)) {
        pm.addPass(std::move(entry.pass));
      }
    }
  };

 protected:
  void AddPass(
      std::unique_ptr<mlir::Pass> pass,
      std::function<bool(const DerivedPipelineOptions &)> enable_condition) {
    passes_.push_back({std::move(pass), enable_condition});
  }

  template <typename DerivedPipelinePass, typename DerivedPipelinePassOptions>
  friend class PipelinePass;

  std::vector<mlir::Pass *> GetPasses() {
    std::vector<mlir::Pass *> passes;
    passes.reserve(passes_.size());
    for (auto &&entry : passes_) {
      passes.push_back(entry.pass.get());
    }
    return passes;
  }

 private:
  std::vector<PipelineEntry> passes_;
};

/// PipelinePass is a wrapper class to run a pipeline of passes as a single
/// pass. This is an implementation detail of the pipelines mechanism in TFL
/// Converter framework. Users should not need to interact with this class
/// directly.
template <typename Pipeline, typename PipelineOptions = EmptyPassOptions>
class PipelinePass
    : public Pass<PipelinePass<Pipeline, PipelineOptions>, PipelineOptions> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipelinePass);

  PipelinePass() { pipeline_->AddPasses(); };
  PipelinePass(const PipelinePass &) {};
  explicit PipelinePass(const PipelineOptions &options)
      : Pass<PipelinePass<Pipeline, PipelineOptions>, PipelineOptions>(
            options) {
    pipeline_.AddPasses();
  };

  std::unique_ptr<::mlir::Pass> clonePass() const override {
    auto pass = std::make_unique<PipelinePass<Pipeline, PipelineOptions>>();
    pass->GetOptions().copyOptionValuesFrom(this->GetOptions());
    return std::move(pass);
  }

  /// Function to satisfy the mlir::Pass interface
  static llvm::StringRef GetArgument() { return Pipeline::GetArgument(); }

  static llvm::StringRef GetDescription() { return Pipeline::GetDescription(); }

  static llvm::StringRef GetName() { return Pipeline::GetName(); }

  void runOnOperation() final {
    ModuleOp module_op = this->getOperation();

    // Create a temporary OpPassManager to run the passes. Nesting is set to be
    // implicit to allow for the nesting to happen under-the-hood.
    OpPassManager pm(ModuleOp::getOperationName(),
                     OpPassManager::Nesting::Implicit);
    pipeline_->GetPipeline(pm, this->GetOptions());
    if (failed(this->runPipeline(pm, module_op))) {
      this->signalPassFailure();
    }
  };

  void ApplyOptionsVisitor(const PassOptionsSetter &visitor) final {
    visitor.SetOptions(this->GetOptions());

    for (auto &&pass : pipeline_->GetPasses()) {
      if (auto *derived_pass = dynamic_cast<MutableOptionsPass *>(pass)) {
        derived_pass->ApplyOptionsVisitor(visitor);
      }
    }
  }

 private:
  std::unique_ptr<Pipeline> pipeline_ = std::make_unique<Pipeline>();
};
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PIPELINE_H_
