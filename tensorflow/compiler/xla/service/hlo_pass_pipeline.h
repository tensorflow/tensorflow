/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_PIPELINE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_PIPELINE_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {

// Pipeline of HLO passes.
class HloPassPipeline : public HloPassInterface {
 public:
  explicit HloPassPipeline(const string& name) : name_(name) {}
  tensorflow::StringPiece name() const override { return name_; }

  // Add a pass to the pipeline. It should be called with the arguments for the
  // pass constructor:
  //
  //   pipeline.AddPass<FooPass>(constructor_arg1, constructor_arg2);
  //
  // Returns a reference to the added pass.
  template <typename T, typename... Args>
  T& AddPass(Args&&... args) {
    CHECK(!run_called_) << "AddPass cannot be called after Run";
    auto pass = new T(std::forward<Args>(args)...);
    passes_.push_back(std::unique_ptr<T>(pass));
    return *pass;
  }

  // Add an invariant-checking pass to the pipeline. It will be run before and
  // after each HLO pass. The invariant checking pass must not mutate the graph
  // (it is required to always return "false" from its Run() method).
  template <typename T, typename... Args>
  T& AddInvariantChecker(Args&&... args) {
    CHECK(!run_called_) << "AddInvariantChecker cannot be called after Run";
    auto pass = new T(std::forward<Args>(args)...);
    invariant_checkers_.push_back(std::unique_ptr<T>(pass));
    return *pass;
  }

  // Run all passes on the given HLO module.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  const string name_;
  std::vector<std::unique_ptr<HloPassInterface>> passes_;
  std::vector<std::unique_ptr<HloPassInterface>> invariant_checkers_;
  bool run_called_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(HloPassPipeline);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_PASS_PIPELINE_H_
