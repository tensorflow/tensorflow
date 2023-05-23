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

#include <algorithm>
#include <memory>

#include "google/protobuf/text_format.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/tac_filter.pb.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

using ::third_party::tensorflow::compiler::mlir::lite::experimental::tac::
    FunctionFilter;
using ::third_party::tensorflow::compiler::mlir::lite::experimental::tac::
    TacFilter;
using ::third_party::tensorflow::compiler::mlir::lite::experimental::tac::
    TacFilters;

class TacFilterPass
    : public PassWrapper<TacFilterPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TacFilterPass)

  TacFilterPass() = default;
  TacFilterPass(const TacFilterPass& other) {
    this->tac_filters_ = other.tac_filters_;
  }
  explicit TacFilterPass(TacFilters* tac_filters) {
    tac_filters_ = tac_filters;
  }

 private:
  TacFilters* tac_filters_ = nullptr;

  llvm::StringRef getArgument() const final { return "tfl-tac-filter"; }
  llvm::StringRef getDescription() const final {
    return "This pass marks the ops to skip target annotation by inserting "
           "`tac.skip_target_annotation` attribute to them based on user "
           "provided config.";
  }

  Option<bool> use_test_setting_{
      *this, "use-test-setting",
      llvm::cl::desc(
          "Whether to use the test config for the tac filter protobuf."),
      llvm::cl::init(false)};

  void runOnOperation() override;
};

void ApplyFunctionTacFilter(func::FuncOp func,
                            FunctionFilter::FunctionFilterType type,
                            OpBuilder& builder) {
  for (Operation& op : func.front()) {
    if (type == FunctionFilter::SKIP_TARGET_ANNOTATION) {
      op.setAttr(kSkipTargetAnnotation, builder.getUnitAttr());
    } else if (type == FunctionFilter::INCLUDE_TARGET_ANNOTATION) {
      op.removeAttr(kSkipTargetAnnotation);
    }
  }
}

void ApplyTacFilter(ModuleOp module, const TacFilter& tac_filter,
                    OpBuilder& builder) {
  if (tac_filter.has_function_filter()) {
    llvm::Regex func_regex(
        tac_filter.function_filter().function_name_pattern());
    for (auto func : module.getOps<func::FuncOp>()) {
      if (!func_regex.match(func.getName())) {
        continue;
      }

      ApplyFunctionTacFilter(func, tac_filter.function_filter().filter_type(),
                             builder);
    }
    return;
  }

  llvm::Regex op_regex(tac_filter.op_filter().op_name_pattern());
  module.walk([&](Operation* op) {
    auto named_loc = op->getLoc().dyn_cast<NameLoc>();
    if (!named_loc) {
      return;
    }
    if (!op_regex.match(named_loc.getName())) {
      return;
    }

    op->setAttr(kSkipTargetAnnotation, builder.getUnitAttr());
  });
}

void TacFilterPass::runOnOperation() {
  TacFilters test_tac_filters;
  if (use_test_setting_) {
    // Sets up the test config used in the mlir LIT test.
    google::protobuf::TextFormat::ParseFromString(R"(
      tac_filters {
        function_filter {
          function_name_pattern: "^testFunction"
        }
      }
      tac_filters {
        function_filter {
          function_name_pattern: "testFunctionInclude"
          filter_type: INCLUDE_TARGET_ANNOTATION
        }
      }
      tac_filters {
        op_filter {
          op_name_pattern: "^test_op"
        }
      }
    )",
                                        &test_tac_filters);
    tac_filters_ = &test_tac_filters;
  }

  if (!tac_filters_) {
    return;
  }

  ModuleOp module = getOperation();
  OpBuilder builder(module);
  std::sort(tac_filters_->mutable_tac_filters()->pointer_begin(),
            tac_filters_->mutable_tac_filters()->pointer_end(),
            [](const TacFilter* a, const TacFilter* b) {
              const bool a_is_function_filter = a->has_function_filter();
              const bool b_is_function_filter = b->has_function_filter();
              if (a_is_function_filter != b_is_function_filter) {
                // Function filter is applied before op filter.
                return a_is_function_filter > b_is_function_filter;
              }

              if (!a_is_function_filter && !b_is_function_filter) {
                // The order of 2 op filters don't matter.
                return false;
              }

              const bool a_is_function_exclude =
                  (a->function_filter().filter_type() ==
                   FunctionFilter::SKIP_TARGET_ANNOTATION);
              const bool b_is_function_exclude =
                  (b->function_filter().filter_type() ==
                   FunctionFilter::SKIP_TARGET_ANNOTATION);
              // Function exclude filter is applied before function include
              // filter.
              return a_is_function_exclude > b_is_function_exclude;
            });
  for (const TacFilter& tac_filter : tac_filters_->tac_filters()) {
    ApplyTacFilter(module, tac_filter, builder);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTacFilterPass(
    TacFilters* tac_filters) {
  return std::make_unique<TacFilterPass>(tac_filters);
}

static PassRegistration<TacFilterPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
