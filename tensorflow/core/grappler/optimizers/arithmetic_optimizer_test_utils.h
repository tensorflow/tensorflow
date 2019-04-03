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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_TEST_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_TEST_UTILS_H_

#include "tensorflow/core/grappler/optimizers/arithmetic_optimizer.h"
#include "tensorflow/core/grappler/optimizers/constant_folding.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace grappler {

class ArithmeticOptimizerTest : public GrapplerTest {
 protected:
  // Optimize a graph using ArithmeticOptimizer and prune all the nodes that no
  // longer have any output consumers.
  void OptimizeAndPrune(ArithmeticOptimizer* optimizer, GrapplerItem* item,
                        GraphDef* output) {
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(ModelPruner().Optimize(nullptr, *item, output));
  }

  // Run ArithmeticOptimizer twice to make sure the rewrite is idempotent.
  void OptimizeTwice(ArithmeticOptimizer* optimizer, GrapplerItem* item,
                     GraphDef* output) {
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));
  }

  // Run ArithmeticOptimizer twice to make sure the rewrite is idempotent.
  // Optionally run a constant folding pass before pruning.
  void OptimizeTwiceAndPrune(ArithmeticOptimizer* optimizer, GrapplerItem* item,
                             GraphDef* output, bool const_folding = false) {
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));

    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(optimizer->Optimize(nullptr, *item, output));

    if (const_folding) {
      item->graph.Swap(output);
      output->Clear();
      TF_EXPECT_OK(ConstantFolding(/*cpu_device=*/nullptr)
                       .Optimize(nullptr, *item, output));
    }

    item->graph.Swap(output);
    output->Clear();
    TF_EXPECT_OK(ModelPruner().Optimize(nullptr, *item, output));
  }

  // TODO(ezhulenev): Make private. After migration to stages each test
  // should explicitly enable required optimization for tests isolation
  void DisableAllStages(ArithmeticOptimizer* optimizer) {
    ArithmeticOptimizer::ArithmeticOptimizerOptions options;
    options.dedup_computations = false;
    options.combine_add_to_addn = false;
    options.convert_sqrt_div_to_rsqrt_mul = false;
    options.convert_pow = false;
    options.convert_log1p = false;
    options.optimize_max_or_min_of_monotonic = false;
    options.fold_conjugate_into_transpose = false;
    options.fold_multiply_into_conv = false;
    options.fold_transpose_into_matmul = false;
    options.hoist_common_factor_out_of_aggregation = false;
    options.hoist_cwise_unary_chains = false;
    options.minimize_broadcasts = false;
    options.remove_identity_transpose = false;
    options.remove_involution = false;
    options.remove_idempotent = false;
    options.remove_redundant_bitcast = false;
    options.remove_redundant_cast = false;
    options.remove_redundant_reshape = false;
    options.remove_negation = false;
    options.remove_logical_not = false;
    options.reorder_cast_like_and_value_preserving = false;
    options.replace_mul_with_square = false;
    options.simplify_aggregation = false;
    options.unary_ops_composition = false;
    optimizer->options_ = options;
  }

  void DisableAddToAddNCombining(ArithmeticOptimizer* optimizer) {
    optimizer->options_.combine_add_to_addn = false;
  }

  void EnableOnlyAddToAddNCombining(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.combine_add_to_addn = true;
  }

  void EnableOnlyFoldConjugateIntoTranspose(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.fold_conjugate_into_transpose = true;
  }

  void EnableOnlyFoldMultipleIntoConv(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.fold_multiply_into_conv = true;
  }

  void EnableOnlyFoldTransposeIntoMatMul(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.fold_transpose_into_matmul = true;
  }

  void EnableOnlyHoistCommonFactor(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.hoist_common_factor_out_of_aggregation = true;
  }

  void EnableOnlyMinimizeBroadcasts(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.minimize_broadcasts = true;
  }

  void EnableOnlyRemoveIdentityTranspose(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_identity_transpose = true;
  }

  void EnableOnlyRemoveInvolution(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_involution = true;
  }

  void EnableOnlyRemoveRedundantBitcast(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_bitcast = true;
  }

  void EnableOnlyRemoveRedundantCast(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_cast = true;
  }

  void EnableOnlyRemoveRedundantReshape(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_redundant_reshape = true;
  }

  void EnableOnlyRemoveNegation(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_negation = true;
  }

  void EnableOnlyReorderCastAndTranspose(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.reorder_cast_like_and_value_preserving = true;
  }

  void EnableOnlyReplaceMulWithSquare(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.replace_mul_with_square = true;
  }

  void EnableOnlyHoistCWiseUnaryChains(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.hoist_cwise_unary_chains = true;
  }

  void EnableOnlySqrtDivToRsqrtMul(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.convert_sqrt_div_to_rsqrt_mul = true;
  }

  void EnableOnlyLogSoftmax(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.convert_log_softmax = true;
  }

  void EnableOnlyConvertPow(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.convert_pow = true;
  }

  void EnableOnlyFuseSquaredDiff(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.fuse_squared_diff = true;
  }

  void EnableOnlyRemoveIdempotent(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_idempotent = true;
  }

  void EnableOnlyRemoveLogicalNot(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_logical_not = true;
  }

  void EnableOnlySimplifyAggregation(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.simplify_aggregation = true;
  }

  void EnableOnlyLog1p(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.convert_log1p = true;
  }

  void EnableOnlyOptimizeMaxOrMinOfMonotonic(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.optimize_max_or_min_of_monotonic = true;
  }

  void EnableOnlyExpm1(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.convert_expm1 = true;
  }

  void EnableOnlyUnaryOpsComposition(ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.unary_ops_composition = true;
  }

  void EnableOnlyRemoveStackStridedSliceSameAxis(
      ArithmeticOptimizer* optimizer) {
    DisableAllStages(optimizer);
    optimizer->options_.remove_stack_strided_slice_same_axis = true;
  }
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_TEST_UTILS_H_
