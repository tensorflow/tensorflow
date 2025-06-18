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
#include "tensorflow/core/tpu/kernels/sparse_core_ops_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/jit/flags.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/tpu/status_helper.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

std::vector<int> ConvertBinarySplitsToBucketSplits(int64 split,
                                                   int max_division_level) {
  std::vector<int> bucket_splits;
  uint32 current_index = 0;
  while (split > 0) {
    if (split % 2 == 1) {
      int split_level = absl::bit_width(current_index + 1) - 1;
      int split_offset = current_index - (1 << split_level) + 1;
      int split_size = 1 << (max_division_level - 1 - split_level);
      bucket_splits.push_back(split_size + split_offset * split_size * 2);
    }
    split >>= 1;
    current_index += 1;
  }
  absl::c_sort(bucket_splits);
  return bucket_splits;
}

int64 ConvertBucketSplitsToBinarySplits(std::vector<int> bucket_splits,
                                        int max_division_level) {
  int64 binary_splits = 0;
  for (auto& bucket_split : bucket_splits) {
    int split_level = max_division_level - 1;
    while (bucket_split > 0 && bucket_split % 2 == 0) {
      --split_level;
      bucket_split = bucket_split >> 1;
    }
    binary_splits |= (1LL << ((1 << split_level) - 1 + bucket_split / 2));
  }
  return binary_splits;
}

absl::Status ValidateInputCombiner(const std::string& combiner) {
  if (combiner != "sum" && combiner != "mean" && combiner != "sqrtn" &&
      !absl::StartsWith(combiner, "custom")) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid combiner: only \"sum\", \"mean\", \"sqrtn\", and "
                     "\"custom\" are supported, but got ",
                     combiner));
  }
  return absl::OkStatus();
}

std::function<float(float)> GetCombinerScaleContributionFunction(
    absl::string_view combiner) {
  if (combiner == "sum") {
    return [](float x) -> float { return 1.f; };
  } else if (combiner == "mean") {
    return [](float x) -> float { return x; };
  } else {  // combiner == "sqrtn"
    return [](float x) -> float { return x * x; };
  }
}

std::function<float(float)> GetCombinerScaleTransformFunction(
    absl::string_view combiner) {
  if (combiner == "sum") {
    return [](float x) -> float { return 1; };
  } else if (combiner == "mean") {
    return [](float x) -> float { return x == 0.0f ? 0.0f : 1.0 / x; };
  } else {  // combiner == "sqrtn"
    return
        [](float x) -> float { return x == 0.0f ? 0.0f : 1.0 / std::sqrt(x); };
  }
}

absl::Status GetMaxIdsAndUniquesExternal(
    const std::string& program_key, const std::string& table_name,
    int64_t num_samples_per_sparse_core, int64_t feature_width,
    int64_t* max_ids_per_partition, int64_t* max_unique_ids_per_partition) {
  SparseCore_GetMaxIdsAndUniques_Params params;
  params.program_key = program_key.c_str();
  params.table_name = table_name.c_str();
  params.num_samples_per_sparse_core = num_samples_per_sparse_core;
  params.feature_width = feature_width;
  StatusHelper status;
  params.status = status.c_status;

  stream_executor::tpu::OpsApiFn()->SparseCore_GetMaxIdsAndUniquesFn(&params);
  *max_ids_per_partition = params.max_ids_per_partition;
  *max_unique_ids_per_partition = params.max_unique_ids_per_partition;
  return status.status();
}

std::vector<std::vector<std::string>> GetTableStacks(
    const std::vector<int64_t>& table_height,
    const std::vector<int64_t>& table_width,
    const std::vector<int64_t>& table_num_samples,
    const std::vector<int64_t>& table_group,
    const std::vector<std::string>& table_names, int64_t num_tpu_chips) {
  if (GetDisableTableStacking()) {
    std::vector<std::vector<std::string>> stacks(table_names.size());
    for (int i = 0; i < table_names.size(); ++i) stacks[i] = {table_names[i]};
    return stacks;
  }

  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>>
      table_data(table_height.size());
  for (int i = 0; i < table_height.size(); ++i)
    table_data[i] =
        std::make_tuple(table_height[i], table_width[i], table_num_samples[i],
                        table_group[i], table_names[i]);

  // Sort tables by name so that we have a deterministic stacking.
  std::sort(table_data.begin(), table_data.end(), [](auto& lh, auto& rh) {
    return std::get<4>(lh) < std::get<4>(rh);
  });

  absl::flat_hash_map<int64_t, std::vector<std::vector<std::string>>>
      stacks_by_group;
  absl::flat_hash_map<int64_t, std::vector<int64_t>> stacks_height_by_group;
  absl::flat_hash_map<int64_t, std::vector<int64_t>> stacks_width_by_group;
  absl::flat_hash_map<int64_t, std::vector<int64_t>> stacks_samples_by_group;

  const int64_t mem_limit = GetXlaSparseCoreStackingMemLimit();
  const int64_t table_shard_limit = GetXlaSparseCoreStackingTableShardLimit();

  for (const auto& table : table_data) {
    int64_t height;
    int64_t width;
    int64_t num_samples;
    int64_t group;
    std::string name;
    std::tie(height, width, num_samples, group, name) = table;

    // Want per SparseCore samples.
    num_samples /= 4;

    // Find a stack to fit in. We need to stay under the limit on activation
    // sizes (if set) and the limit on table shard sizes (if set).
    int64_t stack_id = 0;
    for (; stack_id < stacks_by_group[group].size(); ++stack_id)
      if (((mem_limit == 0) ||
           (sizeof(float) * width *
                (num_samples + stacks_samples_by_group[group][stack_id]) <
            mem_limit)) &&
          ((table_shard_limit == 0) ||
           (sizeof(float) * (height + stacks_height_by_group[group][stack_id]) *
                width / num_tpu_chips <
            table_shard_limit)))
        break;

    // Create a new stack if we didn't find a stack to join.
    if (stack_id == stacks_by_group[group].size()) {
      stacks_by_group[group].resize(stacks_by_group[group].size() + 1);
      stacks_height_by_group[group].push_back(0);
      stacks_width_by_group[group].push_back(width);
      stacks_samples_by_group[group].push_back(0);
    }

    // Add the table to the stack and track the number of samples and height
    // of the table.
    stacks_by_group[group][stack_id].push_back(name);
    stacks_height_by_group[group][stack_id] += height;
    stacks_samples_by_group[group][stack_id] += num_samples;
  }

  // Merge all the stacks into one list.
  std::vector<std::vector<std::string>> table_stacks;
  for (const auto& [group, stacks] : stacks_by_group)
    table_stacks.insert(table_stacks.end(), stacks.begin(), stacks.end());

  return table_stacks;
}

ABSL_ATTRIBUTE_WEAK int GetMinibatchMaxDivisionLevel() {
  XlaSparseCoreFlags* sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_minibatch_max_division_level;
}

ABSL_ATTRIBUTE_WEAK bool GetDisableTableStacking() {
  XlaSparseCoreFlags* sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_disable_table_stacking;
}

ABSL_ATTRIBUTE_WEAK int64_t GetXlaSparseCoreStackingMemLimit() {
  XlaSparseCoreFlags* sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_stacking_mem_limit_bytes;
}

ABSL_ATTRIBUTE_WEAK int64_t GetXlaSparseCoreStackingTableShardLimit() {
  XlaSparseCoreFlags* sparse_core_flags = GetXlaSparseCoreFlags();
  return sparse_core_flags->tf_xla_sparse_core_stacking_table_shard_limit_bytes;
}

xla::XlaOp ApplyWeightClippingToTable(xla::XlaBuilder* builder,
                                      xla::XlaOp table, float clip_weight_min,
                                      float clip_weight_max) {
  xla::XlaOp clip_weight_min_op = xla::ConstantR0(builder, clip_weight_min);
  xla::XlaOp clip_weight_max_op = xla::ConstantR0(builder, clip_weight_max);
  xla::XlaOp clipped_table =
      xla::Clamp(clip_weight_min_op, table, clip_weight_max_op);
  return clipped_table;
}

xla::XlaComputation BuildSgdOptimizerComputation(const int32_t feature_width,
                                                 const float clip_weight_min,
                                                 const float clip_weight_max) {
  auto sgd_optimizer_builder =
      std::make_unique<xla::XlaBuilder>("sgd_optimizer_builder");

  xla::Shape per_row_shape =
      xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

  xla::XlaOp gradient =
      xla::Parameter(sgd_optimizer_builder.get(), 0, per_row_shape, "gradient");

  xla::XlaOp embedding_table = xla::Parameter(sgd_optimizer_builder.get(), 1,
                                              per_row_shape, "embedding_table");

  xla::XlaOp learning_rate = xla::Parameter(sgd_optimizer_builder.get(), 2,
                                            per_row_shape, "learning_rate");

  xla::XlaOp updated_embedding_table =
      embedding_table - learning_rate * gradient;

  // Apply the weight clipping.
  xla::XlaOp clipped_embedding_table = ApplyWeightClippingToTable(
      sgd_optimizer_builder.get(), updated_embedding_table, clip_weight_min,
      clip_weight_max);

  xla::XlaOp updated_tables =
      xla::Tuple(sgd_optimizer_builder.get(), {clipped_embedding_table});

  return sgd_optimizer_builder->Build(updated_tables).value();
}

xla::XlaComputation BuildAdagradOptimizerComputation(
    const int32_t feature_width, const float clip_weight_min,
    const float clip_weight_max) {
  auto adagrad_optimizer_builder =
      std::make_unique<xla::XlaBuilder>("adagrad_optimizer_builder");

  xla::Shape per_row_shape =
      xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

  xla::XlaOp gradient = xla::Parameter(adagrad_optimizer_builder.get(), 0,
                                       per_row_shape, "gradient");

  xla::XlaOp embedding_table = xla::Parameter(
      adagrad_optimizer_builder.get(), 1, per_row_shape, "embedding_table");

  xla::XlaOp accumulator = xla::Parameter(adagrad_optimizer_builder.get(), 2,
                                          per_row_shape, "accumulator");

  xla::XlaOp learning_rate = xla::Parameter(adagrad_optimizer_builder.get(), 3,
                                            per_row_shape, "learning_rate");

  xla::XlaOp new_accumulator = accumulator + gradient * gradient;

  xla::XlaOp updated_embedding_table =
      embedding_table - learning_rate * gradient / xla::Sqrt(new_accumulator);

  // Apply the weight clipping.
  xla::XlaOp clipped_embedding_table = ApplyWeightClippingToTable(
      adagrad_optimizer_builder.get(), updated_embedding_table, clip_weight_min,
      clip_weight_max);

  xla::XlaOp updated_tables =
      xla::Tuple(adagrad_optimizer_builder.get(),
                 {clipped_embedding_table, new_accumulator});
  return adagrad_optimizer_builder->Build(updated_tables).value();
}

xla::XlaComputation BuildAdagradMomentumOptimizerComputation(
    const int32_t feature_width, const bool use_nesterov, const float exponent,
    const float beta1, const float beta2, const float epsilon,
    const float clip_weight_min, const float clip_weight_max) {
  auto adagrad_momentum_optimizer_builder =
      std::make_unique<xla::XlaBuilder>("adagrad_momentum_optimizer_builder");

  xla::Shape per_row_shape =
      xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

  xla::XlaOp gradient = xla::Parameter(adagrad_momentum_optimizer_builder.get(),
                                       0, per_row_shape, "gradient");
  xla::XlaOp embedding_table =
      xla::Parameter(adagrad_momentum_optimizer_builder.get(), 1, per_row_shape,
                     "embedding_table");
  xla::XlaOp accumulator =
      xla::Parameter(adagrad_momentum_optimizer_builder.get(), 2, per_row_shape,
                     "accumulator");
  xla::XlaOp momenta = xla::Parameter(adagrad_momentum_optimizer_builder.get(),
                                      3, per_row_shape, "momenta");
  xla::XlaOp learning_rate =
      xla::Parameter(adagrad_momentum_optimizer_builder.get(), 4, per_row_shape,
                     "learning_rate");

  xla::XlaOp beta1_op =
      xla::ConstantR0(adagrad_momentum_optimizer_builder.get(), beta1);
  xla::XlaOp beta2_op =
      xla::ConstantR0(adagrad_momentum_optimizer_builder.get(), beta2);
  xla::XlaOp epsilon_op =
      xla::ConstantR0(adagrad_momentum_optimizer_builder.get(), epsilon);

  // If beta_2 == 1:
  //    accumulator(t) = accumulator(t-1) + gradient(t) ^ 2
  // Else:
  //    accumulator(t) = beta_2 * accumulator(t-1) +
  //                    (1-beta_2) * gradient(t) ^ 2
  xla::XlaOp exponent_op = xla::ConstantR0(
      adagrad_momentum_optimizer_builder.get(), 1.0f / exponent);
  xla::XlaOp one =
      xla::ConstantR0(adagrad_momentum_optimizer_builder.get(), 1.0f);

  xla::XlaOp new_accumulator = xla::Select(
      xla::Eq(beta2_op, one), accumulator + gradient * gradient,
      beta2_op * accumulator + (one - beta2_op) * gradient * gradient);

  // scaled_gradient = (accumulator + epsilon)^(-1/k) * gradient
  xla::XlaOp scaled_gradients =
      Pow(new_accumulator + epsilon_op, xla::Neg(exponent_op)) * gradient;

  // momenta(t) = beta1 * momenta(t-1) + scaled_gradient(t)
  xla::XlaOp new_momenta = beta1_op * momenta + scaled_gradients;

  // Table update:
  // non-nesterov: update = momenta_t
  // nesterov:     update = beta_1 * momenta_t + scaled_gradient
  // weights(t) = weights(t-1) - lr * update
  xla::XlaOp updated_embedding_table;
  if (use_nesterov) {
    updated_embedding_table =
        embedding_table -
        learning_rate * (beta1_op * new_momenta + scaled_gradients);
  } else {
    updated_embedding_table = embedding_table - learning_rate * new_momenta;
  }

  // Apply the weight clipping.
  xla::XlaOp clipped_embedding_table = ApplyWeightClippingToTable(
      adagrad_momentum_optimizer_builder.get(), updated_embedding_table,
      clip_weight_min, clip_weight_max);

  xla::XlaOp updated_tables =
      xla::Tuple(adagrad_momentum_optimizer_builder.get(),
                 {clipped_embedding_table, new_accumulator, new_momenta});
  return adagrad_momentum_optimizer_builder->Build(updated_tables).value();
}

xla::XlaComputation BuildAdamOptimizerComputation(
    const int32_t feature_width, const bool use_sum_inside_sqrt,
    const float beta1, const float beta2, const float epsilon,
    const float clip_weight_min, const float clip_weight_max) {
  auto adam_optimizer_builder =
      std::make_unique<xla::XlaBuilder>("adam_optimizer_builder");

  xla::Shape per_row_shape =
      xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

  xla::XlaOp gradient = xla::Parameter(adam_optimizer_builder.get(), 0,
                                       per_row_shape, "gradient");
  xla::XlaOp embedding_table = xla::Parameter(adam_optimizer_builder.get(), 1,
                                              per_row_shape, "embedding_table");
  xla::XlaOp momenta =
      xla::Parameter(adam_optimizer_builder.get(), 2, per_row_shape, "momenta");
  xla::XlaOp velocity = xla::Parameter(adam_optimizer_builder.get(), 3,
                                       per_row_shape, "velocity");
  xla::XlaOp learning_rate = xla::Parameter(adam_optimizer_builder.get(), 4,
                                            per_row_shape, "learning_rate");

  xla::XlaOp beta1_op = xla::ConstantR0(adam_optimizer_builder.get(), beta1);
  xla::XlaOp beta2_op = xla::ConstantR0(adam_optimizer_builder.get(), beta2);
  xla::XlaOp epsilon_op =
      xla::ConstantR0(adam_optimizer_builder.get(), epsilon);

  // Depending on sum_inside_sqrt, the denominator is either:
  //     sum_inside_sqrt==true: sqrt(v + eps^2)
  //     sum_inside_sqrt==false: sqrt(v) + eps
  // To simplify the for loop below, write the sqrt denominator as:
  //     sqrt(v + e1) + e2
  // and set e1 and e2 appropriately:
  xla::XlaOp zero = xla::ConstantR0(adam_optimizer_builder.get(), 0.0f);
  xla::XlaOp one = xla::ConstantR0(adam_optimizer_builder.get(), 1.0f);
  xla::XlaOp e1 = use_sum_inside_sqrt ? epsilon_op * epsilon_op : zero;
  xla::XlaOp e2 = use_sum_inside_sqrt ? zero : epsilon_op;

  // momentum(t) = beta_1 * momentum(t-1)
  //                      + (1-beta_1)*gradient(t)
  xla::XlaOp new_momenta = beta1_op * momenta + (one - beta1_op) * gradient;

  // velocity(t) = beta_2 * velocity(t-1)
  //                      + (1-beta_2)*gradient(t)*gradient(t)
  xla::XlaOp new_velocity =
      beta2_op * velocity + (one - beta2_op) * gradient * gradient;

  xla::XlaOp updated_embedding_table =
      embedding_table -
      learning_rate * new_momenta / (xla::Sqrt(new_velocity + e1) + e2);

  // Apply the weight clipping.
  xla::XlaOp clipped_embedding_table = ApplyWeightClippingToTable(
      adam_optimizer_builder.get(), updated_embedding_table, clip_weight_min,
      clip_weight_max);

  xla::XlaOp updated_tables =
      xla::Tuple(adam_optimizer_builder.get(),
                 {clipped_embedding_table, new_momenta, new_velocity});
  return adam_optimizer_builder->Build(updated_tables).value();
}

xla::XlaComputation BuildFtrlOptimizerComputation(
    int32_t feature_width, bool multiply_linear_by_learning_rate, float beta,
    float learning_rate_power, float l1_regularization_strength,
    float l2_regularization_strength, float clip_weight_min,
    float clip_weight_max) {
  auto ftrl_optimizer_builder =
      std::make_unique<xla::XlaBuilder>("ftrl_optimizer_builder");

  xla::Shape per_row_shape =
      xla::ShapeUtil::MakeShapeWithType<float>({1, feature_width});

  xla::XlaOp gradient = xla::Parameter(ftrl_optimizer_builder.get(), 0,
                                       per_row_shape, "gradient");

  xla::XlaOp embedding_table = xla::Parameter(ftrl_optimizer_builder.get(), 1,
                                              per_row_shape, "embedding_table");
  xla::XlaOp accumulator = xla::Parameter(ftrl_optimizer_builder.get(), 2,
                                          per_row_shape, "accumulator");
  xla::XlaOp linear =
      xla::Parameter(ftrl_optimizer_builder.get(), 3, per_row_shape, "linear");
  xla::XlaOp learning_rate = xla::Parameter(ftrl_optimizer_builder.get(), 4,
                                            per_row_shape, "learning_rate");

  // accumulator(t) = accumulator(t-1) + gradient(t) ^ 2
  xla::XlaOp new_accumulator = accumulator + gradient * gradient;

  xla::XlaOp learning_rate_power_op =
      xla::ConstantR0(ftrl_optimizer_builder.get(), learning_rate_power);

  xla::XlaOp power_old = Pow(accumulator, xla::Neg(learning_rate_power_op));
  xla::XlaOp power_new = Pow(new_accumulator, xla::Neg(learning_rate_power_op));
  xla::XlaOp delta_p = power_new - power_old;

  xla::XlaOp zero = xla::ConstantR0(ftrl_optimizer_builder.get(), 0.0f);

  xla::XlaOp two = xla::ConstantR0(ftrl_optimizer_builder.get(), 2.0f);

  xla::XlaOp l1_regularization_strength_op =
      xla::ConstantR0(ftrl_optimizer_builder.get(), l1_regularization_strength);

  xla::XlaOp l2_regularization_strength_op =
      xla::ConstantR0(ftrl_optimizer_builder.get(), l2_regularization_strength);

  xla::XlaOp beta_op = xla::ConstantR0(ftrl_optimizer_builder.get(), beta);

  // Note:
  //    min(|linear(t)|, lr*l1)*sgn(linear(t))
  // can be written as
  //    clamp( -lr*l1, linear(t), lr*l1)
  // assuming lr>0 and l1>0.
  xla::XlaOp new_linear;
  xla::XlaOp numer;
  xla::XlaOp denom;
  if (multiply_linear_by_learning_rate) {
    new_linear = linear + learning_rate * gradient - delta_p * embedding_table;
    // if multiply_linear:
    //   linear(t) = linear(t-1) + lr*g - delta_p * table(t-1)
    //   Update numerator:
    //      N = min(|linear(t)|, lr*l1)*sgn(linear(t)) - linear(t)
    //   Update denomninator:
    //      D = power(t) + 2*lr*l2 + beta
    //   table(t) = N / D
    numer = xla::Select(
        xla::Eq(l1_regularization_strength_op, zero), xla::Neg(new_linear),
        xla::Clamp(xla::Neg(learning_rate * l1_regularization_strength_op),
                   new_linear, learning_rate * l1_regularization_strength_op) -
            new_linear);
    denom = power_new + two * learning_rate * l2_regularization_strength_op +
            beta_op;
  } else {
    new_linear = linear + gradient - delta_p * embedding_table / learning_rate;
    // if NOT multiply_linear:
    //   linear(t) = linear(t-1) + g - (1/lr) * delta_p * table(t-1)
    //   Update numerator:
    //     N = min(|linear(t)|, l1)*sgn(linear(t)) - linear(t)
    //   Update denomninator:
    //     D = (1/lr) * (power(t) + beta) + 2*l2
    //   table(t) = N / D
    numer = xla::Select(xla::Eq(l1_regularization_strength_op, zero),
                        xla::Neg(new_linear),
                        xla::Clamp(xla::Neg(l1_regularization_strength_op),
                                   new_linear, l1_regularization_strength_op) -
                            new_linear);
    denom = (power_new + beta_op) / learning_rate +
            two * l2_regularization_strength_op;
  }
  xla::XlaOp updated_embedding_table = numer / denom;

  // Apply the weight clipping.
  xla::XlaOp clipped_embedding_table = ApplyWeightClippingToTable(
      ftrl_optimizer_builder.get(), updated_embedding_table, clip_weight_min,
      clip_weight_max);

  xla::XlaOp updated_tables =
      xla::Tuple(ftrl_optimizer_builder.get(),
                 {clipped_embedding_table, new_accumulator, new_linear});
  return ftrl_optimizer_builder->Build(updated_tables).value();
}

}  // namespace tensorflow
