/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

constexpr int64_t kDefaultCheapCost = 1;

int64_t GetRankedTensorSize(mlir::TensorType type) {
  auto shape = type.getShape();

  int64_t size = 1;
  for (int64_t dim : shape) {
    // For unknown dimensions, use 1 as the size because it is usually the batch
    // dimension.
    //
    // TODO(chky): Find out a better default number for this case.
    size *= std::max(kDefaultCheapCost, dim);
  }

  return size;
}

int64_t InferTensorSize(const CostContext& context, mlir::TensorType type) {
  if (type.hasRank()) return GetRankedTensorSize(type);
  return context.default_unranked_tensor_size;
}

// The cost function for tf.LookupTableFindV2.
int64_t InferLookupTableFindV2Cost(const CostContext& context,
                                   mlir::TF::LookupTableFindV2Op op) {
  // tf.LookupTableFindV2 ops are usually more costly than tf.AddV2 with the
  // same input size, as it involves more operations like hashing, map lookup,
  // etc.
  constexpr int64_t kLookupTableFindCostScale = 8;
  constexpr int64_t kLookupTableFindStringKeyCostScale = 16;

  auto value_type = op.getValues().getType().cast<mlir::TensorType>();
  auto key_type = op.getKeys().getType().cast<mlir::TensorType>();

  int64_t output_size = InferTensorSize(context, value_type);

  int64_t cost = kLookupTableFindCostScale * output_size;

  if (key_type.getElementType().isa<mlir::TF::StringType>())
    cost *= kLookupTableFindStringKeyCostScale;

  return cost;
}

// The cost function for tf.GatherV2.
int64_t InferGatherV2Cost(const CostContext& context, mlir::TF::GatherV2Op op) {
  return InferTensorSize(context,
                         op.getOutput().getType().cast<mlir::TensorType>());
}

// The cost function for tf.SparseSegmentSumOp.
template <typename OpType>
int64_t InferSparseSegmentOpCost(const CostContext& context, OpType op) {
  return InferTensorSize(
      context, op.getOutput().getType().template cast<mlir::TensorType>());
}

// CostFunctionRegistry is a map from op names to their cost functions.
using CostFunctionRegistry = absl::flat_hash_map<std::string, CostFunction>;

void RegisterCostFunction(CostFunctionRegistry& registry,
                          absl::string_view op_name,
                          CostFunction cost_function) {
  auto r = registry.try_emplace(op_name, std::move(cost_function));
  assert(r.second);
  (void)r;
}

template <typename OpType, typename F>
void RegisterCostFunction(CostFunctionRegistry& registry, F f) {
  RegisterCostFunction(
      registry, OpType::getOperationName().str(),
      [f = std::move(f)](const CostContext& context, mlir::Operation* op) {
        return f(context, llvm::cast<OpType>(op));
      });
}

CostFunctionRegistry& GetCostFunctionRegistry() {
  static auto* const registry = []() {
    auto* registry = new CostFunctionRegistry;
    // TODO(chky): Find a more scalable way to register cost functions. One
    // option is to incorporate it is TF MLIR ODS.
    RegisterCostFunction<mlir::TF::GatherV2Op>(*registry, InferGatherV2Cost);
    RegisterCostFunction<mlir::TF::SparseSegmentSumOp>(
        *registry, InferSparseSegmentOpCost<mlir::TF::SparseSegmentSumOp>);
    RegisterCostFunction<mlir::TF::SparseSegmentMeanOp>(
        *registry, InferSparseSegmentOpCost<mlir::TF::SparseSegmentMeanOp>);
    RegisterCostFunction<mlir::TF::SparseSegmentSqrtNOp>(
        *registry, InferSparseSegmentOpCost<mlir::TF::SparseSegmentSqrtNOp>);
    RegisterCostFunction<mlir::TF::LookupTableFindV2Op>(
        *registry, InferLookupTableFindV2Cost);
    return registry;
  }();
  return *registry;
}

}  // namespace

void RegisterCostFunction(absl::string_view op_name,
                          CostFunction cost_function) {
  RegisterCostFunction(GetCostFunctionRegistry(), op_name,
                       std::move(cost_function));
}

bool HasCostFunctionRegistered(absl::string_view op_name) {
  return GetCostFunctionRegistry().contains(op_name);
}

int64_t CostAnalysis::GetCost(mlir::Operation* op) const {
  assert(cost_map_.count(op) > 0);
  return cost_map_.lookup(op);
}

void CostAnalysis::AnalyzeArguments(mlir::func::FuncOp func_op) {
  // Use the max size among function inputs as the default size of dynamic
  // shaped tensors in the function.
  for (auto arg : func_op.getArguments()) {
    auto type = arg.getType().cast<mlir::TensorType>();
    if (type.hasRank()) {
      max_arg_size_ = std::max(max_arg_size_, GetRankedTensorSize(type));
    }
  }
}

void CostAnalysis::AnalyzeBlock(mlir::Block* block) {
  for (auto& op : *block) {
    EvaluateCost(&op);
  }
}

void CostAnalysis::EvaluateCost(mlir::Operation* op) {
  if (!llvm::isa<mlir::TF::TensorFlowDialect>(op->getDialect())) {
    cost_map_[op] = max_arg_size_;
    return;
  }

  // These ops are cheap regardless of their input sizes.
  //
  // TODO(chky): Find a more scalable way to figure out cheap ops.
  if (llvm::isa<mlir::TF::ShapeOp, mlir::TF::StridedSliceOp,
                mlir::TF::ReshapeOp, mlir::TF::ExpandDimsOp>(op)) {
    cost_map_[op] = kDefaultCheapCost;
    return;
  }

  // Try to use its cost function if it is registered.
  const auto& registry = GetCostFunctionRegistry();
  absl::string_view op_name = op->getName().getStringRef();
  auto iter = registry.find(op_name);
  if (iter != registry.end()) {
    CostContext context;
    context.default_unranked_tensor_size = max_arg_size_;
    cost_map_[op] = iter->second(context, op);
    return;
  }

  // For other ops, use the sum of input sizes as its cost.
  int64_t cost = kDefaultCheapCost;
  for (auto operand : op->getOperands()) {
    auto type = operand.getType().cast<mlir::TensorType>();
    if (type.hasRank()) {
      cost += GetRankedTensorSize(type);
    } else {
      // For unranked tensors, use the max size among the input tensors. This is
      // because the only dynamic information of the function should be the
      // input, so the size of dynamic tensors should be usually capped by
      // inputs' sizes.
      cost += max_arg_size_;
    }
  }

  cost_map_[op] = cost;
}

}  // namespace tfrt_compiler
}  // namespace tensorflow
