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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/parallelization.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/analysis/cost_analysis.h"
#include "tensorflow/compiler/mlir/tfrt/constants.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tfrt/compiler/stream_analysis.h"  // from @tf_runtime

namespace tensorflow {
namespace mlrt_compiler {
namespace {

using tensorflow::tfrt_compiler::CostAnalysis;
using tfrt::compiler::Stream;
using tfrt::compiler::StreamAnalysis;

std::string GetStreamFunctionName(absl::string_view func_name,
                                  const Stream& stream) {
  return absl::StrCat(func_name, "_stream_", stream.id());
}

bool IsConstant(mlir::Operation* op) {
  return op && llvm::isa<mlir::TF::ConstOp, mlir::TF::_TfrtGetResourceOp>(op);
}

// StreamInfo is a bookkeeping for inputs, futures, and promises for a stream.
struct StreamInfo {
  const Stream* parent = nullptr;

  // The values that are produced by constant ops. Instead of using
  // promise/await to pass these values between streams, we can just copying
  // these ops to the streams that use these constants.
  llvm::SetVector<mlir::Value> constants;
  // The values that are the inputs to the stream.
  llvm::SetVector<mlir::Value> inputs;
  // The values that will be the futures to the stream.
  llvm::SetVector<mlir::Value> futures;
  // The values that will be the control futures (i.e. futures with no data) to
  // the stream.
  llvm::SetVector<mlir::Operation*> control_futures;
  // The values that will be the promises to the stream.
  llvm::SetVector<mlir::Value> promises;
  // The values that will be the control promises (i.e., promises with no data)
  // to the stream.
  llvm::SetVector<mlir::Operation*> control_promises;
  // The values that are defined by the operations in the stream. Note that all
  // values in `futures` will also be in `results`.
  llvm::DenseSet<mlir::Value> results;

  bool contains_only_constants = true;

  bool IsRoot() const { return parent == nullptr; }
};

// Preprocess the block to produce StreamInfo for every stream.
llvm::DenseMap<const Stream*, StreamInfo> PreprocessStreamInfo(
    mlir::Block& block,
    const llvm::DenseMap<mlir::Operation*,
                         llvm::SmallSetVector<mlir::Operation*, 4>>&
        control_predecessors,
    const StreamAnalysis& stream_analysis) {
  llvm::DenseMap<const Stream*, StreamInfo> stream_map;

  // All values that will be promises in the block.
  llvm::DenseSet<mlir::Value> promises;

  // All operations that will be control promises in the block.
  llvm::DenseSet<mlir::Operation*> control_promises;

  // Keep track of all available values and controls as we traverse the stream
  // tree in depth-first order.
  llvm::DenseSet<mlir::Value> available_values;
  llvm::DenseSet<mlir::Operation*> available_controls;

  struct Entry {
    explicit Entry(const Stream* stream) : stream(stream) {}

    const Stream* stream = nullptr;

    // Keep track of the next operation to be processed. If all operations are
    // processed, we can pop this stream from the DFS stack.
    int op_idx = 0;
  };

  std::vector<Entry> stack;
  stack.reserve(stream_analysis.GetNumStreams());

  // We first push the entry for the root stream.
  const auto& root_stream = stream_analysis.GetRootStream();
  auto& root_stream_info = stream_map[&root_stream];
  available_values.insert(block.getArguments().begin(),
                          block.getArguments().end());
  root_stream_info.results.insert(block.getArguments().begin(),
                                  block.getArguments().end());
  stack.push_back(Entry(&root_stream));

  // The root stream's first operation a dummy operation that defines all block
  // arguments.
  for (auto* child_stream : root_stream.GetChildStreamsForRootOp()) {
    stream_map[child_stream].parent = &root_stream;
    stack.push_back(Entry(child_stream));
  }

  // The first DFS traveral populates inputs and futures for every stream but
  // not promises. We only know whether a value definition is a promise only
  // after traversing all streams, so it is not possible to know it in the first
  // pass.
  while (!stack.empty()) {
    auto& [stream, op_idx] = stack.back();
    auto& stream_info = stream_map[stream];

    auto ops = stream->ops();

    // If we finish processing all operations in the stream, we can pop this
    // stream, as well as the values defined by its operations.
    if (op_idx == ops.size()) {
      for (auto* op : stream->ops()) {
        // Erase the values and controls produced by the current stream.
        for (auto result : op->getResults()) {
          available_values.erase(result);
        }
        available_controls.erase(op);
      }
      // Futures and control futures will also be available, so we erase them as
      // well.
      for (auto future : stream_info.futures) {
        available_values.erase(future);
      }
      for (auto* control_future : stream_info.control_futures) {
        available_controls.erase(control_future);
      }

      if (!stream_info.IsRoot()) {
        // Merge inputs, futures, and promises into the parent stream, as they
        // will be passed down from the root in the output program.
        DCHECK_GT(stream_map.count(stream_info.parent), 0);
        auto& parent_info = stream_map[stream_info.parent];

        for (const auto& input : stream_info.inputs) {
          DCHECK(available_values.contains(input));
          if (!parent_info.results.contains(input)) {
            // An input in the current stream will be an input in the parent
            // stream only if it is not a result in the parent stream.
            parent_info.inputs.insert(input);
          }
        }

        for (auto future : stream_info.futures) {
          DCHECK(!available_values.contains(future));
          parent_info.futures.insert(future);
        }
        for (auto* control_future : stream_info.control_futures) {
          DCHECK(!available_controls.contains(control_future));
          parent_info.control_futures.insert(control_future);
        }
      }

      // Update the global promise set.
      promises.insert(stream_info.futures.begin(), stream_info.futures.end());
      control_promises.insert(stream_info.control_futures.begin(),
                              stream_info.control_futures.end());

      stack.pop_back();
      continue;
    }

    // We process the operations one by one. If the operation has child streams,
    // we process the child streams first before continuing to the next
    // operation.
    bool has_child_streams = false;
    for (; op_idx < ops.size() && !has_child_streams; ++op_idx) {
      auto* op = ops[op_idx];

      stream_info.contains_only_constants &= IsConstant(op);

      // Check every operand to see whether it is a future or input.
      for (mlir::Value operand : op->getOperands()) {
        // If the value is defined in the current stream, nothing needs to be
        // done.
        if (!stream_info.results.contains(operand)) {
          if (available_values.insert(operand).second) {
            // If the operand is not available in the current stream or any
            // parent stream, it will be a future and then become a result.
            if (IsConstant(operand.getDefiningOp())) {
              stream_info.constants.insert(operand);
            } else {
              stream_info.futures.insert(operand);
            }
            stream_info.results.insert(operand);
          } else {
            // If the operand is not available in the current stream but
            // available in the parent stream, it is an input.
            if (IsConstant(operand.getDefiningOp())) {
              stream_info.constants.insert(operand);
            } else {
              stream_info.inputs.insert(operand);
            }
          }
        }
      }

      // Insert mlrt.await_control if this op has control deps on other ops.
      if (auto ctrl_iter = control_predecessors.find(op);
          ctrl_iter != control_predecessors.end()) {
        const auto& ctrl_deps = ctrl_iter->second;

        for (mlir::Operation* control_dep : ctrl_deps) {
          if (available_controls.insert(control_dep).second) {
            // If the control is not already available, it will be a control
            // future and then become available.
            stream_info.control_futures.insert(control_dep);
          }
        }
      }

      // Update results of this operations.
      for (mlir::Value result : op->getResults()) {
        available_values.insert(result);
        stream_info.results.insert(result);
      }

      // Update this op as an available control.
      available_controls.insert(op);

      // Pause processing the current stream to process the child streams first.
      const auto& child_streams = stream->GetChildStreams(op);
      has_child_streams = !child_streams.empty();
      for (auto* child_stream : child_streams) {
        stream_map[child_stream].parent = stream;
        stack.push_back(Entry(child_stream));
      }
    }
  }

  // The second pass populates promises for each stream. We also need to merge
  // promises in a child to its parent stream. We can do this by traversing the
  // operation in reverse program order.
  for (auto& op : llvm::reverse(block)) {
    const auto& stream = stream_analysis.GetStream(&op);
    auto& stream_info = stream_map[&stream];

    for (mlir::Value result : op.getResults()) {
      if (promises.contains(result)) {
        stream_info.promises.insert(result);
      }
    }

    if (control_promises.contains(&op)) {
      stream_info.control_promises.insert(&op);
    }

    for (const auto* child_stream : stream.GetChildStreams(&op)) {
      const auto& child_info = stream_map[child_stream];

      stream_info.promises.insert(child_info.promises.begin(),
                                  child_info.promises.end());
      stream_info.control_promises.insert(child_info.control_promises.begin(),
                                          child_info.control_promises.end());
    }
  }

  // Special handling for the dummy operation in the root.
  auto& root_info = stream_map[&root_stream];
  for (const auto* child_stream : root_stream.GetChildStreamsForRootOp()) {
    const auto& child_info = stream_map[child_stream];

    root_info.promises.insert(child_info.promises.begin(),
                              child_info.promises.end());
    root_info.control_promises.insert(child_info.control_promises.begin(),
                                      child_info.control_promises.end());
  }

  return stream_map;
}

// A custom struct that groups mappings for values, futures and promises for a
// stream during creating the corresponding stream function.
struct Mapping {
  // This is the mappings for the SSA values used in the original and new
  // operations.
  mlir::IRMapping value_mapping;

  // Maps the original tensor value that will be a future to the corresponding
  // !mlrt.future value.
  mlir::IRMapping future_mapping;

  // Maps the original tensor value that will be a promise to the corresponding
  // !mlrt.promise value.
  mlir::IRMapping promise_mapping;

  // In addition to value mappings, we also need mappings for input control
  // dependencies to the corresponding !mlrt.future and !mlrt.promise values.
  llvm::DenseMap<mlir::Operation*, mlir::Value> future_control_mapping;
  llvm::DenseMap<mlir::Operation*, mlir::Value> promise_control_mapping;
};

mlrt::compiler::AsyncOp CreateAsyncOp(
    mlir::OpBuilder& builder, absl::string_view function_name,
    const llvm::DenseMap<const Stream*, StreamInfo>& stream_map,
    const Stream& stream, const Mapping& mapping, mlir::Location loc) {
  auto iter = stream_map.find(&stream);
  DCHECK(iter != stream_map.end());
  const auto& stream_info = iter->second;

  if (stream_info.contains_only_constants) return nullptr;

  const auto& [value_mapping, future_mapping, promise_mapping,
               future_control_mapping, promise_control_mapping] = mapping;

  llvm::SmallVector<mlir::Value> async_operands;

  for (auto input : stream_info.inputs) {
    async_operands.push_back(value_mapping.lookup(input));
    DCHECK(async_operands.back());
  }

  for (auto future : stream_info.futures) {
    async_operands.push_back(future_mapping.lookup(future));
    DCHECK(async_operands.back());
  }

  for (auto* control_future : stream_info.control_futures) {
    DCHECK_GT(future_control_mapping.count(control_future), 0);
    async_operands.push_back(future_control_mapping.lookup(control_future));
    DCHECK(async_operands.back());
  }

  for (auto promise : stream_info.promises) {
    async_operands.push_back(promise_mapping.lookup(promise));
    DCHECK(async_operands.back());
  }

  for (auto* control_promise : stream_info.control_promises) {
    DCHECK_GT(promise_control_mapping.count(control_promise), 0);
    async_operands.push_back(promise_control_mapping.lookup(control_promise));
    DCHECK(async_operands.back());
  }

  return builder.create<mlrt::compiler::AsyncOp>(
      loc, builder.getType<mlrt::compiler::AsyncHandleType>(), async_operands,
      mlir::SymbolRefAttr::get(builder.getContext(),
                               GetStreamFunctionName(function_name, stream)));
}

mlir::func::FuncOp CreateStreamFunction(
    mlir::OpBuilder& builder, Mapping& mapping, absl::string_view name,
    const Stream& stream, const StreamInfo& stream_info, mlir::Location loc) {
  if (stream_info.contains_only_constants) return nullptr;

  auto& [value_mapping, future_mapping, promise_mapping, future_control_mapping,
         promise_control_mapping] = mapping;

  llvm::SmallVector<mlir::Type> arg_types;
  for (mlir::Value input : stream_info.inputs) {
    arg_types.push_back(input.getType());
  }

  arg_types.append(
      stream_info.futures.size() + stream_info.control_futures.size(),
      builder.getType<mlrt::compiler::FutureType>());
  arg_types.append(
      stream_info.promises.size() + stream_info.control_promises.size(),
      builder.getType<mlrt::compiler::PromiseType>());

  // The stream function has no result.
  auto func_type = builder.getFunctionType(arg_types, /*results=*/{});

  auto func = builder.create<mlir::func::FuncOp>(
      loc, GetStreamFunctionName(name, stream), func_type);
  func.setVisibility(mlir::func::FuncOp::Visibility::Private);

  // Populate the body of the stream function by copying over the operations
  // in the stream.
  auto* new_block = func.addEntryBlock();

  // Replace inputs with the function arguments.
  for (int i = 0; i < stream_info.inputs.size(); ++i) {
    value_mapping.map(stream_info.inputs[i], new_block->getArgument(i));
  }

  // Maps the original tensor value that will be a future or a promise to
  // the corresponding !mlrt.future or !mlrt.promise value.
  size_t start = stream_info.inputs.size();
  for (int i = 0; i < stream_info.futures.size(); ++i) {
    future_mapping.map(stream_info.futures[i],
                       new_block->getArgument(i + start));
  }

  start += stream_info.futures.size();
  for (int i = 0; i < stream_info.control_futures.size(); ++i) {
    future_control_mapping[stream_info.control_futures[i]] =
        new_block->getArgument(i + start);
  }

  start += stream_info.control_futures.size();
  for (int i = 0; i < stream_info.promises.size(); ++i) {
    promise_mapping.map(stream_info.promises[i],
                        new_block->getArgument(i + start));
  }

  start += stream_info.promises.size();
  for (int i = 0; i < stream_info.control_promises.size(); ++i) {
    promise_control_mapping[stream_info.control_promises[i]] =
        new_block->getArgument(i + start);
  }

  return func;
}

void CreateAllocateFuturesOp(mlir::OpBuilder& builder, Mapping& mapping,
                             const StreamInfo& stream_info,
                             mlir::Location loc) {
  auto& [value_mapping, future_mapping, promise_mapping, future_control_mapping,
         promise_control_mapping] = mapping;

  DCHECK_EQ(stream_info.futures.size(), stream_info.promises.size());

  llvm::SmallVector<mlir::Type> promise_types(
      stream_info.promises.size(),
      builder.getType<mlrt::compiler::PromiseType>());
  llvm::SmallVector<mlir::Type> future_types(
      stream_info.futures.size(),
      builder.getType<mlrt::compiler::FutureType>());

  if (!stream_info.futures.empty()) {
    auto allocate_futures = builder.create<tf_mlrt::AllocateFuturesOp>(
        loc, promise_types, future_types, stream_info.futures.size());
    for (int i = 0; i < stream_info.futures.size(); ++i) {
      future_mapping.map(stream_info.futures[i],
                         allocate_futures.getFutures()[i]);
    }

    for (int i = 0; i < stream_info.futures.size(); ++i) {
      // Use the original values in `futures` to make sure futures[i] shares the
      // state with promises[i].
      DCHECK(stream_info.promises.contains(stream_info.futures[i]));
      promise_mapping.map(stream_info.futures[i],
                          allocate_futures.getPromises()[i]);
    }
  }

  DCHECK_EQ(stream_info.control_futures.size(),
            stream_info.control_promises.size());
  if (!stream_info.control_futures.empty()) {
    promise_types.resize(stream_info.control_promises.size(),
                         builder.getType<mlrt::compiler::PromiseType>());
    future_types.resize(stream_info.control_futures.size(),
                        builder.getType<mlrt::compiler::FutureType>());

    auto allocate_control_futures =
        builder.create<mlrt::compiler::AllocateControlFuturesOp>(
            loc, promise_types, future_types,
            stream_info.control_futures.size());
    for (int i = 0; i < stream_info.control_futures.size(); ++i) {
      future_control_mapping[stream_info.control_futures[i]] =
          allocate_control_futures.getFutures()[i];
    }
    for (int i = 0; i < stream_info.control_futures.size(); ++i) {
      // Use the original operations in `control_futures` to make sure
      // control_futures[i] shares the state with control_promises[i].
      DCHECK(stream_info.control_promises.contains(
          stream_info.control_futures[i]));
      promise_control_mapping[stream_info.control_futures[i]] =
          allocate_control_futures.getPromises()[i];
    }
  }
}

class TensorflowCostModel : public StreamAnalysis::CostModelInterface {
 public:
  explicit TensorflowCostModel(CostAnalysis* cost_analysis)
      : cost_analysis_(*cost_analysis) {}

  std::optional<int64_t> GetOperationCost(mlir::Operation* op) const override {
    return cost_analysis_.GetCost(op);
  }

 private:
  const CostAnalysis& cost_analysis_;
};

bool SkipControlDep(mlir::Operation* op) {
  // TODO(chky): Consider define side effects more properly for these ops.
  return llvm::isa<mlir::TF::TPUCompileMlirAndExecuteOp, mlir::TF::AssertOp>(
      op);
}

void ParallelizeBlock(
    absl::string_view name, mlir::Block& block,
    const mlir::TF::SideEffectAnalysis::Info& side_effect_analysis,
    const tfrt_stub::CostRecorder* cost_recorder) {
  // First, we use SideEffectAnalysis to find out control predecessors for each
  // operation. We use this map later to insert control futures.
  llvm::DenseMap<mlir::Operation*, llvm::SmallSetVector<mlir::Operation*, 4>>
      control_predecessors;
  for (auto& op : block) {
    auto& deps = control_predecessors[&op];
    for (auto* dep : side_effect_analysis.DirectControlPredecessors(&op)) {
      // If we skip the control deps of `op`, then we need to use the control
      // deps of these control deps instead.
      if (SkipControlDep(dep)) {
        for (auto* d : control_predecessors[dep]) {
          DCHECK(!SkipControlDep(d));
          deps.insert(d);
        }
      } else {
        deps.insert(dep);
      }
    }
  }

  // Remove skipped control deps.
  for (auto& op : block) {
    if (SkipControlDep(&op)) {
      control_predecessors.erase(&op);
    }
  }

  // Perform stream analysis.
  CostAnalysis cost_analysis(
      llvm::cast<mlir::func::FuncOp>(block.getParentOp()), cost_recorder);
  TensorflowCostModel cost_model(&cost_analysis);
  StreamAnalysis stream_analysis(block, &cost_model);

  // Preprocess all streams to gather StreamInfos for all streams, without
  // modifying the program.
  llvm::DenseMap<const Stream*, StreamInfo> stream_map =
      PreprocessStreamInfo(block, control_predecessors, stream_analysis);

  // Then we perform a DFS traversal to create stream functions and insert async
  // operations.
  std::vector<const Stream*> stack;
  stack.reserve(stream_analysis.GetNumStreams());

  const auto& root_stream = stream_analysis.GetRootStream();
  stack.push_back(&root_stream);

  llvm::SmallVector<mlir::Operation*> to_remove;

  mlir::OpBuilder builder(block.getParentOp());

  while (!stack.empty()) {
    const auto* stream = stack.back();
    stack.pop_back();
    DCHECK(stream);

    DCHECK_GT(stream_map.count(stream), 0);
    const auto& stream_info = stream_map[stream];

    Mapping mapping;
    auto& [value_mapping, future_mapping, promise_mapping,
           future_control_mapping, promise_control_mapping] = mapping;

    // `async_handles` keeps the !mlrt.async_handle created in the stream. A
    // mlrt.await_handle op will be inserted at the end of the stream function
    // for each async handle.
    llvm::SmallVector<mlir::Value> async_handles;

    mlir::func::FuncOp stream_func;
    if (!stream_info.IsRoot()) {
      // If it is not a root stream, we need to create a new function for this
      // stream. And futures and promises are also passed as parameters. For the
      // root stream, futures and promises are allocated in the body.

      // Insert the stream function before the original function.
      builder.setInsertionPoint(block.getParentOp());

      stream_func =
          CreateStreamFunction(builder, mapping, name, *stream, stream_info,
                               block.getParentOp()->getLoc());

      if (stream_func) {
        // Set the insertion point to the start of the new block in the
        // function.
        builder.setInsertionPointToStart(&stream_func.front());
      }
    } else {
      stream_func = llvm::cast<mlir::func::FuncOp>(block.getParentOp());

      DCHECK_EQ(stream, &root_stream);
      // If it is the root stream, we insert new operations in the original
      // function. And we need to allocate all the futures used here.
      builder.setInsertionPointToStart(&block);

      // The block arguments of the root stream are in the `results`. There will
      // be no additional inputs in `inputs`.
      DCHECK(stream_info.inputs.empty());

      // Put the original arguments in the mapping as they are not changed.
      for (auto arg : block.getArguments()) {
        value_mapping.map(arg, arg);
      }

      // Insert a tf_mlrt.allocate_futures op to allocate all futures used.
      CreateAllocateFuturesOp(builder, mapping, stream_info,
                              block.getParentOp()->getLoc());

      // Lastly for the root stream, we need to handle the dummy op that defines
      // the arguments.
      for (const auto* child_stream : stream->GetChildStreamsForRootOp()) {
        stack.push_back(child_stream);
        if (auto async =
                CreateAsyncOp(builder, name, stream_map, *child_stream, mapping,
                              block.getParentOp()->getLoc())) {
          async_handles.push_back(async);
        }
      }
    }

    for (auto* op : stream->ops()) {
      to_remove.push_back(op);
    }

    // Skip empty streams.
    if (!stream_func) continue;

    mlir::Operation* return_op = nullptr;

    // Cloning the operations in the stream. If the operand is a future, a
    // tf_mlrt.Await op will be inserted. If the result is a promise, a
    // tf_mlrt.Promise will be inserted. Similar to control futures and control
    // promises.
    for (auto* op : stream->ops()) {
      // Clone the current op into the function of this stream, using the
      // new operands, which can be futures.
      for (mlir::Value operand : op->getOperands()) {
        if (stream_info.constants.contains(operand) &&
            !value_mapping.contains(operand)) {
          builder.clone(*operand.getDefiningOp(), value_mapping);
        } else if (stream_info.futures.contains(operand) &&
                   !value_mapping.contains(operand)) {
          // Insert Await op if it is a future.
          auto future_value = builder.create<tf_mlrt::TFAwaitOp>(
              op->getLoc(), operand.getType(), future_mapping.lookup(operand));

          // Now this future is available in the current stream, so it can be a
          // normal value.
          value_mapping.map(operand, future_value);
        }
      }

      if (auto ctrl_iter = control_predecessors.find(op);
          ctrl_iter != control_predecessors.end()) {
        const auto& ctrl_deps = ctrl_iter->second;

        for (mlir::Operation* control_dep : ctrl_deps) {
          // This control may be available in the ancestors or in a previous
          // AwaitControl, we only insert a new AwaitControl if it is not.
          if (stream_info.control_futures.contains(control_dep)) {
            if (auto iter = future_control_mapping.find(control_dep);
                iter != future_control_mapping.end()) {
              builder.create<mlrt::compiler::AwaitControlOp>(
                  control_dep->getLoc(), iter->second);

              // Now we no longer need this control dep in this stream.
              future_control_mapping.erase(iter);
            }
          }
        }
      }

      // Clone the op using the value mapping that includes values from futures.
      auto* new_op = builder.clone(*op, value_mapping);

      // TODO(chky): Ensure the original return op is in the root stream. This
      // is currently an implicit guarantee in stream analysis.
      if (llvm::isa<mlir::func::ReturnOp>(op)) {
        DCHECK(stream_info.IsRoot()) << name << " " << stream->id();
        return_op = new_op;
      }

      for (mlir::Value result : op->getResults()) {
        if (stream_info.promises.contains(result)) {
          // Insert Promise op if the result is a promise.
          builder.create<tf_mlrt::TFPromiseOp>(op->getLoc(),
                                               promise_mapping.lookup(result),
                                               value_mapping.lookup(result));
        }
      }

      if (stream_info.control_promises.contains(op)) {
        // Insert Promise op if this op produce a control dependency to ops in
        // other streams.
        builder.create<mlrt::compiler::PromiseControlOp>(
            op->getLoc(), promise_control_mapping[op]);
      }

      // If this op has child streams, insert mlrt.async ops.
      for (auto* child_stream : stream->GetChildStreams(op)) {
        stack.push_back(child_stream);
        if (auto async = CreateAsyncOp(builder, name, stream_map, *child_stream,
                                       mapping, op->getLoc())) {
          async_handles.push_back(async);
        }
      }
    }

    // Create the return op for non-root streams.
    //
    // TODO(chky): Ensure the original return op is in the root stream. This is
    // currently an implicit guarantee in stream analysis.
    if (!return_op) {
      DCHECK(!stream_info.IsRoot()) << name << " " << stream->id();
      return_op =
          builder.create<mlir::func::ReturnOp>(block.getParentOp()->getLoc());
    }

    // We need to wait for async executions at the end of the stream function,
    // in order to manage resource lifetime and handle errors properly. These
    // mlrt.await_handle ops are inserted before the return op.
    builder.setInsertionPoint(return_op);
    for (auto handle : async_handles) {
      builder.create<mlrt::compiler::AwaitHandleOp>(
          block.getParentOp()->getLoc(), handle);
    }
  }

  // Remove the operations in the original block.
  for (auto* op : llvm::reverse(to_remove)) {
    op->dropAllDefinedValueUses();
    op->erase();
  }
}

class ParallelizationPass
    : public mlir::PassWrapper<ParallelizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  ParallelizationPass() = default;
  ParallelizationPass(uint64_t cost_threshold,
                      bool merge_inter_dependent_streams,
                      const tfrt_stub::CostRecorder* cost_recorder) {
    cost_threshold_ = cost_threshold;
    merge_inter_dependent_streams_ = merge_inter_dependent_streams;
    cost_recorder_ = cost_recorder;
  }
  ParallelizationPass(const ParallelizationPass&) {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ParallelizationPass)

 private:
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<tensorflow::tf_mlrt::TensorflowMlrtDialect>();
    registry.insert<mlrt::compiler::MlrtDialect>();
  }

  llvm::StringRef getArgument() const final {
    return "tf-mlrt-parallelization";
  }

  llvm::StringRef getDescription() const final {
    return "Parallelize tf graphs by inserting mlrt async operations.";
  }

  void runOnOperation() override {
    auto module = getOperation();

    mlir::Builder builder(module);
    module->setAttr("tfrt.cost_threshold",
                    builder.getI64IntegerAttr(cost_threshold_));
    module->setAttr("tfrt.merge_inter_dependent_streams",
                    builder.getBoolAttr(merge_inter_dependent_streams_));

    mlir::TF::SideEffectAnalysis side_effect_analysis(module);

    for (auto func_op :
         llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
      ParallelizeBlock(func_op.getSymName(), func_op.front(),
                       side_effect_analysis.GetAnalysisForFunc(func_op),
                       cost_recorder_);
    }
  }

  Option<uint64_t> cost_threshold_{
      *this, "tfrt-cost-threshold",
      llvm::cl::desc("If a sequence of operations has a cost lower than the "
                     "cost-threshold, the sequence will be executed as a block "
                     "in the same thread."),
      llvm::cl::init(1)};
  Option<bool> merge_inter_dependent_streams_{
      *this, "tfrt-merge-inter-dependent-streams",
      llvm::cl::desc("If true, streams with inter data depenedencies will be "
                     "preferred to be merged for inline execution."),
      llvm::cl::init(false)};
  const tfrt_stub::CostRecorder* cost_recorder_ = nullptr;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateParallelizationPass(
    uint64_t cost_threshold, bool merge_inter_dependent_streams,
    const tfrt_stub::CostRecorder* cost_recorder) {
  return std::make_unique<ParallelizationPass>(
      cost_threshold, merge_inter_dependent_streams, cost_recorder);
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateParallelizationPass() {
  return std::make_unique<ParallelizationPass>();
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
