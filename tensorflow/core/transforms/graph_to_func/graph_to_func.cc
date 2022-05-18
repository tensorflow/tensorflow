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

#include "tensorflow/core/transforms/graph_to_func/graph_to_func.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/platform/errors.h"

using tensorflow::Status;
using tensorflow::errors::InvalidArgument;

namespace mlir {
namespace tfg {

// TODO(jpienaar): Move to helper header/this shouldn't be needed once we
// upgrade to C++17.
static inline absl::string_view ToStringView(llvm::StringRef ref) {
  return {ref.data(), ref.size()};
}

static std::string OpResultToSlotName(OpResult value) {
  return (TFOp(*value.getDefiningOp()).name() + ":" +
          Twine(value.cast<OpResult>().getResultNumber()))
      .str();
}

tensorflow::Status GraphToFunc(GraphOp graph, ArrayRef<Value> feeds,
                               ArrayRef<Value> fetches,
                               ArrayRef<Value> control_rets, StringRef name) {
  OpBuilder builder(graph);
  ControlType control_ty = ControlType::get(graph.getContext());
  llvm::SmallVector<Type> arg_types;
  llvm::SmallVector<Type> ret_types;
  for (Value feed : feeds) {
    arg_types.push_back(feed.getType());
    arg_types.push_back(control_ty);
  }
  for (Value fetch : fetches) ret_types.push_back(fetch.getType());
  FunctionType func_type = builder.getFunctionType(arg_types, ret_types);
  auto loc = graph.getLoc();
  auto func_op = builder.create<GraphFuncOp>(loc, name, func_type,
                                             /*generic=*/false);
  func_op->setAttr("tfg.lifted_graph_version", graph.version());
  func_op.getRegion().takeBody(graph.getRegion());
  Block *body = func_op.getBody();
  llvm::SmallVector<Attribute> args_rets_attrs;
  for (Value feed : feeds) {
    feed.replaceAllUsesWith(body->addArgument(feed.getType(), loc));
    body->addArgument(control_ty, loc);
    llvm::SmallVector<NamedAttribute> arg_attrs;
    std::string slot = OpResultToSlotName(feed.cast<OpResult>());
    arg_attrs.push_back(
        builder.getNamedAttr("tfg.name", builder.getStringAttr(slot)));
    args_rets_attrs.push_back(builder.getDictionaryAttr(arg_attrs));
    args_rets_attrs.push_back(Attribute{});
  }
  func_op.setAllArgAttrs(args_rets_attrs);

  args_rets_attrs.clear();
  for (Value fetch : fetches) {
    llvm::SmallVector<NamedAttribute> arg_attrs;
    std::string slot = OpResultToSlotName(fetch.cast<OpResult>());
    arg_attrs.push_back(
        builder.getNamedAttr("tfg.name", builder.getStringAttr(slot)));
    args_rets_attrs.push_back(builder.getDictionaryAttr(arg_attrs));
  }
  func_op.setAllResultAttrs(args_rets_attrs);

  OpBuilder body_builder = OpBuilder::atBlockEnd(func_op.getBody());
  body_builder.create<ReturnOp>(loc, fetches, control_rets);
  graph.erase();
  return Status::OK();
}

Status GraphToFunc(GraphOp graph, ArrayRef<std::string> feeds_names,
                   ArrayRef<std::string> fetches_names,
                   ArrayRef<std::string> control_rets_names, StringRef name) {
  DenseMap<StringRef, int> feeds_to_position;
  feeds_to_position.reserve(feeds_names.size());
  for (const auto &indexed_name : llvm::enumerate(feeds_names)) {
    const std::string &name = indexed_name.value();
    if (!feeds_to_position.insert({StringRef(name), indexed_name.index()})
             .second)
      return InvalidArgument("GraphToFunc: got duplicated feed name: ", name);
  }
  DenseMap<StringRef, int> fetches_to_position;
  fetches_to_position.reserve(fetches_names.size());
  for (const auto &indexed_name : llvm::enumerate(fetches_names)) {
    const std::string &name = indexed_name.value();
    if (feeds_to_position.count(name))
      return InvalidArgument("GraphToFunc: name is both a feed and a fetch: '",
                             name, "'");
    if (!fetches_to_position.insert({StringRef(name), indexed_name.index()})
             .second)
      return InvalidArgument("GraphToFunc: got duplicated fetch name: '", name,
                             "'");
  }
  DenseMap<StringRef, int> control_rets_to_position;
  control_rets_to_position.reserve(control_rets_names.size());
  for (const auto &indexed_name : llvm::enumerate(control_rets_names)) {
    if (!control_rets_to_position
             .insert({StringRef(indexed_name.value()), indexed_name.index()})
             .second)
      return InvalidArgument("GraphToFunc: got duplicated control_ret name: '",
                             indexed_name.value(), "'");
  }

  SmallVector<ValueRange> feeds(feeds_names.size());
  SmallVector<ValueRange> fetches(fetches_names.size());
  SmallVector<Value> control_rets(control_rets_names.size());
  for (Operation &op : *graph.getBody()) {
    TFOp tf_op(op);
    StringRef node_name = tf_op.name();

    // feeds and fetches are supposed to be mutually exclusive but control-ret
    // may include both of them.
    auto control_ret_pos = control_rets_to_position.find(node_name);
    if (control_ret_pos != control_rets_to_position.end()) {
      control_rets[control_ret_pos->second] = tf_op.controlRet();
    }
    auto feed_pos = feeds_to_position.find(node_name);
    if (feed_pos != feeds_to_position.end()) {
      feeds[feed_pos->second] = op.getResults().drop_back();
      continue;
    }
    auto fetch_pos = fetches_to_position.find(node_name);
    if (fetch_pos != fetches_to_position.end()) {
      fetches[fetch_pos->second] = op.getResults().drop_back();
      continue;
    }
  }

  for (const auto &feed_info : feeds_to_position) {
    if (feeds[feed_info.second].empty()) {
      (void)emitOptionalWarning(graph.getLoc(), "Can't find feed: '",
                                ToStringView(feed_info.first), "'");
    }
  }
  for (const auto &fetch_info : fetches_to_position) {
    if (fetches[fetch_info.second].empty()) {
      (void)emitOptionalWarning(graph.getLoc(), "Can't find fetch: '",
                                ToStringView(fetch_info.first), "'");
    }
  }
  for (const auto &control_ret_info : control_rets_to_position) {
    if (!control_rets[control_ret_info.second]) {
      (void)emitOptionalWarning(graph.getLoc(), "Can't find control rets: '",
                                ToStringView(control_ret_info.first), "'");
    }
  }

  SmallVector<Value> flattened_feeds;
  for (ValueRange values : feeds) {
    for (Value value : values) flattened_feeds.push_back(value);
  }

  SmallVector<Value> flattened_fetches;
  for (ValueRange values : fetches) {
    for (Value value : values) flattened_fetches.push_back(value);
  }

  llvm::erase_if(control_rets, [](Value value) { return !value; });

  return GraphToFunc(graph, flattened_feeds, flattened_fetches, control_rets,
                     name);
}

}  // namespace tfg
}  // namespace mlir
