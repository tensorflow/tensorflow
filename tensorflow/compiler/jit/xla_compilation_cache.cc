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

#include "tensorflow/compiler/jit/xla_compilation_cache.h"

#include <numeric>

#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

XlaCompilationCache::XlaCompilationCache(const XlaCompiler::Options& options)
    : compiler_(options) {}

XlaCompilationCache::~XlaCompilationCache() = default;

string XlaCompilationCache::DebugString() {
  return "XLA JIT compilation cache";
}

// Compute a string signature which encodes the shapes of the
// arguments in the supplied list.
string XlaCompilationCache::SignatureDebugString(const Signature& sig) {
  string result = sig.name;
  for (const auto& a : sig.arg_types) {
    strings::StrAppend(&result, ",", DataTypeString(a.first),
                       a.second.DebugString());
  }

  for (const auto& v : sig.arg_values) {
    strings::StrAppend(&result, "; ", v.first, ":", v.second.DebugString());
  }
  return result;
}

bool XlaCompilationCache::Signature::operator==(const Signature& other) const {
  if (name != other.name) return false;
  if (arg_types != other.arg_types) return false;

  if (arg_values.size() != other.arg_values.size()) return false;
  for (int i = 0; i < arg_values.size(); ++i) {
    if (arg_values[i].first != other.arg_values[i].first ||
        arg_values[i].second.tensor_data() !=
            other.arg_values[i].second.tensor_data()) {
      return false;
    }
  }
  return true;
}

uint64 XlaCompilationCache::Signature::Hash::operator()(
    const XlaCompilationCache::Signature& signature) const {
  uint64 h = std::hash<string>()(signature.name);
  for (const auto& arg : signature.arg_types) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, std::hash<int>()(arg.second.dims()));
    for (int dim : arg.second.dim_sizes()) {
      h = Hash64Combine(h, std::hash<int>()(dim));
    }
  }
  for (const auto& arg : signature.arg_values) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, Hash64(arg.second.tensor_data().data(),
                                arg.second.tensor_data().size()));
  }
  return h;
}

namespace {

// Builds a XlaCompiler::Argument vector from the arguments to the _XlaLaunch
// op. The first `num_constant_args` arguments must be host-memory Tensors.
std::vector<XlaCompiler::Argument> BuildArguments(int num_constant_args,
                                                  OpKernelContext* ctx) {
  std::vector<XlaCompiler::Argument> args(ctx->num_inputs());
  int parameter_num = 0;
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    args[i].type = ctx->input(i).dtype();
    args[i].shape = ctx->input(i).shape();
    if (i < num_constant_args || ctx->input(i).NumElements() == 0) {
      args[i].parameter = -1;
      args[i].constant_value = ctx->input(i);
    } else {
      args[i].parameter = parameter_num;
      ++parameter_num;
    }
  }
  return args;
}

}  // namespace

Status XlaCompilationCache::Compile(
    const NameAttrList& function, int num_constant_args, OpKernelContext* ctx,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::LocalExecutable** executable) {
  VLOG(1) << "XlaCompilationCache::Compile " << DebugString();

  if (VLOG_IS_ON(2)) {
    std::vector<string> argshapes;
    VLOG(2) << "num_inputs = " << ctx->num_inputs()
            << " num_constant_args= " << num_constant_args;
    for (int i = 0; i < ctx->num_inputs(); i++) {
      TensorShape shape = ctx->input(i).shape();
      VLOG(2) << i << ": dtype=" << ctx->input_dtype(i)
              << " present=" << ctx->has_input(i)
              << " shape=" << shape.DebugString();
      argshapes.push_back(shape.DebugString());
    }
    VLOG(2) << "num_outputs = " << ctx->num_outputs();
    for (int i = 0; i < ctx->num_outputs(); i++) {
      VLOG(2) << i << ": dtype=" << ctx->expected_output_dtype(i);
    }
  }
  Signature signature;
  signature.name = Canonicalize(function.name(), function.attr());
  for (int i = 0; i < ctx->num_inputs(); ++i) {
    signature.arg_types.emplace_back(ctx->input_dtype(i),
                                     ctx->input(i).shape());
    if (i < num_constant_args) {
      signature.arg_values.emplace_back(i, ctx->input(i));
    }
  }

  VLOG(2) << "Signature: " << SignatureDebugString(signature);
  // The outer lock protects the existence of the cache entry. It does not
  // protect the contents of the cache entry.
  Entry* entry;
  {
    mutex_lock lock(mu_);
    // Find or create a cache entry.
    std::unique_ptr<Entry>& e = cache_[signature];
    if (!e) {
      e.reset(new Entry);
    }
    entry = e.get();
  }

  // Acquire the cache entry lock and compile, if necessary.
  // TODO(phawkins): this locking will need to be restructured when we implement
  // cache eviction.
  mutex_lock entry_lock(entry->mu);
  if (!entry->compiled) {
    // Do the actual JIT compilation without holding the lock (it can take
    // a long time.)
    std::vector<XlaCompiler::Argument> args =
        BuildArguments(num_constant_args, ctx);

    std::unique_ptr<FunctionLibraryRuntime> flr(NewFunctionLibraryRuntime(
        compiler_.device_mgr(), ctx->env(), compiler_.device(),
        TF_GRAPH_DEF_VERSION,
        ctx->function_library()->GetFunctionLibraryDefinition(),
        OptimizerOptions(), nullptr /* custom_kernel_creator */));

    entry->compiled = true;
    entry->compilation_status = compiler_.CompileFunction(
        flr.get(), function, args, &entry->compilation_result);
  }
  *compilation_result = &entry->compilation_result;
  if (entry->compilation_status.ok() && executable) {
    if (entry->executable == nullptr &&
        !entry->compilation_result.computation.IsNull()) {
      entry->compilation_status = compiler_.BuildExecutable(
          entry->compilation_result, &entry->executable);
    }
    *executable = entry->executable.get();
  }

  Status status = entry->compilation_status;
  return status;
}

}  // namespace tensorflow
