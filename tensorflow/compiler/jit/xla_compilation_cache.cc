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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

XlaCompilationCache::XlaCompilationCache(xla::LocalClient* client,
                                         DeviceType device_type)
    : client_(client), device_type_(std::move(device_type)) {}
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
    strings::StrAppend(&result, "; ", v.DebugString());
  }
  return result;
}

bool XlaCompilationCache::Signature::operator==(const Signature& other) const {
  if (name != other.name) return false;
  if (arg_types != other.arg_types) return false;

  if (arg_values.size() != other.arg_values.size()) return false;
  for (int i = 0; i < arg_values.size(); ++i) {
    if (arg_values[i].tensor_data() != other.arg_values[i].tensor_data()) {
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
    h = Hash64Combine(
        h, Hash64(arg.tensor_data().data(), arg.tensor_data().size()));
  }
  return h;
}

Status XlaCompilationCache::BuildSignature(
    const NameAttrList& function, const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    Signature* signature) {
  signature->name = Canonicalize(function.name(), AttrSlice(&function.attr()));
  signature->arg_values.reserve(constant_args.size());

  signature->arg_types.reserve(ctx->num_inputs() - constant_args.size());

  for (int i = 0; i < ctx->num_inputs(); ++i) {
    if (constant_args.count(i) > 0) {
      // Use the values of compile time constants in the signature.
      signature->arg_values.push_back(constant_args.at(i));
    } else if (variable_args.count(i) > 0) {
      const OptionalTensor& variable = variable_args.at(i);
      if (variable.present) {
        signature->arg_types.emplace_back(variable.value.dtype(),
                                          variable.value.shape());
      } else {
        signature->arg_types.emplace_back(DT_INVALID, TensorShape());
      }
    } else {
      signature->arg_types.emplace_back(ctx->input_dtype(i),
                                        ctx->input(i).shape());
    }
  }
  return Status::OK();
}

namespace {

// Builds a XlaCompiler::Argument vector from the arguments to the XlaLaunch op.
Status BuildArguments(const std::map<int, Tensor>& constant_args,
                      const std::map<int, OptionalTensor>& variable_args,
                      OpKernelContext* ctx,
                      std::vector<XlaCompiler::Argument>* args) {
  args->resize(ctx->num_inputs());

  for (int64 input_num = 0; input_num < ctx->num_inputs(); ++input_num) {
    XlaCompiler::Argument& arg = (*args)[input_num];
    if (constant_args.count(input_num) > 0) {
      // Handles compile-time constants.
      const Tensor& input = constant_args.at(input_num);
      TF_RET_CHECK(input.dtype() != DT_RESOURCE);
      arg.kind = XlaCompiler::Argument::kConstant;
      arg.type = input.dtype();
      arg.shape = input.shape();
      arg.constant_value = input;
    } else if (variable_args.count(input_num) == 0) {
      // Handles the non-constant arguments.
      const Tensor& input = ctx->input(input_num);
      TF_RET_CHECK(input.dtype() != DT_RESOURCE);
      if (input.NumElements() > 0) {
        arg.kind = XlaCompiler::Argument::kParameter;
      } else {
        arg.kind = XlaCompiler::Argument::kConstant;
        arg.constant_value = input;
      }
      arg.type = input.dtype();
      arg.shape = input.shape();
    } else {
      // Handles resource variables.
      const Tensor& input = ctx->input(input_num);
      TF_RET_CHECK(input.dtype() == DT_RESOURCE);
      const OptionalTensor& variable = variable_args.at(input_num);
      arg.name = variable.name;
      arg.kind = XlaCompiler::Argument::kResource;
      arg.resource_kind = XlaResource::kVariable;
      if (variable.present) {
        const Tensor& value = variable.value;
        arg.type = value.dtype();
        arg.shape = value.shape();
        arg.initialized = true;
      } else {
        // The values of uninitialized variables are not passed as inputs, since
        // they are meaningless. However, it is legal to assign to a resource
        // variable for the first time inside the XLA computation, so we do
        // permit uninitialized variables.
        arg.initialized = false;
        arg.type = DT_INVALID;
        arg.shape = TensorShape();
      }
    }
  }

  return Status::OK();
}

}  // namespace

Status XlaCompilationCache::BuildExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result,
    std::unique_ptr<xla::LocalExecutable>* executable) {
  VLOG(2) << "Compiling to local executable";

  std::vector<const xla::Shape*> argument_layouts(
      result.xla_input_shapes.size());
  for (int i = 0; i < result.xla_input_shapes.size(); ++i) {
    argument_layouts[i] = &result.xla_input_shapes[i];
  }
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(client_->default_device_ordinal());
  build_options.set_result_layout(result.xla_output_shape);
  build_options.set_device_allocator(options.device_allocator);
  build_options.set_resource_update_count(result.resource_updates.size());

  auto compile_result =
      client_->Compile(*result.computation, argument_layouts, build_options);
  if (!compile_result.ok()) {
    return compile_result.status();
  }
  *executable = std::move(compile_result.ValueOrDie());
  return Status::OK();
}

Status XlaCompilationCache::Compile(
    const XlaCompiler::Options& options, const NameAttrList& function,
    const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::LocalExecutable** executable,
    const XlaCompiler::CompileOptions* compile_options) {
  return CompileImpl(options, function, constant_args, variable_args, ctx,
                     compilation_result, executable, compile_options, false);
}

Status XlaCompilationCache::CompileSingleOp(
    const XlaCompiler::Options& options,
    const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::LocalExecutable** executable,
    const XlaCompiler::CompileOptions* compile_options) {
  const NodeDef& def = ctx->op_kernel().def();
  NameAttrList name;
  name.set_name(def.op());
  *name.mutable_attr() = def.attr();
  return CompileImpl(options, name, constant_args, variable_args, ctx,
                     compilation_result, executable, compile_options, true);
}

Status XlaCompilationCache::CompileImpl(
    const XlaCompiler::Options& options, const NameAttrList& function,
    const std::map<int, Tensor>& constant_args,
    const std::map<int, OptionalTensor>& variable_args, OpKernelContext* ctx,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::LocalExecutable** executable,
    const XlaCompiler::CompileOptions* compile_options,
    bool compile_single_op) {
  VLOG(1) << "XlaCompilationCache::Compile " << DebugString();

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "num_inputs=" << ctx->num_inputs()
            << " num_constant_args=" << constant_args.size()
            << " num_variable_args=" << variable_args.size();
    for (int i = 0; i < ctx->num_inputs(); i++) {
      TensorShape shape = ctx->input(i).shape();
      VLOG(2) << i << ": dtype=" << DataTypeString(ctx->input_dtype(i))
              << " present=" << ctx->has_input(i)
              << " shape=" << shape.DebugString();
    }
    for (auto& iterator : variable_args) {
      const OptionalTensor& variable = iterator.second;
      VLOG(2) << "variable present=" << variable.present
              << " type=" << DataTypeString(variable.value.dtype())
              << " shape=" << variable.value.shape().DebugString()
              << " TF arg= " << iterator.first;
    }
    VLOG(2) << "num_outputs = " << ctx->num_outputs();
    for (int i = 0; i < ctx->num_outputs(); i++) {
      VLOG(2) << i << ": dtype=" << ctx->expected_output_dtype(i);
    }
  }

  TF_RET_CHECK(constant_args.size() + variable_args.size() <=
               ctx->num_inputs());

  Signature signature;
  TF_RETURN_IF_ERROR(
      BuildSignature(function, constant_args, variable_args, ctx, &signature));

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
    VLOG(1) << "Compilation cache miss for signature: "
            << SignatureDebugString(signature);
    // Do the actual JIT compilation without holding the lock (it can take
    // a long time.)
    std::vector<XlaCompiler::Argument> args;
    TF_RETURN_IF_ERROR(
        BuildArguments(constant_args, variable_args, ctx, &args));

    XlaCompiler compiler(options);
    entry->compiled = true;

    if (compile_single_op) {
      entry->compilation_status = compiler.CompileSingleOp(
          compile_options ? *compile_options : XlaCompiler::CompileOptions(),
          signature.name, ctx, args, &entry->compilation_result);
    } else {
      entry->compilation_status = compiler.CompileFunction(
          compile_options ? *compile_options : XlaCompiler::CompileOptions(),
          function, args, &entry->compilation_result);
    }
  }
  *compilation_result = &entry->compilation_result;
  if (entry->compilation_status.ok() && executable) {
    if (entry->executable == nullptr) {
      entry->compilation_status = BuildExecutable(
          options, entry->compilation_result, &entry->executable);
    }
    *executable = entry->executable.get();
  }

  Status status = entry->compilation_status;
  return status;
}

}  // namespace tensorflow
