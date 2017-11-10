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
    const NameAttrList& function, int num_constant_args,
    const std::vector<OptionalTensor>& variable_args, OpKernelContext* ctx,
    Signature* signature) {
  signature->name = Canonicalize(function.name(), AttrSlice(&function.attr()));
  signature->arg_values.resize(num_constant_args);

  signature->arg_types.reserve(ctx->num_inputs() - num_constant_args);

  // Inputs are in the order: constants, non-constants, resource variables.
  int input_num = 0;
  // Use the values of compile time constants in the signature->
  while (input_num < num_constant_args) {
    signature->arg_values[input_num] = ctx->input(input_num);
    ++input_num;
  }
  // Add the types and shapes of the remaining arguments.
  while (input_num < ctx->num_inputs() - variable_args.size()) {
    signature->arg_types.emplace_back(ctx->input_dtype(input_num),
                                      ctx->input(input_num).shape());
    ++input_num;
  }
  // For variable signatures, use the type and shape of the variable's
  // current value.
  for (const OptionalTensor& variable : variable_args) {
    TF_RET_CHECK(input_num < ctx->num_inputs());
    if (variable.present) {
      signature->arg_types.emplace_back(variable.value.dtype(),
                                        variable.value.shape());
    } else {
      signature->arg_types.emplace_back(DT_INVALID, TensorShape());
    }
    ++input_num;
  }
  return Status::OK();
}

namespace {

// Builds a XlaCompiler::Argument vector from the arguments to the _XlaLaunch
// op. The first `num_constant_args` arguments must be host-memory Tensors.
Status BuildArguments(int num_constant_args,
                      const std::vector<OptionalTensor>& variable_args,
                      OpKernelContext* ctx,
                      std::vector<XlaCompiler::Argument>* args) {
  args->resize(ctx->num_inputs());

  int input_num = 0;

  // Handles compile-time constants.
  TF_RET_CHECK(num_constant_args <= ctx->num_inputs());
  while (input_num < num_constant_args) {
    const Tensor& input = ctx->input(input_num);
    TF_RET_CHECK(input.dtype() != DT_RESOURCE);
    XlaCompiler::Argument& arg = (*args)[input_num];
    arg.kind = XlaCompiler::Argument::kConstant;
    arg.type = input.dtype();
    TF_RETURN_IF_ERROR(
        TensorShapeToXLAShape(input.dtype(), input.shape(), &arg.shape));
    arg.constant_value = input;
    ++input_num;
  }

  // Handles the non-constant arguments.
  int num_variable_args = variable_args.size();
  int num_nonconst_args =
      ctx->num_inputs() - num_variable_args - num_constant_args;
  TF_RET_CHECK(num_nonconst_args >= 0);
  while (input_num < num_constant_args + num_nonconst_args) {
    const Tensor& input = ctx->input(input_num);
    TF_RET_CHECK(input.dtype() != DT_RESOURCE);
    XlaCompiler::Argument& arg = (*args)[input_num];
    if (input.NumElements() > 0) {
      arg.kind = XlaCompiler::Argument::kParameter;
    } else {
      arg.kind = XlaCompiler::Argument::kConstant;
      arg.constant_value = input;
    }
    arg.type = input.dtype();
    TF_RETURN_IF_ERROR(
        TensorShapeToXLAShape(input.dtype(), input.shape(), &arg.shape));
    ++input_num;
  }

  // Handles resource variables.
  TF_RET_CHECK(input_num + num_variable_args == ctx->num_inputs());
  for (int variable_id = 0; variable_id < num_variable_args; ++variable_id) {
    const Tensor& input = ctx->input(input_num);
    TF_RET_CHECK(input.dtype() == DT_RESOURCE);

    XlaCompiler::Argument& arg = (*args)[input_num];

    arg.name = variable_args[variable_id].name;
    arg.kind = XlaCompiler::Argument::kResource;
    arg.resource_kind = XlaResource::kVariable;
    if (variable_args[variable_id].present) {
      const Tensor& value = variable_args[variable_id].value;
      arg.type = value.dtype();
      TF_RETURN_IF_ERROR(
          TensorShapeToXLAShape(value.dtype(), value.shape(), &arg.shape));
      arg.initialized = true;
    } else {
      // The values of uninitialized variables are not passed as inputs, since
      // they are meaningless. However, it is legal to assign to a resource
      // variable for the first time inside the XLA computation, so we do permit
      // uninitialized variables.
      arg.initialized = false;
      arg.type = DT_INVALID;
      arg.shape = xla::Shape();
    }
    ++input_num;
  }

  return Status::OK();
}

}  // namespace

Status XlaCompilationCache::BuildExecutable(
    const XlaCompiler::Options& options,
    const XlaCompiler::CompilationResult& result,
    std::unique_ptr<xla::LocalExecutable>* executable) {
  VLOG(2) << "Compiling to local executable";
  xla::Shape opaque_shape = xla::ShapeUtil::MakeOpaqueShape();

  std::vector<const xla::Shape*> argument_layouts(
      result.xla_input_shapes.size());
  for (int i = 0; i < result.xla_input_shapes.size(); ++i) {
    argument_layouts[i] = &result.xla_input_shapes[i];
  }
  if (result.requires_runtime_context) {
    // The final arg is the XlaLocalRuntimeContext*.
    argument_layouts.push_back(&opaque_shape);
  }
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(client_->default_device_ordinal());
  build_options.set_result_layout(result.xla_output_shape);

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
    int num_constant_args, const std::vector<OptionalTensor>& variable_args,
    OpKernelContext* ctx,
    const XlaCompiler::CompilationResult** compilation_result,
    xla::LocalExecutable** executable) {
  VLOG(1) << "XlaCompilationCache::Compile " << DebugString();

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "num_inputs=" << ctx->num_inputs()
            << " num_constant_args=" << num_constant_args
            << " num_variable_args=" << variable_args.size();
    for (int i = 0; i < ctx->num_inputs(); i++) {
      TensorShape shape = ctx->input(i).shape();
      VLOG(2) << i << ": dtype=" << DataTypeString(ctx->input_dtype(i))
              << " present=" << ctx->has_input(i)
              << " shape=" << shape.DebugString();
    }
    for (const OptionalTensor& variable : variable_args) {
      VLOG(2) << "variable present=" << variable.present
              << " type=" << DataTypeString(variable.value.dtype())
              << " shape=" << variable.value.shape().DebugString();
    }
    VLOG(2) << "num_outputs = " << ctx->num_outputs();
    for (int i = 0; i < ctx->num_outputs(); i++) {
      VLOG(2) << i << ": dtype=" << ctx->expected_output_dtype(i);
    }
  }

  TF_RET_CHECK(num_constant_args + variable_args.size() <= ctx->num_inputs());

  Signature signature;
  TF_RETURN_IF_ERROR(BuildSignature(function, num_constant_args, variable_args,
                                    ctx, &signature));

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
        BuildArguments(num_constant_args, variable_args, ctx, &args));

    XlaCompiler compiler(options);
    entry->compiled = true;
    entry->compilation_status =
        compiler.CompileFunction(XlaCompiler::CompileOptions(), function, args,
                                 &entry->compilation_result);
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
