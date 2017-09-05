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

#include "tensorflow/compiler/tf2xla/xla_compiler.h"

#include <numeric>

#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

// Checks that arguments `args` match types `types`.
Status CheckSignature(const DataTypeVector& types,
                      const std::vector<XlaCompiler::Argument>& args) {
  if (args.size() != types.size()) {
    return errors::Internal("Compilation arguments have ", args.size(),
                            " elements while function has ", types.size());
  }
  for (int i = 0; i < types.size(); ++i) {
    if (types[i] != args[i].type && types[i] != DT_RESOURCE) {
      return errors::Internal(
          "Argument ", i, " has declared type ", DataTypeString(args[i].type),
          " but function parameter has type ", DataTypeString(types[i]));
    }
  }
  return Status::OK();
}

}  // namespace

bool XlaCompiler::Argument::operator==(
    const XlaCompiler::Argument& other) const {
  if (std::tie(kind, type, name, tensor_array_size) !=
      std::tie(other.kind, other.type, other.name, other.tensor_array_size)) {
    return false;
  }
  if (!xla::ShapeUtil::Equal(shape, other.shape)) {
    return false;
  }
  if (constant_value.shape() != other.constant_value.shape()) {
    return false;
  }
  return constant_value.tensor_data() == other.constant_value.tensor_data();
}

XlaCompiler::XlaCompiler(XlaCompiler::Options options)
    : options_(options),
      initialization_status_(Status::OK()),
      next_step_id_(1),
      device_(
          new XlaCompilationDevice(SessionOptions(), *options_.device_type)),
      device_mgr_({device_}) {
  // We no longer need the device_type.
  options_.device_type = nullptr;

  if (options_.populate_resource_manager) {
    initialization_status_ =
        (*options_.populate_resource_manager)(device_->resource_manager());
  }

  local_flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(),

                                                      FunctionDefLibrary{}));
  local_pflr_.reset(new ProcessFunctionLibraryRuntime(
      &device_mgr_, Env::Default(), options.graph_def_version,
      local_flib_def_.get(), OptimizerOptions(),
      nullptr /* custom_kernel_creator */));
  pflr_.reset(new ProcessFunctionLibraryRuntime(
      &device_mgr_, Env::Default(), options.graph_def_version, options.flib_def,
      OptimizerOptions(), nullptr /* custom_kernel_creator */));

  local_flib_runtime_ = local_pflr_->GetFLR(device_->name());
  flib_runtime_ = pflr_->GetFLR(device_->name());
}

XlaCompiler::~XlaCompiler() = default;

int64 XlaCompiler::NextStepId() {
  return next_step_id_++;
}

uint64 XlaCompiler::SignatureHash::operator()(
    const std::pair<string, std::vector<Argument>>& signature) const {
  return std::hash<string>()(signature.first);
}

static Status GetFunctionBody(const NameAttrList& function,
                              FunctionLibraryRuntime* flib_runtime,
                              const FunctionBody** fbody) {
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(flib_runtime->Instantiate(
      function.name(), AttrSlice(&function.attr()), &handle));

  *fbody = flib_runtime->GetFunctionBody(handle);
  TF_RET_CHECK(*fbody);
  return Status::OK();
}

Status XlaCompiler::CompileFunction(
    const XlaCompiler::CompileOptions& options, const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args,
    XlaCompiler::CompilationResult* result) {
  const string function_id =
      Canonicalize(function.name(), AttrSlice(&function.attr()));
  VLOG(1) << "XlaCompiler::CompileFunction " << function_id;

  auto it = cache_.find({function_id, args});
  if (it != cache_.end()) {
    *result = it->second;
    return Status::OK();
  }

  const FunctionBody* fbody;
  if (!GetFunctionBody(function, local_flib_runtime_, &fbody).ok()) {
    TF_RETURN_IF_ERROR(GetFunctionBody(function, flib_runtime_, &fbody));
  }

  TF_RETURN_IF_ERROR(CheckSignature(fbody->arg_types, args));

  std::unique_ptr<Graph> graph(new Graph(options_.flib_def));
  CopyGraph(*fbody->graph, graph.get());

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "XlaCompiler::CompileFunction: "
            << dump_graph::DumpGraphToFile(
                   strings::StrCat("xla_compile_function_", function_id),
                   *graph);
  }

  // Optimize the graph before running the compiler.
  OptimizerOptions opts;
  opts.set_do_common_subexpression_elimination(true);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(true);
  GraphOptimizer optimizer(opts);
  optimizer.Optimize(flib_runtime_, flib_runtime_->env(),
                     /*device=*/nullptr, &graph, /*shape_map=*/nullptr);

  VLOG(1) << "====================================================";
  TF_RETURN_IF_ERROR(
      CompileGraph(options, function_id, std::move(graph), args, result));
  VLOG(1) << "====================================================";

  cache_[{function_id, args}] = *result;
  return Status::OK();
}

Status XlaCompiler::BuildExecutable(
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
  xla::LocalClient* local_client = static_cast<xla::LocalClient*>(client());
  xla::ExecutableBuildOptions build_options;
  build_options.set_device_ordinal(local_client->default_device_ordinal());
  build_options.set_platform(local_client->platform());
  build_options.set_result_layout(result.xla_output_shape);
  build_options.set_has_hybrid_result(
      options_.local_executable_has_hybrid_result);

  auto compile_result = local_client->Compile(*result.computation,
                                              argument_layouts, build_options);
  if (!compile_result.ok()) {
    return compile_result.status();
  }
  *executable = std::move(compile_result.ValueOrDie());
  return Status::OK();
}

namespace {

Status ExecuteGraph(XlaContext* xla_context, std::unique_ptr<Graph> graph,
                    XlaCompilationDevice* device, FunctionLibraryRuntime* flib,
                    int64 step_id) {
  // Resource cleanup is a bit messy. XlaContext is a ref-counted resource; the
  // resource manager takes ownership via Create, and unrefs via Cleanup.  We
  // explicitly add a reference to ensure the refcount at entry is maintained at
  // all exit points; Create and Cleanup are always called in this function.
  //
  // The Executor requires us to use ScopedStepContainer. We wrap it in a
  // unique_ptr so we can capture the cleanup status in the end.
  xla_context->Ref();
  Status cleanup_status;
  auto step_container = xla::MakeUnique<ScopedStepContainer>(
      step_id, [&cleanup_status, device](const string& name) {
        cleanup_status = device->resource_manager()->Cleanup(name);
      });
  TF_RETURN_IF_ERROR(device->resource_manager()->Create(
      step_container->name(), XlaContext::kXlaContextResourceName,
      xla_context));

  // Create a LocalExecutor that will own and run the graph.
  LocalExecutorParams exec_params;
  exec_params.device = device;
  exec_params.function_library = flib;
  exec_params.create_kernel = [flib](const NodeDef& ndef, OpKernel** kernel) {
    return flib->CreateKernel(ndef, kernel);
  };
  exec_params.delete_kernel = [](OpKernel* kernel) { delete kernel; };
  Executor* exec_ptr = nullptr;
  TF_RETURN_IF_ERROR(NewLocalExecutor(exec_params, graph.release(), &exec_ptr));
  std::unique_ptr<Executor> exec(exec_ptr);
  // At this point ownership of the graph has been transferred to exec.

  // Run the graph symbolically, turning the graph into an XLA computation.
  Executor::Args exec_args;
  exec_args.step_id = step_id;
  exec_args.step_container = step_container.get();
  // Run all compilation kernels on the main thread.
  exec_args.runner = [](Executor::Args::Closure c) { c(); };
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      exec->Run(exec_args),
      "Conversion from TensorFlow graph to XLA computation failed.");

  // Explicitly clean up the step container, to capture the cleanup status.
  step_container.reset();
  return cleanup_status;
}

// Builds XLA computations for each of the arguments to the computation.
// `args` are the arguments to the computation.
Status BuildArguments(const std::vector<XlaCompiler::Argument>& args,
                      bool use_tuple_arg, xla::ComputationBuilder* builder,
                      std::vector<XlaContext::Argument>* context_args,
                      std::vector<int>* input_mapping,
                      std::vector<xla::Shape>* input_shapes) {
  context_args->resize(args.size());

  // Argument numbers of arguments and resources that are to be passed to the
  // XLA computation as runtime parameters.
  std::vector<int> parameters, resources;
  parameters.reserve(args.size());
  resources.reserve(args.size());

  for (std::vector<XlaCompiler::Argument>::size_type i = 0; i < args.size();
       ++i) {
    XlaContext::Argument& context_arg = (*context_args)[i];
    context_arg.kind = args[i].kind;
    context_arg.name = args[i].name;
    context_arg.value.constant_value = args[i].constant_value;
    context_arg.value.type = args[i].type;

    switch (args[i].kind) {
      case XlaCompiler::Argument::kVariable:
      case XlaCompiler::Argument::kTensorArray:
      case XlaCompiler::Argument::kStack:
        context_arg.is_resource = true;
        if (args[i].initialized) {
          resources.push_back(i);
          context_arg.value.is_constant = false;
        } else {
          context_arg.value.is_constant = true;
        }
        context_arg.tensor_array_size = args[i].tensor_array_size;
        break;
      case XlaCompiler::Argument::kParameter:
        parameters.push_back(i);
        context_arg.value.is_constant = false;
        break;
      case XlaCompiler::Argument::kConstant:
        context_arg.value.is_constant = true;
        break;
      case XlaCompiler::Argument::kInvalid:
        return errors::Internal("Unreachable case in BuildArguments()");
    }
  }

  // Append parameters containing variable values after the other runtime
  // parameters.
  parameters.insert(parameters.end(), resources.begin(), resources.end());
  if (parameters.empty()) {
    return Status::OK();
  }

  input_shapes->resize(parameters.size());
  input_mapping->resize(parameters.size());
  for (std::vector<int>::size_type i = 0; i < input_shapes->size(); ++i) {
    const XlaCompiler::Argument& arg = args[parameters[i]];
    // Computes the shapes of non-constant arguments.
    (*input_shapes)[i] = arg.shape;
    (*input_mapping)[i] = parameters[i];
  }

  if (use_tuple_arg) {
    xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(*input_shapes);
    xla::ComputationDataHandle tuple =
        builder->Parameter(0, tuple_shape, "arg_tuple");
    for (std::vector<int>::size_type i = 0; i < input_shapes->size(); ++i) {
      (*context_args)[parameters[i]].value.handle =
          builder->GetTupleElement(tuple, i);
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < input_shapes->size(); ++i) {
      (*context_args)[parameters[i]].value.handle =
          builder->Parameter(i, (*input_shapes)[i], strings::StrCat("arg", i));
    }
  }
  return Status::OK();
}

// Builds the XLA computation.
//
// `retvals` is the list of retvals produced by _Retval operators, in index
// order. `variable_map` is a map from variable ID numbers to XlaOpContext
// variable states, generated by the symbolic evaluation.
// If `has_side_effects` is true, the computation has side effects and should be
// built even if it has no outputs.
// If `return_updated_values_for_all_resources` is true, all resources will be
// included in `resource_updates`, regardless of whether their value changed.
// Sets `*num_nonconst_outputs` to the number of outputs of the `computation`.
// Sets `*resource_updates` to a description of resources whose values are
// written by the computation; the variable writes are the last
// `resource_updates.size()` return values from the computation. Each entry in
// `resource_updates` is a (input_index, type) pair, where `input_index` is the
// index of a resource variable argument to the computation, and `type` is the
// type of the final output.
Status BuildComputation(
    const std::vector<XlaContext::HandleOrConstant>& retvals,
    const std::vector<std::unique_ptr<XlaResource>>& resources,
    bool has_side_effects, bool return_updated_values_for_all_resources,
    xla::ComputationBuilder* builder, xla::Computation* computation,
    int* num_computation_outputs, int* num_nonconst_outputs,
    std::vector<XlaCompiler::ResourceUpdate>* resource_updates) {
  std::vector<xla::ComputationDataHandle> elems;
  elems.reserve(retvals.size());
  for (const XlaContext::HandleOrConstant& retval : retvals) {
    if (!retval.is_constant) {
      elems.push_back(retval.handle);
    }
  }
  *num_nonconst_outputs = elems.size();

  // Add return values for resources whose values have changed.
  std::vector<const XlaResource*> arg_vars;
  arg_vars.reserve(resources.size());
  for (const auto& var : resources) {
    if (var->arg_num >= 0) {
      arg_vars.push_back(var.get());
    }
  }
  std::sort(arg_vars.begin(), arg_vars.end(),
            [](const XlaResource* a, const XlaResource* b) {
              return a->arg_num < b->arg_num;
            });

  for (const XlaResource* var : arg_vars) {
    bool modified = var->value.handle() != var->initial_value.handle();
    if (return_updated_values_for_all_resources || modified) {
      resource_updates->emplace_back();
      XlaCompiler::ResourceUpdate& update = resource_updates->back();
      update.input_index = var->arg_num;
      update.type = var->type;
      update.modified = modified;
      elems.push_back(var->value);
    }
  }

  *num_computation_outputs = elems.size();
  if (!elems.empty() || has_side_effects) {
    // Builds a empty tuple return value for computations that have side effects
    // but have no return values.
    xla::ComputationDataHandle handle = builder->Tuple(elems);

    // TODO(b/31775371): to workaround bug, we must build a no-op computation
    // that is guaranteed to be constructed after all of the formal parameters
    // to the computation. Once the bug is fixed, we could avoid tupling here.
    if (elems.size() == 1) {
      handle = builder->GetTupleElement(handle, 0);
    }

    // Builds the XLA computation.
    xla::StatusOr<xla::Computation> computation_status = builder->Build();
    if (!computation_status.ok()) {
      return computation_status.status();
    }
    *computation = computation_status.ConsumeValueOrDie();
  }
  return Status::OK();
}

void AssignMajorToMinorLayout(xla::Shape* shape) {
  if (xla::ShapeUtil::IsTuple(*shape)) {
    for (xla::Shape& elem_shape : *shape->mutable_tuple_shapes()) {
      AssignMajorToMinorLayout(&elem_shape);
    }
  } else {
    auto& minor_to_major = *shape->mutable_layout()->mutable_minor_to_major();
    minor_to_major.Resize(xla::ShapeUtil::Rank(*shape), 0);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  }
}

}  // namespace

Status XlaCompiler::CompileGraph(const XlaCompiler::CompileOptions& options,
                                 string const& name,
                                 std::unique_ptr<Graph> graph,
                                 const std::vector<XlaCompiler::Argument>& args,
                                 CompilationResult* result) {
  VLOG(1) << "Executing graph symbolically to populate ComputationBuilder.";

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "XlaCompiler::CompileGraph: "
            << dump_graph::DumpGraphToFile(
                   strings::StrCat("xla_compile_graph_", name), *graph);
  }

  // Report the error here if initialization failed.
  TF_RETURN_IF_ERROR(initialization_status_);

  // Converts Tensorflow's graph control-flow constructs into functional
  // control-flow that can be compiled into XLA code.
  TF_RETURN_IF_ERROR(
      FunctionalizeControlFlow(graph.get(), local_flib_def_.get()));

  xla::ComputationBuilder builder(client(), name);
  XlaContext* context =
      new XlaContext(this, &builder, options_.allow_cpu_custom_calls,
                     options.resolve_compile_time_constants);
  core::ScopedUnref context_unref(context);

  result->tuple_arg = options.use_tuple_arg;

  std::vector<XlaContext::Argument> context_args;
  TF_RETURN_IF_ERROR(BuildArguments(args, options.use_tuple_arg, &builder,
                                    &context_args, &result->input_mapping,
                                    &result->xla_input_shapes));
  context->set_args(std::move(context_args));

  TF_RETURN_IF_ERROR(ExecuteGraph(context, std::move(graph), device_,
                                  flib_runtime_, NextStepId()));

  int num_nonconst_outputs;
  int num_computation_outputs;
  result->computation = std::make_shared<xla::Computation>();
  TF_RETURN_IF_ERROR(BuildComputation(
      context->retvals(), context->resources(), context->has_side_effects(),
      options.return_updated_values_for_all_resources, &builder,
      result->computation.get(), &num_computation_outputs,
      &num_nonconst_outputs, &result->resource_updates));

  result->requires_runtime_context = context->has_context_parameter();

  // Tuple arguments and runtime context parameters are incompatible.
  CHECK(!(options.use_tuple_arg && result->requires_runtime_context));

  VLOG(2) << "Outputs: total: " << context->retvals().size()
          << " nonconstant: " << num_nonconst_outputs;
  result->outputs.resize(context->retvals().size());
  for (std::vector<XlaContext::HandleOrConstant>::size_type i = 0;
       i < context->retvals().size(); ++i) {
    const XlaContext::HandleOrConstant& retval = context->retvals()[i];
    if (retval.is_constant) {
      OutputDescription& output = result->outputs[i];
      output.shape = retval.constant_value.shape();
      output.is_constant = true;
      output.constant_value = retval.constant_value;
    }
  }

  if (result->computation->IsNull()) {
    return Status::OK();
  }

  // Compute the output shapes, if there is a computation with non-constant
  // outputs.
  auto computation_shape = client()->GetComputationShape(*result->computation);
  if (!computation_shape.ok()) {
    return computation_shape.status();
  }

  result->xla_output_shape.Swap(
      computation_shape.ValueOrDie()->mutable_result());
  VLOG(2) << "XLA output shape: "
          << xla::ShapeUtil::HumanString(result->xla_output_shape);

  // Tensorflow expects a major-to-minor order of results.
  AssignMajorToMinorLayout(&result->xla_output_shape);

  // Converts the output shapes to TensorShapes.
  int computation_output = 0;
  for (std::vector<XlaContext::HandleOrConstant>::size_type i = 0;
       i < context->retvals().size(); ++i) {
    const XlaContext::HandleOrConstant& retval = context->retvals()[i];
    if (!retval.is_constant) {
      CHECK_LT(computation_output, num_computation_outputs);
      OutputDescription& output = result->outputs[i];
      output.is_constant = false;
      if (num_computation_outputs > 1) {
        TF_RETURN_IF_ERROR(XLAShapeToTensorShape(
            xla::ShapeUtil::GetTupleElementShape(result->xla_output_shape,
                                                 computation_output),
            &output.shape));
      } else {
        TF_RETURN_IF_ERROR(
            XLAShapeToTensorShape(result->xla_output_shape, &output.shape));
      }
      ++computation_output;
    }
  }

  for (std::vector<ResourceUpdate>::size_type i = 0;
       i < result->resource_updates.size(); ++i) {
    if (num_computation_outputs > 1) {
      result->resource_updates[i].shape = xla::ShapeUtil::GetTupleElementShape(
          result->xla_output_shape, computation_output);
    } else {
      CHECK_EQ(0, computation_output);
      result->resource_updates[i].shape = result->xla_output_shape;
    }
    ++computation_output;
  }
  return Status::OK();
}

Status XlaCompiler::GetChannelHandle(const string& key,
                                     xla::ChannelHandle* channel) {
  auto result = channels_.emplace(key, xla::ChannelHandle());
  if (result.second) {
    TF_ASSIGN_OR_RETURN(result.first->second, client()->CreateChannelHandle());
  }
  *channel = result.first->second;
  VLOG(1) << "Channel: " << key << " " << channel->DebugString();
  return Status::OK();
}

}  // namespace tensorflow
