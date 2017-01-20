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
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {

bool HasRetval(const Graph& graph) {
  for (const Node* n : graph.nodes()) {
    if (n->type_string() == "_Retval") return true;
  }
  return false;
}

Status CheckSignature(const DataTypeVector& tf_types,
                      const xla::Shape& xla_shape) {
  if (xla::ShapeUtil::IsTuple(xla_shape)) {
    if (xla::ShapeUtil::TupleElementCount(xla_shape) != tf_types.size()) {
      return errors::Internal("XLA shape has ",
                              xla::ShapeUtil::TupleElementCount(xla_shape),
                              " elements while function has ", tf_types.size());
    }
    for (int i = 0; i < tf_types.size(); ++i) {
      xla::PrimitiveType type;
      TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tf_types[i], &type));
      if (type !=
          xla::ShapeUtil::GetTupleElementShape(xla_shape, i).element_type()) {
        return errors::Internal(
            "element ", i, " has XLA type ",
            xla::ShapeUtil::GetTupleElementShape(xla_shape, i).element_type(),
            " and TensorFlow type ", DataTypeString(tf_types[i]));
      }
    }
  } else {
    if (tf_types.size() != 1) {
      return errors::Internal("Expected singleton type, got ", tf_types.size(),
                              " types");
    }
    xla::PrimitiveType type;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tf_types[0], &type));
    if (type != xla_shape.element_type()) {
      return errors::Internal("singleton element has XLA type ",
                              xla_shape.element_type(), " and TensorFlow type ",
                              DataTypeString(tf_types[0]));
    }
  }
  return Status::OK();
}

}  // namespace

XlaCompiler::XlaCompiler(const XlaCompiler::Options& options)
    : client_(options.client),
      allow_cpu_custom_calls_(options.allow_cpu_custom_calls),
      local_executable_has_hybrid_result_(
          options.local_executable_has_hybrid_result),
      next_step_id_(1),
      device_(new XlaCompilationDevice(SessionOptions(), options.device_type)),
      device_mgr_({device_}) {}

XlaCompiler::~XlaCompiler() = default;

int64 XlaCompiler::NextStepId() {
  mutex_lock l(mu_);
  return next_step_id_++;
}

Status XlaCompiler::CompileFunction(
    FunctionLibraryRuntime* flr, const NameAttrList& function,
    const std::vector<XlaCompiler::Argument>& args,
    XlaCompiler::CompilationResult* result) {
  const string function_id = Canonicalize(function.name(), function.attr());
  VLOG(1) << "XlaCompiler::CompileFunction " << function_id;

  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(
      flr->Instantiate(function.name(), function.attr(), &handle));

  const FunctionBody* fbody = flr->GetFunctionBody(handle);
  CHECK(fbody);

  return CompileFunctionBody(flr, *fbody, function_id, args,
                             /*use_tuple_arg=*/false, result);
}

Status XlaCompiler::CompileSubComputation(FunctionLibraryRuntime* flr,
                                          const NameAttrList& function,
                                          const xla::Shape& input_shape,
                                          const xla::Shape& output_shape,
                                          xla::Computation* computation) {
  const string function_id = Canonicalize(function.name(), function.attr());
  VLOG(1) << "XlaCompiler::CompileSubComputation " << function_id;

  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(
      flr->Instantiate(function.name(), function.attr(), &handle));

  const FunctionBody* fbody = flr->GetFunctionBody(handle);
  CHECK(fbody);

  TF_RETURN_IF_ERROR(CheckSignature(fbody->arg_types, input_shape));
  TF_RETURN_IF_ERROR(CheckSignature(fbody->ret_types, output_shape));

  const bool use_tuple_arg = xla::ShapeUtil::IsTuple(input_shape);

  std::vector<XlaCompiler::Argument> args(fbody->arg_types.size());
  if (use_tuple_arg) {
    for (int i = 0; i < args.size(); ++i) {
      xla::Shape xla_shape =
          xla::ShapeUtil::GetTupleElementShape(input_shape, i);
      args[i].type = fbody->arg_types[i];
      args[i].shape = XLAShapeToTensorShape(xla_shape);
      args[i].parameter = i;
    }
  } else {
    args[0].type = fbody->arg_types[0];
    args[0].shape = XLAShapeToTensorShape(input_shape);
    args[0].parameter = 0;
  }

  CompilationResult result;
  TF_RETURN_IF_ERROR(CompileFunctionBody(flr, *fbody, function_id, args,
                                         use_tuple_arg, &result));

  if (!xla::ShapeUtil::Compatible(result.xla_output_shape, output_shape)) {
    return errors::Internal("output shape mismatch from compilation");
  }
  *computation = std::move(result.computation);

  return Status::OK();
}

Status XlaCompiler::CompileFunctionBody(
    FunctionLibraryRuntime* flr, const FunctionBody& fbody,
    const string& function_id, const std::vector<XlaCompiler::Argument>& args,
    bool use_tuple_arg, XlaCompiler::CompilationResult* result) {
  VLOG(1) << "XlaCompiler::CompileFunctionBody " << function_id;

  std::unique_ptr<Graph> graph(new Graph(flr->GetFunctionLibraryDefinition()));
  CopyGraph(*fbody.graph, graph.get());

  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile(
        strings::StrCat("xla_jit_raw_input_", function_id), *graph);
  }

  if (!HasRetval(*graph)) {
    VLOG(1) << "Graph has no retvals. Skipping compilation.";
    return Status::OK();
  }

  // Optimize the graph to before running throught the translator.
  // TODO(pbar) The constant folder currently does not simplify int32 operations
  // for devices other than CPU.
  OptimizerOptions opts;
  GraphOptimizer optimizer(opts);
  OptimizeGraph(flr, &graph);

  if (VLOG_IS_ON(1)) {
    dump_graph::DumpGraphToFile(
        strings::StrCat("xla_jit_final_graph_", function_id), *graph);
  }

  VLOG(1) << "====================================================";
  TF_RETURN_IF_ERROR(CompileGraph(function_id, std::move(graph), flr, args,
                                  use_tuple_arg, result));
  VLOG(1) << "====================================================";

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
    argument_layouts[i] = &result.xla_input_shapes[i].second;
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
  build_options.set_has_hybrid_result(local_executable_has_hybrid_result_);

  auto compile_result = local_client->Compile(result.computation,
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

  auto runner = [](Executor::Args::Closure c) {
    // TODO(misard) Temporarily just schedule c eagerly while we
    // decide what to do about the fact that the ComputationBuilder is
    // thread-compatible, but we don't really want Op writers to have
    // to remember to acquire a lock around every call to
    // ComputationBuilder. One possibility is to add the (generally
    // useful) ability to run a single-threaded Executor based on an
    // option in LocalExecutorParams. Another is to automagically
    // acquire a lock around ComputationBuilder calls using some
    // wrapper or RAII funny business.
    c();
  };

  // Run the graph symbolically, turning the graph into an XLA computation.
  Executor::Args exec_args;
  exec_args.step_id = step_id;
  exec_args.step_container = step_container.get();
  exec_args.runner = runner;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      exec->Run(exec_args),
      "Conversion from TensorFlow graph to XLA computation failed.");

  // Explicitly clean up the step container, to capture the cleanup status.
  step_container.reset();
  return cleanup_status;
}

}  // namespace

Status XlaCompiler::CompileGraph(string const& name,
                                 std::unique_ptr<Graph> graph,
                                 FunctionLibraryRuntime* flib,
                                 const std::vector<XlaCompiler::Argument>& args,
                                 bool use_tuple_arg,
                                 CompilationResult* result) {
  VLOG(1) << "Executing graph symbolically to populate ComputationBuilder.";

  // Converts the input shapes into xla::Shape instances.
  result->xla_input_shapes.reserve(args.size());
  for (int i = 0; i < args.size(); ++i) {
    if (args[i].parameter < 0) {
      continue;
    }
    result->xla_input_shapes.push_back(std::make_pair(i, xla::Shape()));
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(
        args[i].type, args[i].shape, &result->xla_input_shapes.back().second));
  }

  XlaContext* xla_context =
      new XlaContext(this, client(), name, allow_cpu_custom_calls_);
  core::ScopedUnref xla_context_unref(xla_context);

  TF_RETURN_IF_ERROR(xla_context->BuildArguments(args, use_tuple_arg));

  TF_RETURN_IF_ERROR(
      ExecuteGraph(xla_context, std::move(graph), device_, flib, NextStepId()));

  std::vector<XlaContext::ConstRetVal> compile_time_constants;
  int num_nonconst_outputs;
  TF_RETURN_IF_ERROR(xla_context->CollectResults(
      &result->computation, &result->requires_runtime_context,
      &compile_time_constants, &num_nonconst_outputs));

  VLOG(2) << "Outputs: constant: " << compile_time_constants.size()
          << " nonconstant: " << num_nonconst_outputs;
  result->outputs.resize(compile_time_constants.size() + num_nonconst_outputs);
  for (const auto& c : compile_time_constants) {
    if (!c.status.ok()) {
      Status constant_status = c.status;
      errors::AppendToMessage(&constant_status,
                              "Failed evaluating constant XLA return "
                              "value ",
                              c.index);
      return constant_status;
    }
    if (c.index >= result->outputs.size()) {
      return errors::InvalidArgument("Invalid argument index ", c.index);
    }
    OutputDescription& output = result->outputs[c.index];
    output.shape = c.value.shape();
    output.is_constant = true;
    output.constant_value = c.value;
  }

  if (result->computation.IsNull()) {
    return Status::OK();
  }

  // Compute the output shapes, if there is a computation with non-constant
  // outputs.
  auto computation_shape = client()->GetComputationShape(result->computation);
  if (!computation_shape.ok()) {
    return computation_shape.status();
  }

  result->xla_output_shape.Swap(
      computation_shape.ValueOrDie()->mutable_result());

  auto num_non_constant_outputs =
      (xla::ShapeUtil::IsTuple(result->xla_output_shape))
          ? xla::ShapeUtil::TupleElementCount(result->xla_output_shape)
          : 1;
  // Tensorflow expects a major-to-minor order of results.
  if (1 == num_non_constant_outputs) {
    xla::Shape& s = result->xla_output_shape;
    auto& minor_to_major = *s.mutable_layout()->mutable_minor_to_major();
    minor_to_major.Resize(xla::ShapeUtil::Rank(s), 0);
    std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
  } else {
    for (xla::Shape& s : *result->xla_output_shape.mutable_tuple_shapes()) {
      auto& minor_to_major = *s.mutable_layout()->mutable_minor_to_major();
      minor_to_major.Resize(xla::ShapeUtil::Rank(s), 0);
      std::iota(minor_to_major.rbegin(), minor_to_major.rend(), 0);
    }
  }

  // Converts the output shapes to TensorShapes.
  int computation_output = 0;
  for (int i = 0; i < result->outputs.size(); ++i) {
    if (!result->outputs[i].is_constant) {
      CHECK_LT(computation_output, num_non_constant_outputs);
      if (num_non_constant_outputs > 1) {
        result->outputs[i].shape =
            XLAShapeToTensorShape(xla::ShapeUtil::GetTupleElementShape(
                result->xla_output_shape, computation_output));
      } else {
        result->outputs[i].shape =
            XLAShapeToTensorShape(result->xla_output_shape);
      }
      ++computation_output;
    }
  }
  return Status::OK();
}

Status XlaCompiler::GetChannelHandle(const string& key,
                                     xla::ChannelHandle* channel) {
  mutex_lock lock(mu_);
  auto result = channels_.emplace(key, xla::ChannelHandle());
  if (result.second) {
    TF_ASSIGN_OR_RETURN(result.first->second, client_->CreateChannelHandle());
  }
  *channel = result.first->second;
  return Status::OK();
}

}  // namespace tensorflow
