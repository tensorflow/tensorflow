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

#ifndef TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILER_H_
#define TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILER_H_

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

// The XlaCompiler class is responsible for compilation of a self-contained
// subgraph of a TensorFlow computation using the XLA linear algebra runtime.
// It does a symbolic execution of the graph starting from specific input
// shapes, using a JIT device to convert operators into XLA computations.
//
// XlaCompiler is typically invoked from an `_XlaLaunch` operator once the
// shapes of all input parameters to the computation are known. This is
// because the symbolic execution requires known shapes for all operations.
//
// XlaCompiler compiles Tensorflow graphs that received inputs via _Arg nodes,
// and return outputs via _Retval nodes.
//
// The XlaCompiler requires one Argument struct for each _Arg index, that
// describes each argument. Arguments can be compile-time constants
// (kind kConstant), run-time parameters (kind kParameter), or resource
// variables (kinds kVariable and kUninitializedVariable).
//
// Only kParameter and kVariable arguments become runtime parameters to the
// generated XLA computation. The XLA computation will have run-time parameters
// in the following order:
//   +---------------------+-----------------------------------------+
//   |  kParameter values  |  Initial values of kVariable arguments  |
//   +---------------------+-----------------------------------------+
// Within each block, the arguments are arranged by the _Arg index from which
// they were derived.
// If `Options::requires_runtime_context` is true, then an additional runtime
// context argument is passed as a final argument.
//
// The run-time outputs of the XLA computation are arranged in the following
// order:
//   +------------------+-----------------------------------------+
//   |  _Retval values  |  Updated values of kVariable arguments  |
//   +------------------+-----------------------------------------+
// _Retval values are ordered by _Retval index, whereas kVariable values are
// ordered by the original _Arg position of the variable.
//
// In both inputs and outputs, kVariable values are placed the end. When
// emitting While loop bodies, we must ensure that the loop body has
// identical input and output signatures. By moving variable values
// to the end of the argument list and using the
// `return_updated_values_for_all_variables` option, we can ensure that the
// input and output values of variables appear at the same positions.

class XlaCompiler {
 public:
  // Describes how to derive the value of each _Arg node in the graph/function
  // being compiled. There must be one Argument for each _Arg index.
  struct Argument {
    enum Kind {
      // Default value; not a valid kind.
      kInvalid,

      // Argument is a compile-time constant. No associated runtime parameter.
      kConstant,

      // Argument is a variable that has not been initialized yet. No associated
      // runtime parameter.
      kUninitializedVariable,

      // Argument is a variable that already has a value set. Expects a runtime
      // parameter containing the current value.
      kVariable,

      // Argument is a run-time parameter.
      kParameter,
    };

    Kind kind = kInvalid;

    // The type of the argument. If the argument is a resource variable, this
    // is the type of the variable's value, not DT_RESOURCE.
    DataType type;

    // The shape of the argument. If the argument is a resource variable, this
    // is the shape of the variable's value.
    TensorShape shape;

    // The value of the argument, if it is a compile-time constant. Must be a
    // host-memory tensor.
    Tensor constant_value;

    // The name of this argument, used for debugging.
    string name;
  };

  struct OutputDescription {
    // Type and shape of the output.
    DataType type;
    TensorShape shape;

    // Constant output value, if known to be constant at JIT compilation time.
    // 'Tensor' is in host memory.
    bool is_constant = false;
    Tensor constant_value;
  };

  // Describes a variable write side effect of the computation.
  struct VariableUpdate {
    // Index of the input that contains the variable resource to write to.
    int input_index;

    // Type and shape of the tensor to be written back.
    DataType type;
    TensorShape shape;

    // Was the value of the variable modified by the computation?
    // (Always true, unless `return_updated_values_for_all_variables` is true.)
    bool modified;
  };

  struct CompilationResult {
    // Vector that maps from the parameters of the XLA computation to their
    // original argument positions. To handle compile-time constant inputs and
    // variables, the parameters to the XLA computation may be a subset of the
    // original arguments, and are not necessarily in the same order.)
    std::vector<int> input_mapping;

    // Does the computation require the local runtime context to be passed as
    // the last argument?
    bool requires_runtime_context = false;

    // Input shapes of the computation.
    std::vector<xla::Shape> xla_input_shapes;

    // Should the arguments be packed into a single tuple?
    bool tuple_arg;

    // Output shape in XLA format. The output shape is a tuple if and only if
    // the number of non-constant outputs is not equal to 1.
    xla::Shape xla_output_shape;

    // TensorFlow shapes of outputs, together with the values of any
    // constant arguments. Vector indexed by Tensorflow _Retval number,
    // containing both constant and non-constant results.
    std::vector<OutputDescription> outputs;

    // Variables whose values were updated by the computation, ordered
    // by return value position. Variable updates follow the non-constant
    // results in the outputs of XLA computation.
    std::vector<VariableUpdate> variable_updates;

    // The XLA computation built from the tensorflow subgraph. May be null
    // if the output consists solely of compile-time constants.
    xla::Computation computation;
  };

  struct Options {
    // Name of the compilation device to use.
    DeviceType device_type = DeviceType("");

    xla::Client* client = nullptr;

    // If 'allow_cpu_custom_calls' is true, kernels may make use of CustomCall()
    // for CPU; additionally, an optional XlaLocalRuntimeContext* may be passed
    // to the computation.
    bool allow_cpu_custom_calls = false;

    // If 'local_executable_has_hybrid_result', the top-level pointers of the
    // result tuple of compiled programs are stored in host memory and the
    // nested buffers in device memory, otherwise the whole result tuple is
    // stored in device memory.
    bool local_executable_has_hybrid_result = false;

    // If 'resolve_compile_time_constants' is true, then outputs of a
    // computation that are known to be compile-time constants will be returned
    // as Tensors at compile-time, rather than as run-time outputs of the
    // computation.
    bool resolve_compile_time_constants = true;

    // If `use_tuple_arg` is true, a single tuple parameter will be used for all
    // arguments; if false, each argument gets its own parameter.
    bool use_tuple_arg = false;

    // If 'return_updated_values_for_all_variables' is true, then updated
    // values of all resource variables arguments will be included in the
    // 'variable_updates' of the computation, even if the variable was not
    // modified by the computation. Used when compiling loop bodies to ensure
    // the input and output signatures match.
    bool return_updated_values_for_all_variables = false;

    // If 'prune_unreachable_nodes' is true, then nodes that are not
    // dependencies of graph's _Retval nodes will be pruned before compilation.
    // This is useful to prune stateful operators that should not be executed
    // from a function body.
    bool prune_unreachable_nodes = false;

    // If not nullptr, populate_resource_manager is called with the
    // compilation device's resource manager when the compilation
    // device is created, and can be used to create metadata objects
    // that can be accessed by XLA op kernels.
    std::function<Status(ResourceMgr*)>* populate_resource_manager = nullptr;
  };

  explicit XlaCompiler(Options options);
  ~XlaCompiler();

  // Compiles a Tensorflow function `fn_name_attrs` into an XLA computation.
  // `args` describes the arguments to the function, each of which must either
  // be a runtime-parameter to the XLA computation, a compile-time constant, or
  // a resource variable. Writes the compiled output to `result`.
  //
  // The generated XLA computation returns a tuple containing only the
  // non-constant outputs as a function of the input arguments. Constant
  // arguments are returned as host memory tensors in the output list and are
  // not included in the XLA computation's outputs. The XLA computation is
  // null if there are no data-dependent outputs and no side effects.
  Status CompileFunction(FunctionLibraryRuntime* flr,
                         const NameAttrList& fn_name_attrs,
                         const std::vector<Argument>& args,
                         CompilationResult* result);

  // Compiles a tensorflow::Graph into an xla::Computation.
  // Similar to CompileFunction, but takes a Graph as input rather than a
  // function.
  Status CompileGraph(string const& name, std::unique_ptr<Graph> graph,
                      FunctionLibraryRuntime* flr,
                      const std::vector<Argument>& args,
                      CompilationResult* result);

  // Takes `result` which has been compiled from a Tensorflow subgraph to a
  // XLA computation already, and generates an XLA LocalExecutable `executable`.
  Status BuildExecutable(const CompilationResult& result,
                         std::unique_ptr<xla::LocalExecutable>* executable);

  const Options& options() const { return options_; }
  xla::Client* client() const { return options_.client; }
  XlaCompilationDevice* device() const { return device_; }
  const DeviceMgr* device_mgr() const { return &device_mgr_; }

  // Retrieves the channel handle associated with `key`. Allocates
  // a new channel handle if none exists.
  // Channel handles can be used to communicate between different computations.
  // Computations that communicate should be compiled with the same XlaCompiler.
  Status GetChannelHandle(const string& key, xla::ChannelHandle* channel);

 private:
  Options options_;

  // Status set to non-OK in the constructor if initialization fails.
  Status initialization_status_;

  // Returns the next step sequence number.
  int64 NextStepId();

  mutex mu_;

  // Internal sequence number for steps executed on the compilation device.
  int64 next_step_id_ GUARDED_BY(mu_);

  XlaCompilationDevice* device_;  // Owned by device_mgr_
  DeviceMgr device_mgr_;

  std::unordered_map<string, xla::ChannelHandle> channels_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(XlaCompiler);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILER_H_
