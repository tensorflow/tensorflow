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
// It is typically invoked from an `_XlaLaunch` operator once the shapes
// of all input parameters to the computation are known. This is
// because the symbolic execution requires known shapes for all operations.
class XlaCompiler {
 public:
  // Describes how to derive the value of each _Arg node in the graph/function
  // being compiled. Each argument must be either a parameter of the generated
  // XLA computation (parameter >= 0), or a compile time constant
  // (parameter < 0).
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
    // Shape of the output.
    TensorShape shape;

    // Constant output value, if known to be constant at JIT compilation time.
    // 'Tensor' is in host memory.
    bool is_constant = false;
    Tensor constant_value;
  };

  // Describes a variable write side effect of the computation.
  struct VariableWrite {
    // Index of the input that contains the variable resource to write to.
    int input_index;

    // Type and shape of the tensor to be written back.
    DataType type;
    TensorShape shape;
  };

  struct CompilationResult {
    // Vector of (Tensorflow input number, XLA shape) pairs that describe
    // the arguments of the compiled XLA computation. (Because of constant
    // inputs, the arguments to the XLA computation are a subset of the
    // inputs passed to the JIT.)
    std::vector<std::pair<int, xla::Shape>> xla_input_shapes;

    // Does the computation require the local runtime context to be passed as
    // the last argument?
    bool requires_runtime_context = false;

    // Output shape in XLA format. This is a tuple if and only if
    // there are multiple non-constant outputs.
    xla::Shape xla_output_shape;

    // TensorFlow shapes of outputs, together with the values of any
    // constant arguments. Vector indexed by Tensorflow _Retval number,
    // containing both constant and non-constant results.
    std::vector<OutputDescription> outputs;

    // Variables whose values should be written by the computation back, ordered
    // by return value position. Variable write results follow the non-constant
    // results in the outputs of XLA computation.
    std::vector<VariableWrite> variable_writes;

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
  };

  explicit XlaCompiler(const Options& options);
  ~XlaCompiler();

  // Compiles a Tensorflow function `fn_name_attrs` into an XLA computation.
  // `args` describes the arguments to the function, each of which must either
  // be a parameter to the XLA computation or a compile-time constant.
  // Writes the compiled output to `result`.
  //
  // The generated XLA computation returns a tuple containing only the
  // non-constant outputs as a function of the input arguments. Constant
  // arguments are returned as host memory tensors in the output list and are
  // not included in the XLA computation's outputs. The XLA computation is
  // null if there are no data-dependent outputs.
  Status CompileFunction(FunctionLibraryRuntime* flr,
                         const NameAttrList& fn_name_attrs,
                         const std::vector<Argument>& args,
                         CompilationResult* result);

  // Compiles a tensorflow::Graph into an xla::Computation.
  // Similar to CompileFunction, but takes a Graph as input rather than a
  // function.
  // If `use_tuple_arg` is true, the compilation takes all of its arguments as
  // a single tuple.
  Status CompileGraph(string const& name, std::unique_ptr<Graph> graph,
                      FunctionLibraryRuntime* flr,
                      const std::vector<Argument>& args, bool use_tuple_arg,
                      CompilationResult* result);

  // Helper function that compiles a function to an XLA computation suitable
  // for use as a subroutine in other Computations, e.g., the body of a
  // While loop.
  //
  // The emitted Computation takes a single input parameter with
  // input_shape. If this is a tuple then the tuple element shapes
  // must match the types of the function's _Arg nodes. If input_shape
  // is not a tuple then the function must have a single _Arg node
  // with the same type as input_shape. The shapes of the _Arg values
  // will be compiled to match input_shape.
  //
  // The emitted Computation also returns a single value. If output_shape is a
  // tuple the tuple elements' types and shapes must match the compiled
  // function's _Retval nodes. If output_shape is not a tuple the
  // function must have a single _Retval node with the correct type
  // (and shape after compilation).
  Status CompileSubComputation(FunctionLibraryRuntime* flr,
                               const NameAttrList& fn_name_attrs,
                               const xla::Shape& input_shape,
                               const xla::Shape& output_shape,
                               xla::Computation* computation);

  // Takes <*result>, which has been compiled from a Tensorflow subgraph to a
  // XLA computation already, and generates an XLA LocalExecutable `executable`.
  Status BuildExecutable(const CompilationResult& result,
                         std::unique_ptr<xla::LocalExecutable>* executable);

  xla::Client* client() const { return client_; }
  XlaCompilationDevice* device() const { return device_; }
  const DeviceMgr* device_mgr() const { return &device_mgr_; }

  // Retrieves the channel handle associated with `key`. Allocates
  // a new channel handle if none exists.
  // Channel handles can be used to communicate between different computations.
  // Computations that communicate should be compiled with the same XlaCompiler.
  Status GetChannelHandle(const string& key, xla::ChannelHandle* channel);

 private:
  // Does the real work of Compile() and CompileToComputation().
  Status CompileFunctionBody(FunctionLibraryRuntime* flr,
                             const FunctionBody& function_body,
                             const string& name,
                             const std::vector<Argument>& args,
                             bool use_tuple_arg, CompilationResult* result);

  xla::Client* client_;  // Not owned.
  const bool allow_cpu_custom_calls_;
  const bool local_executable_has_hybrid_result_;
  const bool resolve_compile_time_constants_;

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
