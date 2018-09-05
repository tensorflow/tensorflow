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

#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class XlaContext;

// The XlaCompiler class is responsible for compilation of a self-contained
// subgraph of a TensorFlow computation using the XLA linear algebra runtime.
// It does a symbolic execution of the graph starting from specific input
// shapes, using a JIT device to convert operators into XLA computations.
//
// XlaCompiler is typically invoked from an `XlaLaunch` operator once the
// shapes of all input parameters to the computation are known. This is
// because the symbolic execution requires known shapes for all operations.
//
// XlaCompiler compiles Tensorflow graphs that received inputs via _Arg nodes,
// and return outputs via _Retval nodes.
//
// The XlaCompiler requires one Argument struct for each _Arg index, that
// describes each argument. Arguments can be compile-time constants
// (kind kConstant), run-time parameters (kind kParameter), or resources
// (kind kResource).
//
// Only kParameter and initialized kResource arguments become runtime parameters
// to the generated XLA computation.
//
// The run-time outputs of the XLA computation are arranged in the following
// order:
//   +------------------+-----------------------------------------+
//   |  _Retval values  |  Updated values of kResource arguments  |
//   +------------------+-----------------------------------------+
// _Retval values are ordered by _Retval index, whereas kResource values are
// ordered by the original _Arg position of the variable.
//
// If a shape representation function is provided as part of
// XlaCompiler::CompileOptions, kParameter arguments and return values to an
// entry computation will be reshaped in accordance to the shape function.
// Arguments and return values to a non-entry computation are not reshaped.
// Variable resource arguments are passed and returned in reshaped form, even
// for non-entry computations. This feature allows TensorFlow to keep on-device
// tensors with a different shape to their representation inside the XLA
// computation.
//
// In computation outputs, updated kResource values are placed the end. When
// emitting While loop bodies, we must ensure that the loop body has
// identical input and output signatures. By passing variable values
// at the end of the argument list and using the
// `return_updated_values_for_all_variables` option, we can ensure that the
// input and output values of resources appear at the same positions.
//
// Resources are passed as parameters or returned as resource updates in
// "packed" form.
// kStack resources are packed as (array, size of stack) XLA tuples.
// kTensorArray resources without gradients are packed as the array that
// backs the TensorArray. If gradients are present (`tensor_array_gradients`),
// the packed representation is a (array, gradient0, gradient1, ...) tuple,
// where gradient_k is the value of the k-th gradient in the
// `tensor_array_gradients` ordered set.
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

      // Argument is a Variable, TensorArray, or Stack resource. Has an
      // associated runtime parameter iff `initialized` is true.
      kResource,

      // Argument is a run-time parameter.
      kParameter,
    };

    Kind kind = kInvalid;

    // The type of the argument. If the argument is a resource, this
    // is the type of the variable's value, not DT_RESOURCE.
    DataType type;

    // The shape of the argument. For:
    // * a parameter: the shape of the parameter.
    // * a constant: ignored; the shape given by constant_value is used
    //     instead.
    // * an uninitialized resource: ignored. We don't yet know the shape of an
    //     uninitialized resource (otherwise we would have initialized it!)
    // * an initialized variable: the shape of the variable's value.
    // * an initialized TensorArray or Stack resource: the shape of an entry in
    //   the TensorArray/Stack. Note this is the size of a single entry, not the
    //   XLA data structure that represents the complete stack/array.
    TensorShape shape;

    // The value of the argument, if it is a compile-time constant. Must be a
    // host-memory tensor.
    Tensor constant_value;

    // The name of this argument, used for debugging.
    string name;

    // For a kResource, what kind of resource is it?
    XlaResource::Kind resource_kind = XlaResource::kInvalid;

    // For a kResource, has this resource been initialized?
    bool initialized = false;

    // For a TensorArray or Stack resource, what is the array's declared size?
    // (Used for lazy initialization.)
    int64 tensor_array_size = -1;

    // TensorArray resource parameters are passed as (array, gradient array 0,
    // ..., gradient array k), where the gradient arrays are in the same order
    // as `tensor_array_gradients`.
    std::set<string> tensor_array_gradients;

    bool operator==(const Argument& other) const;
  };

  // Options pertaining to an individual call to CompileGraph() or
  // CompileFunction().
  struct CompileOptions {
    // If `use_tuple_arg` is true, a single tuple parameter will be used for all
    // arguments; if false, each argument gets its own parameter.
    bool use_tuple_arg = false;

    // If 'return_updated_values_for_all_resources' is true, then updated
    // values of all resource arguments will be included in the
    // 'resource_updates' of the computation, even if the resource was not
    // modified by the computation. Used when compiling loop bodies to ensure
    // the input and output signatures match.
    bool return_updated_values_for_all_resources = false;

    // If 'resolve_compile_time_constants' is true, then outputs of a
    // computation that are known to be compile-time constants will be returned
    // as Tensors at compile-time, rather than as run-time outputs of the
    // computation.
    bool resolve_compile_time_constants = true;

    // If 'always_return_tuple' is true, then the output of a computation will
    // always be a tuple. Otherwise, a single-element output will not be wrapped
    // in a tuple.
    bool always_return_tuple = true;

    // True when compiling the entry computation, false for subcomputations
    // (while, call, etc.)
    bool is_entry_computation = true;
  };

  struct OutputDescription {
    // Type and shape of the output. The shape is the unflattened shape.
    // When `type` is DT_RESOURCE, `shape` is the shape of the resource
    // variable's value.
    DataType type;
    TensorShape shape;

    // Constant output value, if known to be constant at JIT compilation time.
    // 'Tensor' is in host memory.
    bool is_constant = false;
    Tensor constant_value;

    // When this output is a resource, i.e. `type == DT_RESOURCE`, this is
    // the index of the input that contains the resource.
    int input_index;
  };

  // Describes a variable write side effect of the computation.
  struct ResourceUpdate {
    // Index of the input that contains the variable resource to write to.
    int input_index;

    // Type and shape of the tensor to be written back.
    // The `shape` field has the same meaning as the Argument::shape field.
    DataType type;
    TensorShape shape;

    // Was the value of the variable modified by the computation?
    // (Always true, unless `return_updated_values_for_all_resources` is true.)
    bool modified;

    // If the resource is a TensorArray, the set of gradients read or written.
    std::set<string> tensor_array_gradients_accessed;
  };

  struct CompilationResult {
    // Vector that maps from the parameters of the XLA computation to their
    // original argument positions. To handle compile-time constant inputs, the
    // parameters to the XLA computation may be a subset of the original
    // arguments. The relative ordering of parameters are maintained.
    std::vector<int> input_mapping;

    // Input shapes of the computation. If we are flattening inputs, these are
    // the flattened shapes.
    std::vector<xla::Shape> xla_input_shapes;

    // Output shape in XLA format. The output shape is always a tuple. If we
    // are flattening outputs, these are the flattened shapes.
    xla::Shape xla_output_shape;

    // TensorFlow shapes of outputs, together with the values of any
    // constant arguments. Vector indexed by Tensorflow _Retval number,
    // containing both constant and non-constant results.
    std::vector<OutputDescription> outputs;

    // TensorFlow shapes and types of sends/recvs from HostCompute Ops to their
    // matching RecvAtHost/SendFromHost Ops in the outer graph.
    tf2xla::HostComputeMetadata host_compute_metadata;

    // Resources whose values were updated by the computation, ordered
    // by return value position (which is the same as the order the resources
    // were passed as arguments). Resource updates follow the non-constant
    // results in the outputs of XLA computation.
    std::vector<ResourceUpdate> resource_updates;

    // The XLA computation built from the tensorflow subgraph.
    std::shared_ptr<xla::XlaComputation> computation;
  };

  typedef std::function<xla::StatusOr<TensorShape>(const TensorShape&,
                                                   DataType)>
      ShapeRepresentationFn;
  struct Options {
    // Name of the compilation device to use. It must be set by the caller.
    // The default empty value is invalid.
    DeviceType device_type = DeviceType("");

    // The device to use during compilation to execute instructions on, for
    // example for auto-tuning.
    // Valid values are defined by `xla::Backend::devices_ordinal_supported()`.
    // -1 indicates the default device should be used.
    int device_ordinal = -1;

    xla::Client* client = nullptr;

    // Function library in which to find function definitions. Must be non-null.
    const FunctionLibraryDefinition* flib_def = nullptr;

    // The graph def version to be compiled.
    int graph_def_version = TF_GRAPH_DEF_VERSION;

    // If 'allow_cpu_custom_calls' is true, kernels may make use of CustomCall()
    // for CPU.
    bool allow_cpu_custom_calls = false;

    // If set, the XLA representation of variables represented to XLA as the
    // shape given by this shape function. Variables are reshaped to this shape
    // on write, and reshaped to their original shape on read.
    ShapeRepresentationFn shape_representation_fn;

    // If not nullptr, populate_resource_manager is called with the
    // compilation device's resource manager when the compilation
    // device is created, and can be used to create metadata objects
    // that can be accessed by XLA op kernels.
    std::function<Status(ResourceMgr*)>* populate_resource_manager = nullptr;

    // If not nullptr, this memory allocator can be used by the compiler for
    // temporary allocations it might want to make during compilation.
    //
    // For example, the compiler may want to try out different algorithms and
    // choose the fastest one, and it might run those algorithms over buffers
    // created using this allocator.
    //
    // The compiler can function correctly without an explicit allocator given
    // here, but on some devices (notably, GPUs), TensorFlow tends to eagerly
    // allocate most or all available memory on the device, leaving none for the
    // compiler to access, unless it can use TensorFlow's allocator.
    xla::DeviceMemoryAllocator* device_allocator = nullptr;
  };

  explicit XlaCompiler(Options options);

  ~XlaCompiler();

  Status CompileFunction(const CompileOptions& options,
                         const NameAttrList& fn_name_attrs,
                         std::vector<Argument> args, CompilationResult* result);

  // Compiles a tensorflow::Graph into an xla::XlaComputation.
  // Similar to CompileFunction, but takes a Graph as input rather than a
  // function.
  Status CompileGraph(const CompileOptions& options, string const& name,
                      std::unique_ptr<Graph> graph,
                      const std::vector<Argument>& args,
                      CompilationResult* result);

  // Compiles a single Op, given by an OpKernelContext, into an
  // xla::XlaComputation. Similar to CompileFunction but takes a single Op as
  // input.
  Status CompileSingleOp(const CompileOptions& options, string const& name,
                         OpKernelContext* ctx,
                         const std::vector<Argument>& args,
                         CompilationResult* result);

  // Returns the shape of the XLA parameter for an argument 'arg'.
  // See the class comment for more details about the argument passing
  // convention.
  Status XLAShapeForArgument(const Argument& arg, bool is_entry_computation,
                             xla::Shape* xla_shape) const;

  // Retrieves the channel handle associated with `key`. Allocates
  // a new channel handle if none exists.
  // Channel handles can be used to communicate between different
  // computations. Computations that communicate should be compiled with the
  // same XlaCompiler.
  Status GetChannelHandle(const string& key, xla::ChannelHandle* channel);

  // Retrieves the host-to-device channel handle associated with `key`.
  // Allocates a new channel handle if none exists.
  Status GetHostToDeviceChannelHandle(const string& key,
                                      xla::ChannelHandle* channel);

  // Retrieves the device-to-host channel handle associated with `key`.
  // Allocates a new channel handle if none exists.
  Status GetDeviceToHostChannelHandle(const string& key,
                                      xla::ChannelHandle* channel);

  // Sets the shapes and types for the device to host transfer associated with
  // 'key'.
  Status SetDeviceToHostMetadata(const string& key,
                                 absl::Span<const DataType> types,
                                 absl::Span<const TensorShape> shapes);

  // Gets the shapes the device to host transfer associated with 'key'.
  Status GetDeviceToHostShapes(const string& key,
                               std::vector<TensorShape>* shapes) const;

  // Sets the shapes and types for the host to device transfer associated with
  // 'key'.
  Status SetHostToDeviceMetadata(const string& key,
                                 absl::Span<const DataType> types,
                                 absl::Span<const TensorShape> shapes);

  // In order to avoid deadlocks from dependencies in host computations, it can
  // be necessary to enforce a partial order on the execution of HostCompute
  // Ops. In particular it may be necessary to constrain the SendToHost for one
  // HostCompute to run before blocking on the RecvAtHost for another
  // HostCompute. The compiler maintains a mapping from 'host_compute_name' to
  // handle, where the handle is an 'output' of the HostCompute Op corresponding
  // to 'host_compute_name'. Another HostCompute Op that needs to be sequenced
  // later can add the handle as an 'input' to enforce the constraints.
  // 'host_compute_name' can be any string the client wishes to use to identify
  // a given HostCompute Op as long as the names are unique within the
  // compilation.
  Status GetHostComputeControlDependency(const string& host_compute_name,
                                         xla::XlaOp* handle);
  Status SetHostComputeControlDependency(const string& host_compute_name,
                                         const xla::XlaOp& handle);

  const Options& options() const { return options_; }
  xla::Client* client() const { return options_.client; }
  FunctionLibraryRuntime* flib_runtime() const { return flib_runtime_; }

 private:
  // Sets the function body `fbody` to the one registered as `function`.
  Status FindFunctionBody(const NameAttrList& function,
                          const FunctionBody** fbody);

  // Returns the optimized graph object in this function body.
  std::unique_ptr<Graph> GetGraph(const FunctionBody* fbody);

  // Builds XLA computations for each of the arguments to the computation.
  // `args` are the arguments to the computation.
  Status BuildArguments(const Graph& graph,
                        const std::vector<XlaCompiler::Argument>& args,
                        bool use_tuple_arg, xla::XlaBuilder* builder,
                        XlaContext* context, std::vector<int>* arg_cores,
                        std::vector<XlaExpression>* arg_expressions,
                        std::vector<int>* input_mapping,
                        std::vector<xla::Shape>* input_shapes,
                        bool is_entry_computation);

  // Graph compiler needs to know how to get an optimized graph from a function
  // body.
  friend class GraphCompiler;
  friend class XlaCompilerTest;

  Options options_;

  // Status set to non-OK in the constructor if initialization fails.
  Status initialization_status_;

  // Returns the next step sequence number.
  int64 NextStepId();

  // Internal sequence number for steps executed on the compilation device.
  int64 next_step_id_;

  XlaCompilationDevice* device_;  // Owned by device_mgr_
  DeviceMgr device_mgr_;

  // To avoid copying the client's function library, use a local function
  // library and runtime for functions created as part of the functionalize
  // control flow transformation.
  std::unique_ptr<FunctionLibraryDefinition> local_flib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> local_pflr_;

  FunctionLibraryRuntime* local_flib_runtime_;  // owned by local_pflr_.
  FunctionLibraryRuntime* flib_runtime_;        // owned by pflr_.

  struct SignatureHash {
    uint64 operator()(
        const std::pair<string, std::vector<Argument>>& signature) const;
  };

  std::unordered_map<std::pair<string, std::vector<Argument>>,
                     CompilationResult, SignatureHash>
      cache_;

  std::unordered_map<string, xla::ChannelHandle> channels_;

  std::unordered_map<string, tf2xla::HostTransferMetadata> host_compute_sends_;
  std::unordered_map<string, tf2xla::HostTransferMetadata> host_compute_recvs_;

  std::unordered_map<string, xla::XlaOp> host_compute_control_output_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaCompiler);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILER_H_
