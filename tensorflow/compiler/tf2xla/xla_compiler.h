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

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stack>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/host_compute_metadata.pb.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/xla_argument.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "xla/client/client.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
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
  // TODO(b/255826209): Remove this alias. Depending on XlaCompiler just to use
  // XlaArgument seeems weird and can cause circular dependencies.
  using Argument = ::tensorflow::XlaArgument;

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

    // If 'always_return_tuple' is true, then the output of a computation will
    // always be a tuple. Otherwise, a single-element output will not be wrapped
    // in a tuple.
    bool always_return_tuple = true;

    // True when compiling the entry computation, false for subcomputations
    // (while, call, etc.)
    bool is_entry_computation = true;

    // True when we should add XLA input & output to the graph/function.
    bool add_token_input_output = false;

    // Resource updates are converted into input / output of xla. The two
    // buffers are aliased with other if this option is true.
    bool alias_resource_update = false;
  };

  using OutputDescription = ::tensorflow::XlaOutputDescription;

  using ResourceUpdate = ::tensorflow::XlaResourceUpdate;

  using CompilationResult = ::tensorflow::XlaCompilationResult;

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

    // A ShapeDeterminationFns (i.e., a bundle of LayoutSelectionFn and
    // ShapeRepresentationFn). Each bundle describes the XLA representation of
    // arguments represented to XLA as the shape given by this shape function.
    // Arguments are input activations or weights to an XLA entry computation.
    // Variables are reshaped to this shape on write, and reshaped to their
    // original shape on read.
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns;

    // If not nullptr, populate_resource_manager is called with the
    // compilation device's resource manager when the compilation
    // device is created, and can be used to create metadata objects
    // that can be accessed by XLA op kernels.
    std::function<absl::Status(ResourceMgr*)>* populate_resource_manager =
        nullptr;

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
    // This must be a shared_ptr, as this is passed all the way down to the
    // cluster compilation. This allows asynchronous compilation to hold a
    // reference until the compilation is finished.
    std::shared_ptr<se::DeviceMemoryAllocator> device_allocator;

    // Alias input and output buffers for parameters that are passed-through XLA
    // modules without being changed.
    bool alias_passthrough_params = false;

    // Enable detailed logging of compilation metadata.
    bool detailed_logging = true;
  };

  // Argument for compiling a single op.
  struct SingleOpCompileArgument {
    // Data type of the output tensors. This is used to create _Retval node.
    std::vector<DataType> output_dtypes;

    // The NodeDef representing the op.
    NodeDef node_def;

    // This is currently only used to obtain MLIR TPU bridge rollout state.
    // Can be removed once full rollout is complete.
    ConfigProto config_proto;

    SingleOpCompileArgument() = default;

    explicit SingleOpCompileArgument(const OpKernelContext& ctx);
  };

  explicit XlaCompiler(Options options);

  ~XlaCompiler();

  // Helper function to populate an XlaCompiler::Argument from XlaResource.
  static void PopulateArgumentFromResource(const XlaResource& resource,
                                           Argument* arg);

  absl::Status CompileFunction(const CompileOptions& options,
                               const NameAttrList& fn_name_attrs,
                               absl::Span<const Argument> args,
                               CompilationResult* result);

  absl::Status CompileSingleOp(
      const CompileOptions& options,
      const SingleOpCompileArgument& single_op_compile_argument,
      absl::Span<const Argument> args, CompilationResult* result);

  // Compiles a tensorflow::Graph into an xla::XlaComputation.
  // Similar to CompileFunction, but takes a Graph as input rather than a
  // function.
  absl::Status CompileGraph(const CompileOptions& options, string const& name,
                            std::unique_ptr<Graph> graph,
                            absl::Span<const Argument> args,
                            CompilationResult* result);

  // Returns the shape of the XLA parameter for an argument 'arg'.
  // See the class comment for more details about the argument passing
  // convention.
  absl::Status XLAShapeForArgument(
      const Argument& arg, bool is_entry_computation,
      const std::optional<xla::HloSharding>& arg_sharding,
      xla::Shape* xla_shape) const;

  // Retrieves the channel handle associated with `key`. Allocates
  // a new channel handle if none exists.
  // Channel handles can be used to communicate between different
  // computations. Computations that communicate should be compiled with the
  // same XlaCompiler.
  absl::Status GetChannelHandle(const string& key, xla::ChannelHandle* channel);

  // Retrieves the host-to-device channel handle associated with `key`.
  // Allocates a new channel handle if none exists.
  absl::Status GetHostToDeviceChannelHandle(const string& key,
                                            xla::ChannelHandle* channel);

  // Retrieves the device-to-host channel handle associated with `key`.
  // Allocates a new channel handle if none exists.
  absl::Status GetDeviceToHostChannelHandle(const string& key,
                                            xla::ChannelHandle* channel);

  // Sets the shapes and types for the device to host transfer associated with
  // 'key'.
  absl::Status SetDeviceToHostMetadata(const string& key,
                                       absl::Span<const DataType> types,
                                       absl::Span<const TensorShape> shapes);

  // Gets the shapes the device to host transfer associated with 'key'.
  absl::Status GetDeviceToHostShapes(const string& key,
                                     std::vector<TensorShape>* shapes) const;

  // Sets the shapes and types for the host to device transfer associated with
  // 'key'.
  absl::Status SetHostToDeviceMetadata(const string& key,
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
  absl::Status GetHostComputeControlDependency(const string& host_compute_name,
                                               xla::XlaOp* handle);
  absl::Status SetHostComputeControlDependency(const string& host_compute_name,
                                               xla::XlaOp handle);

  const Options& options() const { return options_; }
  xla::Client* client() const { return options_.client; }
  FunctionLibraryRuntime* flib_runtime() const { return flib_runtime_; }

  void PushNodeTokenMapping();
  absl::Status PopNodeTokenMapping();
  absl::Status SetNodeToken(const string& node_name, xla::XlaOp op);
  absl::StatusOr<xla::XlaOp> GetNodeToken(const string& node_name);

  // Sets the function body `fbody` to the one registered as `function`.
  absl::Status FindFunctionBody(const NameAttrList& function,
                                const FunctionBody** fbody,
                                const ConfigProto** config_proto = nullptr);

 private:
  absl::Mutex channel_mutex_;
  // Returns the optimized graph object in this function body.
  std::unique_ptr<Graph> GetGraph(const FunctionBody* fbody);

  // Builds XLA computations for each of the arguments to the computation.
  // `args` are the arguments to the computation.
  absl::Status BuildArguments(
      const Graph& graph, const std::vector<XlaCompiler::Argument>& args,
      bool use_tuple_arg, xla::XlaBuilder* builder, XlaContext* context,
      const std::map<int, xla::OpSharding>& arg_shardings,
      std::vector<XlaExpression>* arg_expressions,
      std::vector<int>* input_to_args, std::vector<xla::Shape>* input_shapes,
      bool is_entry_computation);

  xla::ChannelHandle NewChannel(xla::ChannelHandle::ChannelType type);

  // Graph compiler needs to know how to get an optimized graph from a function
  // body.
  friend class GraphCompiler;
  friend class XlaCompilerTest;

  Options options_;

  // Status set to non-OK in the constructor if initialization fails.
  absl::Status initialization_status_;

  // Returns the next step sequence number.
  int64_t NextStepId();

  // Internal sequence number for steps executed on the compilation device.
  int64_t next_step_id_;

  XlaCompilationDevice* device_;  // Owned by device_mgr_
  StaticDeviceMgr device_mgr_;

  // The next sequence number to assign to a channel.
  int64_t next_channel_ ABSL_GUARDED_BY(channel_mutex_) = 1;

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

  // This is used to store <node name, token output> mapping. Side-effecting
  // ops call SetNodeToken() to record its token output, so later side-effecting
  // ops can use GetNodeToken() to get it and use it as token input.
  //
  // It's a stack because we need a mapping like this for each level of nested
  // CompileGraph() call. In CompileGraph(), we will push a new mapping to the
  // stack, and pop the mapping before returning.
  std::stack<std::map<string, xla::XlaOp>> node_token_mapping_stack_;

  XlaCompiler(const XlaCompiler&) = delete;
  void operator=(const XlaCompiler&) = delete;
};


}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_XLA_COMPILER_H_
