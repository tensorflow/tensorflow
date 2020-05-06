/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/single_threaded_executor.h"

#include "tensorflow/core/common_runtime/entry.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {
namespace {

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

static const string& kSingleThreadedExecutor =
    *new string("SINGLE_THREADED_EXECUTOR");

class SingleThreadedExecutorImpl : public Executor {
 public:
  explicit SingleThreadedExecutorImpl(const LocalExecutorParams& params)
      : params_(params) {}

  ~SingleThreadedExecutorImpl() override {
    for (const KernelState& kernel_state : kernels_) {
      params_.delete_kernel(kernel_state.kernel);
    }
    for (const ConstTensorKernelState& kernel_state : const_tensor_kernels_) {
      params_.delete_kernel(kernel_state.kernel);
    }
  }

  Status Initialize(const Graph& graph) {
    // Topologicially sort `graph` to get a sequence of OpKernels.
    std::vector<Node*> ordered_nodes;
    ordered_nodes.reserve(graph.num_nodes());
    GetReversePostOrder(graph, &ordered_nodes);

    if (ordered_nodes.size() != graph.num_nodes()) {
      return errors::InvalidArgument("Graph had ", graph.num_nodes(),
                                     " but reverse post-order had ",
                                     ordered_nodes.size());
    }

    kernels_.reserve(ordered_nodes.size());
    std::vector<Node*> nodes_with_kernels;
    std::vector<Node*> nodes_with_const_tensor_kernels;
    nodes_with_kernels.reserve(ordered_nodes.size());

    std::map<size_t, Node*> arg_index_to_node_map;
    std::unordered_map<Node*, size_t> node_to_index_map;

    // Create the kernel and input-related structures for each node in `graph`.
    for (Node* n : ordered_nodes) {
      for (DataType dt : n->output_types()) {
        if (IsRefType(dt)) {
          return errors::Unimplemented(
              "Single-threaded executor does not support reference-typed "
              "edges.  But saw type ",
              DataTypeString(dt), " in outputs of node ", n->name());
        }
      }
      if (n->IsControlFlow()) {
        return errors::FailedPrecondition(
            "Single-threaded executor does not support low level control flow, "
            " but saw control flow node ",
            n->name(),
            ".  Perhaps your graph contains old-style control flow primitives? "
            "Try using tf.compat.v1.enable_control_flow_v2().");
      }
      if (n->IsSend() || n->IsHostSend() || n->IsRecv() || n->IsHostRecv()) {
        return errors::Unimplemented(
            "Single-threaded executor does not support partitioned graphs.  "
            "But saw send/recv node ",
            n->name());
      }
      if (n->IsCollective()) {
        return errors::Unimplemented(
            "Single-threaded executor does not support collective ops.  But "
            "saw collective node ",
            n->name());
      }

      if (n->IsArg()) {
        int32 arg_index;
        TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &arg_index));
        if (arg_index < 0) {
          return errors::InvalidArgument("Invalid argument index ", arg_index,
                                         " in node ", n->name());
        }
        arg_index_to_node_map[arg_index] = n;
        // We do not create a kernel for Arg nodes, and instead inline the
        // argument handling directly in the executor code.
        continue;
      }

      OpKernel* kernel;
      TF_RETURN_IF_ERROR(params_.create_kernel(n->properties(), &kernel));

      const Tensor* const_tensor;
      if (n->num_outputs() == 1 && (const_tensor = kernel->const_tensor())) {
        // Nodes that produce a single constant tensor are handled specially:
        // we evaluate the tensor once, and propagate it to its consumers as
        // a `const Tensor*`, to avoid refcount manipulation.
        const size_t kernel_index = const_tensor_kernels_.size();
        const_tensor_kernels_.push_back({});
        nodes_with_const_tensor_kernels.push_back(n);
        ConstTensorKernelState& kernel_state =
            const_tensor_kernels_[kernel_index];
        kernel_state.kernel = kernel;
        kernel_state.const_tensor = *const_tensor;
      } else {
        const size_t kernel_index = kernels_.size();
        kernels_.push_back({});
        nodes_with_kernels.push_back(n);
        KernelState& kernel_state = kernels_[kernel_index];
        kernel_state.kernel = kernel;
        kernel_state.num_inputs = n->num_inputs();
        kernel_state.num_outputs = n->num_outputs();
        node_to_index_map[n] = kernel_index;
        if (kernel_index == 0) {
          kernel_state.input_start_index = 0;
        } else {
          const KernelState& previous_kernel_state = kernels_[kernel_index - 1];
          kernel_state.input_start_index =
              previous_kernel_state.input_start_index +
              previous_kernel_state.num_inputs;
        }
      }
    }

    // Build the mapping from each Arg node output to the input slot for the
    // corresponding destination node.
    if (!arg_index_to_node_map.empty()) {
      const size_t num_args = arg_index_to_node_map.rbegin()->first + 1;
      arg_output_locations_.resize(num_args);
      for (const auto& arg_index_node_pair : arg_index_to_node_map) {
        const size_t arg_index = arg_index_node_pair.first;
        const Node* arg_node = arg_index_node_pair.second;
        arg_output_locations_[arg_index].reserve(arg_node->out_edges().size());
        for (const Edge* e : arg_node->out_edges()) {
          if (e->src_output() == Graph::kControlSlot) {
            continue;
          } else if (e->src_output() != 0) {
            return errors::Internal("Invalid output index ", e->src_output(),
                                    " from argument node ", arg_index);
          }
          arg_output_locations_[arg_index].push_back(
              kernels_[node_to_index_map[e->dst()]].input_start_index +
              e->dst_input());
        }
      }
    }

    // Build the mapping from each const tensor kernel to the input slot for the
    // corresponding destination node.
    for (size_t i = 0; i < const_tensor_kernels_.size(); ++i) {
      Node* n = nodes_with_const_tensor_kernels[i];
      ConstTensorKernelState& kernel_state = const_tensor_kernels_[i];
      for (const Edge* e : n->out_edges()) {
        if (e->src_output() == Graph::kControlSlot) {
          continue;
        } else if (e->src_output() != 0) {
          return errors::Internal("Invalid output index ", e->src_output(),
                                  " from node ", n->DebugString());
        }
        kernel_state.output_locations.push_back(
            kernels_[node_to_index_map[e->dst()]].input_start_index +
            e->dst_input());
      }

      bool on_host =
          kernel_state.kernel->output_memory_types()[0] == HOST_MEMORY;
      kernel_state.output_alloc_attr.set_on_host(on_host);
    }

    // Build the mapping from each node output to the input slot for the
    // corresponding destination node.
    for (size_t i = 0; i < kernels_.size(); ++i) {
      Node* n = nodes_with_kernels[i];
      KernelState& kernel_state = kernels_[i];
      kernel_state.output_locations.resize(kernel_state.num_outputs);
      for (const Edge* e : n->out_edges()) {
        if (!e->IsControlEdge()) {
          kernel_state.output_locations[e->src_output()].push_back(
              kernels_[node_to_index_map[e->dst()]].input_start_index +
              e->dst_input());
        }
      }

      // Compute allocator attributes for each node output, and corresponding
      // node input.
      kernel_state.output_alloc_attrs.resize(kernel_state.num_outputs);
      AllocatorAttributes* attrs = kernel_state.output_alloc_attrs.data();

      OpKernel* op_kernel = kernel_state.kernel;
      for (int out = 0; out < n->num_outputs(); out++) {
        DCHECK_LT(out, op_kernel->output_memory_types().size());
        bool on_host = op_kernel->output_memory_types()[out] == HOST_MEMORY;
        if (on_host) {
          AllocatorAttributes h;
          h.set_on_host(on_host);
          attrs[out].Merge(h);
        }
      }
    }

    if (!kernels_.empty()) {
      const KernelState& last_kernel_state = kernels_.back();
      total_num_inputs_ =
          last_kernel_state.input_start_index + last_kernel_state.num_inputs;
      input_alloc_attrs_.resize(total_num_inputs_);
      for (size_t i = 0; i < kernels_.size(); ++i) {
        for (size_t j = 0; j < kernels_[i].output_locations.size(); ++j) {
          for (size_t output_location : kernels_[i].output_locations[j]) {
            input_alloc_attrs_[output_location] =
                kernels_[i].output_alloc_attrs[j];
          }
        }
      }
    } else {
      total_num_inputs_ = 0;
    }
    return Status::OK();
  }

  Status Run(const Args& args) override {
    // The inputs to each kernel are stored contiguously in `inputs`.
    //
    // We use `kernels_[i].input_start_index` and `kernels_[i].num_inputs` to
    // determine the range of elements in this vector that correspond to
    // the inputs of `kernels_[i]`.
    //
    // This vector has the following layout:
    //
    // * Kernel 0, input 0.
    // * Kernel 0, input 1.
    // * ...
    // * Kernel 0, input `kernels_[0].num_inputs - 1`.
    // * Kernel 1, input 0.
    // * ...
    // * Kernel 1, input `kernels_[1].num_inputs - 1`.
    // * ...
    // * Kernel `kernels_.size() - 1`, input 0.
    // * ...
    // * Kernel `kernels_.size() - 1`, input `kernels_.back().num_inputs - 1`.
    //
    // Note that kernels with zero inputs do not correspond to any elements in
    // this vector.
    //
    // We use `ManualConstructor<Tensor>` to avoid the overhead of
    // default-constructing an invalid `Tensor` for each slot at the beginning
    // of execution:
    // * Elements are initialized when the outputs of a kernel execution are
    //   propagated to the inputs of kernels that depend on them.
    // * The elements corresponding to the inputs for kernel `i` are destroyed
    //   after kernel `i` executes.
    // * In an error case (see below), we use the connectivity information in
    //   `KernelState::output_locations` to determine which locations have been
    //   initialized, and manually destroy them.
    std::vector<Entry> inputs(total_num_inputs_);

    // TODO(mrry): Can we avoid copying into these vectors? Consider modifying
    // OpKernelContext to take the TensorValueVec as a pointer into `inputs`.
    TensorValueVec node_inputs;
    AllocatorAttributeVec input_alloc_attrs;

    // Prepare the parameters that will be the same for all kernels.
    OpKernelContext::Params params;
    params.step_id = args.step_id;
    Device* device = params_.device;
    params.device = device;
    params.log_memory = false;              // TODO(mrry): Too severe?
    params.rendezvous = args.rendezvous;
    params.session_state = args.session_state;
    params.tensor_store = args.tensor_store;
    params.cancellation_manager = args.cancellation_manager;
    params.call_frame = args.call_frame;
    params.function_library = params_.function_library;
    params.resource_manager = device->resource_manager();
    params.step_container = args.step_container;
    params.slice_reader_cache = nullptr;  // TODO(mrry): Too severe?
    params.inputs = &node_inputs;
    params.input_alloc_attrs = &input_alloc_attrs;

    Args::Runner runner_copy = args.runner;
    params.runner = &runner_copy;
    params.run_all_kernels_inline = args.run_all_kernels_inline;
    params.stats_collector = args.stats_collector;
    params.executor_type = &kSingleThreadedExecutor;

    // NOTE(mrry): We are assuming that the graph is loopless and condless.
    params.frame_iter = FrameAndIter(0, 0);
    params.is_input_dead = false;

    // TODO(mrry): Add non-default device context inference.
    params.op_device_context = nullptr;
    // TODO(mrry): Consider implementing forwarding.
    params.forward_from_array = nullptr;

    const size_t received_args =
        args.call_frame ? args.call_frame->num_args() : 0;
    if (arg_output_locations_.size() > received_args) {
      return errors::InvalidArgument("Expected ", arg_output_locations_.size(),
                                     " arguments, but only received ",
                                     received_args, ".");
    }

    // ArgOp is a relatively expensive OpKernel due to the Tensor
    // allocations that it performs. Therefore we specialize its implementation
    // and forward arguments directly to the inputs of kernels that consume
    // them.
    for (size_t i = 0; i < arg_output_locations_.size(); ++i) {
      const size_t num_destinations = arg_output_locations_[i].size();
      if (num_destinations > 0) {
        if (args.call_frame->CanConsumeArg(i)) {
          // The first destination input can consume the argument.
          Entry& first_input = inputs[arg_output_locations_[i][0]];
          first_input.state = Entry::State::HAS_VALUE;
          first_input.val.Init();
          args.call_frame->ConsumeArg(i, first_input.val.get());
          // All subsequent destination inputs get a shallow copy of the first
          // destination input.
          //
          // NOTE: If we had metadata about which kernels might attempt to
          // forward their input, we could arrange the kernel order so that
          // one of those kernels was executed last.
          for (size_t j = 1; j < num_destinations; ++j) {
            Entry& input = inputs[arg_output_locations_[i][j]];
            input.state = Entry::State::HAS_VALUE;
            input.val.Init(*first_input.val);
          }
        } else {
          const Tensor* arg;
          TF_CHECK_OK(args.call_frame->GetArg(i, &arg));
          for (size_t j = 0; j < num_destinations; ++j) {
            Entry& input = inputs[arg_output_locations_[i][j]];
            // NOTE: We must make at least one shallow copy of the argument
            // tensor that remains live until all consuming kernels have
            // executed, to keep the reference count > 1, and inhibit buffer
            // forwarding. For simplicity, we shallow copy into the input entry
            // for each consuming kernel.
            input.state = Entry::State::HAS_VALUE;
            input.val.Init(*arg);
          }
        }
      }
    }

    // Kernels that return a constant value (e.g. ConstOp) are relatively
    // expensive due to the Tensor allocations that they perform. Therefore we
    // specialize their implementation and forward their constant value directly
    // to the inputs of kernels that consume them.
    for (const ConstTensorKernelState& kernel_state : const_tensor_kernels_) {
      for (size_t i = 0; i < kernel_state.output_locations.size(); ++i) {
        Entry& input = inputs[kernel_state.output_locations[i]];
        input.state = Entry::State::HAS_CONST_TENSOR;
        input.const_tensor = &kernel_state.const_tensor;
      }
    }

    // Execute the kernels one-at-a-time in topological order.
    for (size_t i = 0; i < kernels_.size(); ++i) {
      const KernelState& kernel_state = kernels_[i];

      // Prepare the per-kernel parameters.
      const size_t input_start_index = kernel_state.input_start_index;
      const size_t num_inputs = kernel_state.num_inputs;
      const size_t num_outputs = kernel_state.num_outputs;

      node_inputs.clear();
      node_inputs.resize(num_inputs);
      input_alloc_attrs.clear();
      input_alloc_attrs.resize(num_inputs);
      for (size_t j = 0; j < num_inputs; ++j) {
        Entry& input = inputs[input_start_index + j];
        switch (input.state) {
          case Entry::State::HAS_CONST_TENSOR:
            // NOTE(mrry): This `const_cast` is necessary because `TensorValue`
            // stores a non-const `Tensor*`, and relies on the `OpKernelContext`
            // accessors making dynamic checks that prevent using an immutable
            // tensor as a mutable tensor.
            node_inputs[j].tensor = const_cast<Tensor*>(input.const_tensor);
            break;
          case Entry::State::HAS_VALUE:
            node_inputs[j].tensor = input.val.get();
            break;
          default:
            DCHECK(false) << "Input did not have a valid value.";
        }
        input_alloc_attrs[j] = input_alloc_attrs_[input_start_index + j];
      }
      params.op_kernel = kernel_state.kernel;
      params.output_attr_array = kernel_state.output_alloc_attrs.data();
      OpKernelContext ctx(&params, num_outputs);

      // Actually execute the kernel.
      device->Compute(kernel_state.kernel, &ctx);
      TF_RETURN_IF_ERROR(ctx.status());

      // Free the inputs to the current kernel.
      for (size_t j = 0; j < num_inputs; ++j) {
        inputs[input_start_index + j].ClearVal();
      }

      // Forward the outputs of the kernel to the inputs of subsequent kernels.
      for (size_t j = 0; j < num_outputs; ++j) {
        TensorValue val = ctx.release_output(j);
        const size_t num_destinations = kernel_state.output_locations[j].size();
        if (num_destinations > 0) {
          // TODO(mrry): Consider flattening the `output_locations` vector
          // to improve the cache-friendliness of this loop.
          for (size_t k = 0; k < num_destinations - 1; ++k) {
            // TODO(mrry): Validate that the types match the expected values or
            // ensure that the necessary validation has already happened.
            Entry& input = inputs[kernel_state.output_locations[j][k]];
            input.state = Entry::State::HAS_VALUE;
            input.val.Init(*val.tensor);
          }
          // Move `arg` to the last consumer to avoid the cost of copying it.
          Entry& input =
              inputs[kernel_state.output_locations[j][num_destinations - 1]];
          input.state = Entry::State::HAS_VALUE;
          input.val.Init(std::move(*val.tensor));
        }
        delete val.tensor;
      }
    }
    return Status::OK();
  }

  void RunAsync(const Args& args, DoneCallback done) override {
    done(Run(args));
  }

 private:
  const LocalExecutorParams params_;

  // All following members are read-only after Initialize().

  // The sum of the number of inputs for each node in the graph. This determines
  // the length of the flat `inputs` vector. See comment at the beginning of
  // `RunAsync()` for details.
  size_t total_num_inputs_;

  // Represents cached graph structure state for each kernel.
  struct KernelState {
    // The kernel object. Not owned.
    //
    // This pointer is managed by `params_.create_kernel()` and
    // `params_.delete_kernel()`.
    OpKernel* kernel;

    // These fields determine the range of elements in `inputs` that corresponds
    // to the inputs of `kernel`.
    size_t input_start_index;
    size_t num_inputs;

    size_t num_outputs;

    // For the `j`th output of `kernel`, `output_locations[j]` contains the
    // locations in the flat `inputs` vector to which that output must be
    // copied. See comment at the beginning of `Run()` for details.
    std::vector<std::vector<size_t>>
        output_locations;  // Length = `num_outputs`.

    // Memory space information for each output of `kernel`.
    std::vector<AllocatorAttributes>
        output_alloc_attrs;  // Length = `num_outputs`.
  };
  std::vector<KernelState> kernels_;

  // For the `i`th argument, `arg_output_locations_[i]` contains the locations
  // in the flat `inputs` vector to which that argument must be copied.
  std::vector<std::vector<size_t>>
      arg_output_locations_;  // Length = `num_args`.

  // Represents cached graph structure state for each kernel that produces
  // a single constant-valued tensor.
  struct ConstTensorKernelState {
    // The kernel object. Not owned.
    //
    // This pointer is managed by `params_.create_kernel()` and
    // `params_.delete_kernel()`.
    OpKernel* kernel;

    // The cached value of `kernel->const_tensor()`.
    //
    // NOTE: We keep a `Tensor` rather than a `const Tensor*` here in order to
    // keep the reference count on the underlying buffer above 1. Otherwise, a
    // kernel could interpret the input as a forwardable tensor, and mutate the
    // underlying constant tensor.
    Tensor const_tensor;

    // For the single output of `kernel`, `output_locations` contains the
    // locations in the flat `inputs` vector to which that output must be
    // copied. See comment at the beginning of `Run()` for details.
    std::vector<size_t> output_locations;  // Length = `num_outputs`.

    // Memory space information for the single output of `kernel`.
    AllocatorAttributes output_alloc_attr;
  };
  std::vector<ConstTensorKernelState> const_tensor_kernels_;

  // Memory space information for each input. This information is stored in the
  // same order as the flat `inputs` vector. See comment at the beginning of
  // `RunAsync()` for details.
  std::vector<AllocatorAttributes>
      input_alloc_attrs_;  // Length = `total_num_inputs_`.
};

class SingleThreadedExecutorRegistrar {
 public:
  SingleThreadedExecutorRegistrar() {
    ExecutorFactory::Register(kSingleThreadedExecutor, new Factory());
  }

 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams& params, const Graph& graph,
                       std::unique_ptr<Executor>* out_executor) override {
      Executor* ret;
      TF_RETURN_IF_ERROR(NewSingleThreadedExecutor(params, graph, &ret));
      out_executor->reset(ret);
      return Status::OK();
    }
  };
};
static SingleThreadedExecutorRegistrar registrar;

}  // namespace

Status NewSingleThreadedExecutor(const LocalExecutorParams& params,
                                 const Graph& graph, Executor** executor) {
  auto impl = absl::make_unique<SingleThreadedExecutorImpl>(params);
  TF_RETURN_IF_ERROR(impl->Initialize(graph));
  *executor = impl.release();
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
