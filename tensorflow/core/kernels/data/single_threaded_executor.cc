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

#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {
namespace {

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<DeviceContext*, 4> DeviceContextVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

class SingleThreadedExecutorImpl : public Executor {
 public:
  explicit SingleThreadedExecutorImpl(const LocalExecutorParams& params)
      : params_(params) {}

  ~SingleThreadedExecutorImpl() override {
    for (const KernelState& kernel_state : kernels_) {
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

    kernels_.resize(ordered_nodes.size());

    std::unordered_map<Node*, size_t> node_to_index_map;

    // Create the kernel and input-related structures for each node in `graph`.
    for (size_t i = 0; i < ordered_nodes.size(); ++i) {
      Node* n = ordered_nodes[i];
      node_to_index_map[n] = i;

      for (DataType dt : n->output_types()) {
        if (IsRefType(dt)) {
          return errors::Unimplemented(
              "Single-threaded executor does not support reference-typed "
              "edges.");
        }
      }

      if (n->IsControlFlow()) {
        return errors::Unimplemented(
            "Single-threaded executor does not support control flow.");
      }
      if (n->IsSend() || n->IsHostSend() || n->IsRecv() || n->IsHostRecv()) {
        return errors::Unimplemented(
            "Single-threaded executor does not support partitioned graphs.");
      }
      if (n->IsCollective()) {
        return errors::Unimplemented(
            "Single-threaded executor does not support collective ops.");
      }

      KernelState& kernel_state = kernels_[i];
      TF_RETURN_IF_ERROR(params_.create_kernel(n->def(), &kernel_state.kernel));
      kernel_state.num_inputs = n->num_inputs();
      kernel_state.num_outputs = n->num_outputs();

      if (i == 0) {
        kernel_state.input_start_index = 0;
      } else {
        const KernelState& previous_kernel_state = kernels_[i - 1];
        kernel_state.input_start_index =
            previous_kernel_state.input_start_index +
            previous_kernel_state.num_inputs;
      }
    }

    // Build the mapping from each node output to the input slot for the
    // corresponding destination node.
    for (size_t i = 0; i < ordered_nodes.size(); ++i) {
      Node* n = ordered_nodes[i];
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
      for (size_t i = 0; i < ordered_nodes.size(); ++i) {
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

  // TODO(mrry): Consider specializing the implementation of Executor::Run()
  // instead, to avoid unnecessary atomic operations in the callback when
  // running synchronously.
  void RunAsync(const Args& args, DoneCallback done) override {
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
    std::vector<ManualConstructor<Tensor>> inputs(total_num_inputs_);

    // TODO(mrry): Can we avoid copying into these vectors? Consider modifying
    // OpKernelContext to take the TensorValueVec as a pointer into `inputs`.
    TensorValueVec node_inputs;
    DeviceContextVec input_device_contexts;
    AllocatorAttributeVec input_alloc_attrs;

    // Prepare the parameters that will be the same for all kernels.
    OpKernelContext::Params params;
    params.step_id = args.step_id;
    Device* device = params_.device;
    params.device = device;
    params.log_memory = false;              // TODO(mrry): Too severe?
    params.record_tensor_accesses = false;  // TODO(mrry): Too severe?
    params.rendezvous = args.rendezvous;
    params.session_state = args.session_state;
    params.tensor_store = args.tensor_store;
    params.cancellation_manager = args.cancellation_manager;
    // TODO(mrry): ArgOp is a relatively expensive OpKernel due to the Tensor
    // allocations that it performs. Consider specializing its handling in the
    // executor.
    params.call_frame = args.call_frame;
    params.function_library = params_.function_library;
    params.resource_manager = device->resource_manager();
    params.step_container = args.step_container;
    params.slice_reader_cache = nullptr;  // TODO(mrry): Too severe?
    params.inputs = &node_inputs;
    params.input_device_contexts = &input_device_contexts;
    params.input_alloc_attrs = &input_alloc_attrs;

    Args::Runner runner_copy = args.runner;
    params.runner = &runner_copy;
    params.stats_collector = args.stats_collector;

    // NOTE(mrry): We are assuming that the graph is loopless and condless.
    params.frame_iter = FrameAndIter(0, 0);
    params.is_input_dead = false;

    // TODO(mrry): Add non-default device context inference.
    params.op_device_context = nullptr;
    // TODO(mrry): Consider implementing forwarding.
    params.forward_from_array = nullptr;

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
        auto t = inputs[input_start_index + j].get();
        node_inputs[j].tensor = t;
        input_alloc_attrs[j] = input_alloc_attrs_[input_start_index + j];
      }
      params.op_kernel = kernel_state.kernel;
      input_device_contexts.clear();
      input_device_contexts.resize(num_inputs);
      params.output_attr_array = kernel_state.output_alloc_attrs.data();
      OpKernelContext ctx(&params, num_outputs);

      // Actually execute the kernel.
      device->Compute(kernel_state.kernel, &ctx);

      if (!ctx.status().ok()) {
        // On failure, we must manually free all intermediate tensors. We have
        // already freed all the inputs for kernels up to (but not including)
        // the `i`th kernel. We scan through the previously executed kernels and
        // destroy any tensors that were destined to be the input for a kernel
        // that has not yet executed.
        for (size_t j = 0; j < i; ++j) {
          const KernelState& executed_kernel_state = kernels_[j];
          for (size_t k = 0; k < executed_kernel_state.num_outputs; ++k) {
            for (size_t output_location :
                 executed_kernel_state.output_locations[k]) {
              if (output_location >= input_start_index) {
                // Only destroy an output location if it is an input to an
                // operation that has not yet executed.
                inputs[output_location].Destroy();
              }
            }
          }
        }
        done(ctx.status());
        return;
      }

      // Free the inputs to the current kernel.
      for (size_t j = 0; j < num_inputs; ++j) {
        inputs[input_start_index + j].Destroy();
      }

      // Forward the outputs of the kernel to the inputs of subsequent kernels.
      for (size_t j = 0; j < num_outputs; ++j) {
        TensorValue val = ctx.release_output(j);
        // TODO(mrry): Consider flattening the `output_locations` vector
        // to improve the cache-friendliness of this loop.
        for (size_t output_location : kernel_state.output_locations[j]) {
          // TODO(mrry): Validate that the types match the expected values or
          // ensure that the necessary validation has already happened.
          inputs[output_location].Init(*val.tensor);
        }
        delete val.tensor;
      }
    }
    done(Status::OK());
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
    // copied. See comment at the beginning of `RunAsync()` for details.
    std::vector<std::vector<size_t>>
        output_locations;  // Length = `num_outputs`.

    // Memory space information for each output of `kernel`.
    std::vector<AllocatorAttributes>
        output_alloc_attrs;  // Length = `num_outputs`.
  };
  std::vector<KernelState> kernels_;

  // Memory space information for each input. This information is stored in the
  // same order as the flat `inputs` vector. See comment at the beginning of
  // `RunAsync()` for details.
  std::vector<AllocatorAttributes>
      input_alloc_attrs_;  // Length = `total_num_inputs_`.
};

class SingleThreadedExecutorRegistrar {
 public:
  SingleThreadedExecutorRegistrar() {
    ExecutorFactory::Register("SINGLE_THREADED_EXECUTOR", new Factory());
  }

 private:
  class Factory : public ExecutorFactory {
    Status NewExecutor(const LocalExecutorParams& params,
                       std::unique_ptr<const Graph> graph,
                       std::unique_ptr<Executor>* out_executor) override {
      Executor* ret;
      TF_RETURN_IF_ERROR(
          NewSingleThreadedExecutor(params, std::move(graph), &ret));
      out_executor->reset(ret);
      return Status::OK();
    }
  };
};
static SingleThreadedExecutorRegistrar registrar;

}  // namespace

Status NewSingleThreadedExecutor(const LocalExecutorParams& params,
                                 std::unique_ptr<const Graph> graph,
                                 Executor** executor) {
  std::unique_ptr<SingleThreadedExecutorImpl> impl(
      new SingleThreadedExecutorImpl(params));
  TF_RETURN_IF_ERROR(impl->Initialize(*graph));
  *executor = impl.release();
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
