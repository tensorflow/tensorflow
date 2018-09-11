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
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_partition.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/util/reffed_status_callback.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/stream.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
typedef FunctionLibraryRuntime::Handle FHandle;

namespace {

// A `PartitionedCallOp` asynchronously executes a function, potentially across
// multiple devices but within a single process. The kernel places and
// partitions a given function's underlying graph, and executes each of the
// partitioned subgraphs as a function.
//
// TODO(akshayka): Support distributed execution.
class PartitionedCallOp : public AsyncOpKernel {
 public:
  explicit PartitionedCallOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
  }

  ~PartitionedCallOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                      errors::Internal("No function library is provided."),
                      done);

    OpInputList args;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("args", &args), done);

    // The function body's graph is placed and partitioned the first time
    // `ComputeAsync` is invoked; every subsequent invocation calls each
    // of the function shards yielded by partitioning.
    //
    // The partitioning step yields a set of devices on which to run the
    // function, and exactly one function shard is created for each device
    // Inputs and outputs are pinned to the local device, for simplicity.
    //
    // TODO(akshayka): Support re-sharding the function on subsequent calls,
    // via, e.g., virtual device annotations and a list of device names supplied
    // through an attribute.
    //
    // TODO(akshayka): Add a fastpath for functions that execute on a single
    // device.
    {
      mutex_lock l(mu_);
      if (function_handles_.find(lib) == function_handles_.end()) {
        if (local_device_name_.empty()) {
          // The full local device name isn't known at kernel construction
          // time, hence the need to set it here.
          local_device_name_ = lib->device()->name();
        }

        // TODO(b/37549631): Because this kernel may correspond to a stateful
        // op, it may be shared by multiple subgraphs, which in turn may have
        // different `FunctionLibraryRuntime` objects and therefore different
        // `FHandle` namespaces. As such, we partition on a per-FLR basis.
        FunctionLibraryRuntime::InstantiateOptions opts;
        FHandle handle;
        OP_REQUIRES_OK_ASYNC(
            ctx,
            lib->Instantiate(func_.name(), AttrSlice(&func_.attr()), opts,
                             &handle),
            done);
        const FunctionBody* fbody = lib->GetFunctionBody(handle);
        OP_REQUIRES_ASYNC(ctx, fbody != nullptr,
                          errors::Internal("Could not find handle ", handle),
                          done);
        auto graph = tensorflow::MakeUnique<Graph>(fbody->graph->flib_def());
        CopyGraph(*fbody->graph, graph.get());
        OP_REQUIRES_OK_ASYNC(ctx, PinResourceArgs(graph.get(), args), done);

        DeviceSet device_set;
        for (auto d : lib->device_mgr()->ListDevices()) {
          device_set.AddDevice(d);
        }

        // The FunctionLibraryRuntime's library cannot be mutated from within
        // an OpKernel, so functions are instantiated in an overlay library.
        OP_REQUIRES_ASYNC(
            ctx, overlay_libs_.find(lib) == overlay_libs_.end(),
            errors::Internal("Found an overlay library but did not "
                             "find cached function partitions; "
                             "this indicates a bug."),
            done);
        FunctionLibraryDefinition* overlay_lib =
            new FunctionLibraryDefinition(*lib->GetFunctionLibraryDefinition());
        overlay_libs_.emplace(lib, overlay_lib);

        GraphOptimizationPassOptions optimization_options;
        // TODO(akshayka): Thread SessionOptions (if any) into this kernel, or
        // make it possible to specify the relevant options via attributes.
        SessionOptions session_options;
        session_options.env = ctx->env();
        optimization_options.session_options = &session_options;
        optimization_options.graph = &graph;
        optimization_options.flib_def = overlay_lib;
        optimization_options.device_set = &device_set;
        Placer placer(graph.get(), &device_set);
        OP_REQUIRES_OK_ASYNC(
            ctx,
            OptimizationPassRegistry::Global()->RunGrouping(
                OptimizationPassRegistry::PRE_PLACEMENT, optimization_options),
            done);
        OP_REQUIRES_OK_ASYNC(ctx, placer.Run(), done);
        OP_REQUIRES_OK_ASYNC(
            ctx,
            OptimizationPassRegistry::Global()->RunGrouping(
                OptimizationPassRegistry::POST_PLACEMENT, optimization_options),
            done);
        OP_REQUIRES_OK_ASYNC(
            ctx,
            OptimizationPassRegistry::Global()->RunGrouping(
                OptimizationPassRegistry::POST_REWRITE_FOR_EXEC,
                optimization_options),
            done);

        std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
        OP_REQUIRES_OK_ASYNC(
            ctx, PartitionHelper(device_set, std::move(graph), &subgraphs),
            done);
        optimization_options.graph = nullptr;
        optimization_options.device_set = nullptr;
        optimization_options.partition_graphs = &subgraphs;
        OP_REQUIRES_OK_ASYNC(ctx,
                             OptimizationPassRegistry::Global()->RunGrouping(
                                 OptimizationPassRegistry::POST_PARTITIONING,
                                 optimization_options),
                             done);

        auto handles = tensorflow::MakeUnique<gtl::FlatMap<string, FHandle>>();
        for (const auto& pair : subgraphs) {
          // TODO(akshayka): Fail gracefully if the set of devices corresponds
          // to more than one address space.
          const string& target = pair.first;
          const auto& subgraph = pair.second;
          OP_REQUIRES_OK_ASYNC(
              ctx, UpdateArgAndRetMetadata(target, subgraph.get()), done);
          FunctionDef shard;
          string unique_name = UniquifyFunctionName(overlay_lib, func_.name());
          OP_REQUIRES_OK_ASYNC(
              ctx, GraphToFunctionDef(*subgraph, unique_name, &shard), done);
          OP_REQUIRES_OK_ASYNC(ctx, overlay_lib->AddFunctionDef(shard), done);
          FunctionLibraryRuntime::InstantiateOptions opts;
          opts.target = target;
          opts.overlay_lib = overlay_lib;
          FHandle handle;
          OP_REQUIRES_OK_ASYNC(
              ctx,
              lib->Instantiate(unique_name, AttrSlice(&shard.attr()), opts,
                               &handle),
              done);
          handles->emplace(target, handle);
        }

        function_handles_.emplace(lib, std::move(handles));
      }
    }
    ExecuteFunctions(lib, ctx, args, std::move(done));
  }

 private:
  typedef std::pair<string, FHandle> DeviceAndFHandle;
  typedef std::pair<std::vector<int>, std::vector<int>> ArgAndRetIndices;
  typedef std::pair<std::vector<AllocatorAttributes>,
                    std::vector<AllocatorAttributes>>
      ArgAndRetAllocAttrs;

  // Pins each arg that emits a `DT_RESOURCE` tensor to the device on which the
  // corresponding resource lives. This ensures that the Placer assigns ops that
  // access these resources to the appropriate devices.
  Status PinResourceArgs(Graph* graph, const OpInputList& args) {
    for (Node* node : graph->op_nodes()) {
      string node_type = node->type_string();
      if (node_type == FunctionLibraryDefinition::kArgOp) {
        const AttrValue* attr_value;
        TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
        int index = attr_value->i();
        TF_RETURN_IF_ERROR(node->attrs().Find("T", &attr_value));
        DataType dtype = attr_value->type();
        if (dtype == DT_RESOURCE) {
          ResourceHandle handle = args[index].flat<ResourceHandle>()(0);
          node->set_assigned_device_name(handle.device());
        }
      }
    }
    return Status::OK();
  }

  // Partitions `graph` and populates `subgraphs` with the partitions.
  Status PartitionHelper(
      const DeviceSet& device_set, std::unique_ptr<Graph> graph,
      std::unordered_map<string, std::unique_ptr<Graph>>* subgraphs) {
    PartitionOptions partition_options;
    partition_options.node_to_loc = [](const Node* node) {
      // TODO(akshayka): To better support the distributed case, first split
      // the graph by worker (e.g,. using the master session's
      // `SplitByWorker` policy), and then recursively partition the
      // per-worker shards at the remote worker(s).
      return node->assigned_device_name();
    };
    int64 edge_name_counter = 0;
    partition_options.new_name = [&edge_name_counter](const string& prefix) {
      return strings::StrCat(prefix, "/_", ++edge_name_counter);
    };
    partition_options.get_incarnation =
        [&device_set](const string& name) -> int64 {
      const Device* d = device_set.FindDeviceByName(name);
      if (d == nullptr) {
        return PartitionOptions::kIllegalIncarnation;
      } else {
        return d->attributes().incarnation();
      }
    };
    partition_options.control_flow_added = false;
    std::unordered_map<string, GraphDef> partitions;
    TF_RETURN_IF_ERROR(Partition(partition_options, graph.get(), &partitions));

    VLOG(3) << "Partitioned function '" << func_.name() << "', yielding "
            << partitions.size() << " shards.";

    const FunctionLibraryDefinition* flib_def = &graph->flib_def();
    for (const auto& partition : partitions) {
      std::unique_ptr<Graph> subgraph(new Graph(flib_def));
      GraphConstructorOptions opts;
      opts.allow_internal_ops = true;
      opts.expect_device_spec = true;
      const string& device = partition.first;
      const GraphDef& graph_def = partition.second;
      TF_RETURN_IF_ERROR(
          ConvertGraphDefToGraph(opts, graph_def, subgraph.get()));
      subgraphs->emplace(device, std::move(subgraph));
    }

    return Status::OK();
  }

  // Each subgraph produced by partitioning the function body contains a subset
  // of the original `Arg` and `Retval` nodes. This function performs
  // bookkeeping to track which `Arg` and `Retval` nodes were placed on a
  // particular device / subgraph.
  //
  // More specifically, this function
  //  (1) rewrites the indices of the `Arg` and `Retval` nodes placed on a
  //      particular device,
  //  (2) records the subsets of `Arg` and `Retval` nodes assigned to the
  //      device, and
  //  (3) records which `Arg` and `Retval` nodes live in host memory.
  Status UpdateArgAndRetMetadata(const string& device, Graph* subgraph) {
    ArgAndRetIndices indices;
    std::vector<int>* arg_indices = &indices.first;
    std::vector<int>* ret_indices = &indices.second;
    std::vector<std::pair<Node*, int>> arg_nodes;
    std::vector<std::pair<Node*, int>> ret_nodes;
    const AttrValue* attr_value;

    // Find the Arg and Retval nodes, along with their corresponding indices
    // in the original function.
    for (Node* node : subgraph->op_nodes()) {
      string node_type = node->type_string();
      if (node_type == FunctionLibraryDefinition::kArgOp) {
        TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
        int index = attr_value->i();
        arg_indices->push_back(index);
        arg_nodes.push_back(std::make_pair(node, index));
      } else if (node_type == FunctionLibraryDefinition::kRetOp) {
        TF_RETURN_IF_ERROR(node->attrs().Find("index", &attr_value));
        int index = attr_value->i();
        ret_indices->push_back(index);
        ret_nodes.push_back(std::make_pair(node, index));
      }
    }

    // Rewrite the indices of the Arg and Retval nodes for this function
    // to range from 0 to the number of Arg nodes, Retval nodes, respectively.
    auto sort_by_index = [](std::pair<Node*, int> one,
                            std::pair<Node*, int> two) -> bool {
      return one.second < two.second;
    };
    std::sort(arg_nodes.begin(), arg_nodes.end(), sort_by_index);
    std::sort(ret_nodes.begin(), ret_nodes.end(), sort_by_index);
    for (int i = 0; i < arg_nodes.size(); ++i) {
      Node* arg = arg_nodes[i].first;
      arg->AddAttr("index", i);
      TF_RETURN_IF_ERROR(arg->attrs().Find("T", &attr_value));
      AllocatorAttributes alloc_attr;
      DataType type = attr_value->type();
      if (MTypeFromDType(type) == HOST_MEMORY) {
        alloc_attr.set_on_host(true);
      }
      arg_and_ret_alloc_attrs_[device].first.push_back(alloc_attr);
    }
    for (int i = 0; i < ret_nodes.size(); ++i) {
      Node* ret = ret_nodes[i].first;
      ret->AddAttr("index", i);
      TF_RETURN_IF_ERROR(ret->attrs().Find("T", &attr_value));
      AllocatorAttributes alloc_attr;
      DataType type = attr_value->type();
      if (MTypeFromDType(type) == HOST_MEMORY) {
        alloc_attr.set_on_host(true);
      }
      arg_and_ret_alloc_attrs_[device].second.push_back(alloc_attr);
    }

    // If this kernel execution corresponds to a StatefulPartitionedCallOp,
    // `arg_and_ret_indices_` might have been populated by a previous
    // invocation.
    if (arg_and_ret_indices_.find(device) == arg_and_ret_indices_.end()) {
      arg_and_ret_indices_.emplace(device, indices);
    }
    return Status::OK();
  }

  std::vector<Tensor> GetArgsForIndices(const std::vector<int>& indices,
                                        const OpInputList& arguments) {
    std::vector<Tensor> args;
    args.reserve(indices.size());
    for (int i : indices) {
      args.push_back(arguments[i]);
    }
    return args;
  }

  void ExecuteFunctions(FunctionLibraryRuntime* lib, OpKernelContext* ctx,
                        const OpInputList& op_args, DoneCallback done)
      LOCKS_EXCLUDED(mu_) {
    const gtl::FlatMap<string, FHandle>* handles;
    {
      mutex_lock l(mu_);
      handles = function_handles_[lib].get();
    }
    if (handles->empty()) {
      // Trivial case where the function body is empty.
      ctx->SetStatus(Status::OK());
      done();
      return;
    }

    FunctionLibraryRuntime::Options opts;
    opts.step_id = ctx->step_id();
    opts.step_container = ctx->step_container();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.stats_collector = ctx->stats_collector();
    // TODO(akshayka): Consider selecting a runner on a per-device basis, i.e.,
    // using device-specific threadpools when available.
    opts.runner = ctx->runner();
    opts.source_device = local_device_name_;
    opts.allow_dead_tensors = true;
    // TODO(akshayka): Accommodate the multiple-worker scenario by adding the
    // constructed rendezvous to a rendezvous manager.
    Rendezvous* rendez = new IntraProcessRendezvous(lib->device_mgr());
    opts.rendezvous = rendez;

    StatusCallback callback = std::bind(
        [](Rendezvous* rendez, DoneCallback& done, const Status& status) {
          rendez->Unref();
          done();
        },
        rendez, std::move(done), std::placeholders::_1);
    auto* refcounted_done = new ReffedStatusCallback(std::move(callback));
    for (int i = 1; i < handles->size(); ++i) {
      refcounted_done->Ref();
    }

    for (const auto& pair : *handles) {
      const string& target = pair.first;
      FHandle handle = pair.second;
      VLOG(3) << "Running function shard on device " << target;
      ArgAndRetIndices indices = arg_and_ret_indices_[target];
      ArgAndRetAllocAttrs alloc_attrs = arg_and_ret_alloc_attrs_[target];
      const std::vector<int>& arg_indices = indices.first;
      const std::vector<int>& ret_indices = indices.second;
      opts.args_alloc_attrs = alloc_attrs.first;
      opts.rets_alloc_attrs = alloc_attrs.second;
      if (target == local_device_name_) {
        opts.remote_execution = false;
        std::vector<Tensor> args = GetArgsForIndices(arg_indices, op_args);
        std::vector<Tensor>* rets = new std::vector<Tensor>;
        lib->Run(
            opts, handle, args, rets,
            [rets, ret_indices, refcounted_done, ctx](const Status& status) {
              if (!status.ok()) {
                VLOG(3) << "Local execution failed: " << status;
                ctx->SetStatus(status);
              } else {
                for (int i = 0; i < rets->size(); ++i) {
                  ctx->set_output(ret_indices[i], (*rets)[i]);
                }
              }
              delete rets;
              VLOG(3) << "Finished local execution.";
              refcounted_done->Unref();
            });
      } else {
        opts.remote_execution = true;
        std::vector<Tensor> args = GetArgsForIndices(arg_indices, op_args);
        std::vector<Tensor>* rets = new std::vector<Tensor>;
        lib->Run(
            opts, handle, args, rets,
            [rets, ret_indices, refcounted_done, ctx](const Status& status) {
              if (!status.ok()) {
                VLOG(3) << "Remote execution failed: " << status;
                ctx->SetStatus(status);
              } else {
                for (int i = 0; i < rets->size(); ++i) {
                  ctx->set_output(ret_indices[i], (*rets)[i]);
                }
              }
              delete rets;
              VLOG(3) << "Finished remote execution.";
              refcounted_done->Unref();
            });
      }
    }
  }

  string UniquifyFunctionName(const FunctionLibraryDefinition* function_library,
                              const string& name) {
    for (;; ++suffix_) {
      const string candidate = strings::StrCat(name, "_", suffix_);
      if (function_library->Find(candidate) == nullptr) {
        return candidate;
      }
    }
  }

  NameAttrList func_;
  string local_device_name_;
  // Contains maps from device names to handles of function partitions, keyed by
  // FunctionLibraryRuntime pointers. (Because this kernel may be instantiated
  // for a stateful op, different invocations of it may use different FLRs.)
  gtl::FlatMap<FunctionLibraryRuntime*,
               std::unique_ptr<gtl::FlatMap<string, FHandle>>>
      function_handles_ GUARDED_BY(mu_);
  // Function partitions are added to overlay libraries.
  gtl::FlatMap<FunctionLibraryRuntime*,
               std::unique_ptr<FunctionLibraryDefinition>>
      overlay_libs_ GUARDED_BY(mu_);
  // Map from device name to the indices of the arguments and return values
  // placed on that device. Read-only after the first invocation.
  gtl::FlatMap<string, ArgAndRetIndices> arg_and_ret_indices_;
  // Map from device name to alloc attrs for arguments and return values of the
  // function placed on that device. Read-only after the first invocation.
  gtl::FlatMap<string, ArgAndRetAllocAttrs> arg_and_ret_alloc_attrs_;

  mutex mu_;

  // Used to uniquify function names in `overlay_libs_`.
  uint32 suffix_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("PartitionedCall").Device(DEVICE_CPU),
                        PartitionedCallOp);
REGISTER_KERNEL_BUILDER(Name("StatefulPartitionedCall").Device(DEVICE_CPU),
                        PartitionedCallOp);
REGISTER_KERNEL_BUILDER(Name("PartitionedCall").Device(DEVICE_GPU),
                        PartitionedCallOp);
REGISTER_KERNEL_BUILDER(Name("StatefulPartitionedCall").Device(DEVICE_GPU),
                        PartitionedCallOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("PartitionedCall").Device(DEVICE_SYCL),
                        PartitionedCallOp);
REGISTER_KERNEL_BUILDER(Name("StatefulPartitionedCall").Device(DEVICE_SYCL),
                        PartitionedCallOp);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace
}  // namespace tensorflow
