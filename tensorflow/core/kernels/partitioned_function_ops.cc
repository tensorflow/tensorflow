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
#include "tensorflow/core/common_runtime/placer.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
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
  explicit PartitionedCallOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx), local_device_name_(ctx->device()->name()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("f", &func_));
  }

  ~PartitionedCallOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                      errors::Internal("No function library is provided."),
                      done);

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
    // TODO(akshayka): Lift the constraint pinning inputs and outputs to the
    // local device.
    //
    // TODO(akshayka): Add a fastpath for functions that execute on a single
    // device.
    {
      mutex_lock l(mu_);
      if (!partitioned_) {
        auto graph = tensorflow::MakeUnique<Graph>(OpRegistry::Global());
        OP_REQUIRES_OK_ASYNC(ctx, GetGraphFromFunction(lib, graph.get()), done);

        DeviceSet device_set;
        for (auto d : lib->device_mgr()->ListDevices()) {
          device_set.AddDevice(d);
        }
        Placer placer(graph.get(), &device_set);
        OP_REQUIRES_OK_ASYNC(ctx, placer.Run(), done);

        std::unordered_map<string, std::unique_ptr<Graph>> subgraphs;
        OP_REQUIRES_OK_ASYNC(
            ctx, PartitionHelper(device_set, std::move(graph), &subgraphs),
            done);

        // The FunctionLibraryRuntime's library cannot be mutated from within
        // an OpKernel, so functions are instantiated in an overlay library.
        overlay_lib_.reset(new FunctionLibraryDefinition(
            *lib->GetFunctionLibraryDefinition()));
        for (const auto& pair : subgraphs) {
          const string& target = pair.first;
          const auto& subgraph = pair.second;
          FunctionDef shard;
          string unique_name = UniquifyFunctionName(func_.name());
          OP_REQUIRES_OK_ASYNC(
              ctx, GraphToFunctionDef(*subgraph, unique_name, &shard), done);
          OP_REQUIRES_OK_ASYNC(ctx, overlay_lib_->AddFunctionDef(shard), done);
          FunctionLibraryRuntime::InstantiateOptions opts;
          opts.target = target;
          opts.overlay_lib = overlay_lib_.get();
          FHandle handle;
          OP_REQUIRES_OK_ASYNC(
              ctx,
              lib->Instantiate(unique_name, AttrSlice(&shard.attr()), opts,
                               &handle),
              done);
          function_handles_.emplace(target, handle);
        }
        partitioned_ = true;
      }
    }
    ExecuteFunctions(lib, ctx, std::move(done));
  }

 private:
  typedef std::pair<string, FHandle> DeviceAndFHandle;

  // `func_` encapsulates the original, unsharded function.
  // Copies the graph backing `func_` into `*graph`, pinning the input and
  // output nodes to the local device.
  //
  // `*graph` must be a freshly allocated graph.
  Status GetGraphFromFunction(FunctionLibraryRuntime* lib, Graph* graph) {
    FunctionLibraryRuntime::InstantiateOptions opts;
    FHandle handle;
    TF_RETURN_IF_ERROR(lib->Instantiate(func_.name(), AttrSlice(&func_.attr()),
                                        opts, &handle));
    const FunctionBody* fbody = lib->GetFunctionBody(handle);
    if (fbody == nullptr) {
      return errors::Internal("Could not find handle ", handle);
    }
    CopyGraph(*fbody->graph, graph);

    // Pin the inputs and outputs to the local device to simplify the
    // function-dispatching logic.
    for (Node* node : graph->op_nodes()) {
      string node_type = node->type_string();
      if (node_type == FunctionLibraryDefinition::kArgOp ||
          node_type == FunctionLibraryDefinition::kRetOp) {
        node->set_assigned_device_name(local_device_name_);
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

  // Executes the partitioned functions.
  void ExecuteFunctions(FunctionLibraryRuntime* lib, OpKernelContext* ctx,
                        DoneCallback done) LOCKS_EXCLUDED(mu_) {
    FunctionLibraryRuntime::Options opts;
    opts.step_id = ctx->step_id();
    opts.step_container = ctx->step_container();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.stats_collector = ctx->stats_collector();
    // TODO(akshayka): Consider selecting a runner on a per-device basis, i.e.,
    // using device-specific threadpools when available.
    opts.runner = ctx->runner();
    opts.source_device = local_device_name_;
    // TODO(akshayka): Accommodate the multiple-worker scenario by adding the
    // constructed rendezvous to a rendezvous manager.
    Rendezvous* rendez = new IntraProcessRendezvous(lib->device_mgr());
    opts.rendezvous = rendez;

    OpInputList arguments;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("args", &arguments), done);
    // Dummy args vector for the remote shards, which do not have inputs.
    std::vector<Tensor> dummy_args;

    StatusCallback callback = std::bind(
        [](Rendezvous* rendez, DoneCallback& done, const Status& status) {
          rendez->Unref();
          done();
        },
        rendez, std::move(done), std::placeholders::_1);
    auto* refcounted_done = new ReffedStatusCallback(std::move(callback));
    for (int i = 1; i < function_handles_.size(); ++i) {
      refcounted_done->Ref();
    }

    for (const auto& pair : function_handles_) {
      const string& target_device = pair.first;
      FHandle handle = pair.second;
      VLOG(3) << "Running function shard on device " << target_device;
      if (target_device == local_device_name_) {
        opts.remote_execution = false;
        std::vector<Tensor> args;
        args.reserve(arguments.size());
        for (const Tensor& argument : arguments) {
          args.push_back(argument);
        }
        auto* rets = new std::vector<Tensor>;
        lib->Run(opts, handle, args, rets,
                 [rets, refcounted_done, ctx](const Status& status) {
                   if (!status.ok()) {
                     ctx->SetStatus(status);
                   } else {
                     for (int i = 0; i < rets->size(); ++i) {
                       ctx->set_output(i, (*rets)[i]);
                     }
                   }
                   delete rets;
                   refcounted_done->Unref();
                 });
      } else {
        opts.remote_execution = true;
        std::vector<Tensor>* dummy_rets = new std::vector<Tensor>;
        lib->Run(opts, handle, dummy_args, dummy_rets,
                 [dummy_rets, refcounted_done, ctx](const Status& status) {
                   if (!status.ok()) {
                     ctx->SetStatus(status);
                   }
                   delete dummy_rets;
                   refcounted_done->Unref();
                 });
      }
    }
  }
  string UniquifyFunctionName(const string& name) {
    for (;; ++suffix_) {
      const string candidate = strings::StrCat(name, "_", suffix_);
      if (overlay_lib_->Find(candidate) == nullptr) {
        return candidate;
      }
    }
  }

  NameAttrList func_;
  const string local_device_name_;
  // Function shards are added to `overlay_lib_`.
  std::unique_ptr<FunctionLibraryDefinition> overlay_lib_;
  // A map from device names to handles of function shards; this map is
  // read-only after the first execution of the OpKernel.
  gtl::FlatMap<string, FHandle> function_handles_;

  mutex mu_;
  bool partitioned_ GUARDED_BY(mu_) = false;

  // Used to uniquify function names in `overlay_lib_`.
  uint32 suffix_ = 0;
};
REGISTER_KERNEL_BUILDER(Name("PartitionedCall").Device(DEVICE_CPU),
                        PartitionedCallOp);
REGISTER_KERNEL_BUILDER(Name("PartitionedCall").Device(DEVICE_GPU),
                        PartitionedCallOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("PartitionedCall").Device(DEVICE_SYCL),
                        PartitionedCallOp);
#endif  // TENSORFLOW_USE_SYCL

}  // namespace
}  // namespace tensorflow
