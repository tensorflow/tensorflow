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
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/util/ptr_util.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/stream.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
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
    string deprecated_config_serialized;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config", &deprecated_config_serialized));
    string config_proto_serialized;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("config_proto", &config_proto_serialized));
    OP_REQUIRES(
        ctx,
        deprecated_config_serialized.empty() || config_proto_serialized.empty(),
        errors::InvalidArgument("Provided both 'config' and 'config_proto' but "
                                "only one should be provided.  Note the "
                                "'config' option is deprecated."));
    if (!deprecated_config_serialized.empty()) {
      OP_REQUIRES(ctx,
                  config_proto_.mutable_graph_options()
                      ->mutable_rewrite_options()
                      ->ParseFromString(deprecated_config_serialized),
                  errors::InvalidArgument("Unable to parse config string as "
                                          "tensorflow::RewriteOptions proto."));
    } else {
      OP_REQUIRES(
          ctx, config_proto_.ParseFromString(config_proto_serialized),
          errors::InvalidArgument("Unable to parse config_proto string as "
                                  "tensorflow::ConfigProto proto."));
    }
    OP_REQUIRES_OK(ctx, ctx->GetAttr("executor_type", &executor_type_));
  }

  ~PartitionedCallOp() override {
    for (const auto& it : handles_) {
      Status status = it.first->ReleaseHandle(it.second);
      if (!status.ok()) {
        LOG(INFO) << "Ignoring error while destructing PartitionedCallOp: "
                  << status.ToString();
      }
    }
  }

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
    // via, e.g., virtual device annotations and a list of device names
    // supplied through an attribute.
    //
    // TODO(akshayka): Add a fastpath for functions that execute on a single
    // device.
    FunctionLibraryRuntime::Handle handle;
    // If we are instantiating the function, we can efficiently extract the
    // inputs while instantiating. Else, we extract them separately below.
    std::vector<Tensor> inputs;
    bool inputs_extracted = false;
    {
      mutex_lock l(mu_);
      auto it = handles_.find(lib);
      if (it == handles_.end()) {
        OP_REQUIRES_OK_ASYNC(ctx, Instantiate(lib, ctx, &inputs, &handle),
                             done);
        inputs_extracted = true;
        handles_[lib] = handle;
      } else {
        handle = it->second;
      }
    }

    if (!inputs_extracted) {
      OpInputList args;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("args", &args), done);
      inputs.reserve(args.size());
      for (const Tensor& tensor : args) {
        inputs.push_back(tensor);
      }
    }

    RunFunction(handle, inputs, lib, ctx, done);
  }

 private:
  Status FillOutputDevices(const FunctionLibraryRuntime& lib,
                           const Device& cpu_device, AttrSlice attrs,
                           FunctionLibraryRuntime::InstantiateOptions* opts) {
    const FunctionLibraryDefinition* flib = lib.GetFunctionLibraryDefinition();
    const FunctionDef* fdef = flib->Find(func_.name());
    if (fdef == nullptr) {
      return errors::NotFound("Failed for find definiton for function \"",
                              func_.name(), "\"");
    }

    bool is_type_list;
    for (const OpDef::ArgDef& ret_def : fdef->signature().output_arg()) {
      DataTypeVector dtypes;
      TF_RETURN_IF_ERROR(ArgNumType(attrs, ret_def, &is_type_list, &dtypes));
      for (DataType dtype : dtypes) {
        if (MTypeFromDType(dtype) == HOST_MEMORY) {
          opts->output_devices.push_back(cpu_device.name());
        } else {
          opts->output_devices.push_back(opts->target);
        }
      }
    }
    return Status::OK();
  }

  Status Instantiate(FunctionLibraryRuntime* lib, OpKernelContext* ctx,
                     std::vector<Tensor>* inputs,
                     FunctionLibraryRuntime::Handle* handle) {
    FunctionLibraryRuntime::InstantiateOptions opts;
    opts.target = lib->device()->name();
    opts.is_multi_device_function = true;
    opts.optimize_graph_fn =
        std::bind(grappler::OptimizeGraph, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3,
                  std::placeholders::_4, config_proto_, std::placeholders::_5);
    opts.graph_collector = ctx->graph_collector();
    opts.executor_type = executor_type_;

    OpInputList args;
    TF_RETURN_IF_ERROR(ctx->input_list("args", &args));
    Device* cpu_device;
    TF_RETURN_IF_ERROR(lib->device_mgr()->LookupDevice("CPU:0", &cpu_device));

    inputs->reserve(args.size());
    for (const Tensor& tensor : args) {
      inputs->push_back(tensor);
      DataType dtype = tensor.dtype();
      if (dtype == DT_RESOURCE) {
        const ResourceHandle& handle = tensor.flat<ResourceHandle>()(0);
        opts.input_devices.push_back(handle.device());
      } else if (MTypeFromDType(dtype) == HOST_MEMORY) {
        opts.input_devices.push_back(cpu_device->name());
      } else {
        opts.input_devices.push_back(opts.target);
      }
    }

    TF_RETURN_IF_ERROR(
        FillOutputDevices(*lib, *cpu_device, AttrSlice(&func_.attr()), &opts));

    TF_RETURN_IF_ERROR(
        lib->Instantiate(func_.name(), AttrSlice(&func_.attr()), opts, handle));
    return Status::OK();
  }

  void RunFunction(FunctionLibraryRuntime::Handle handle,
                   const std::vector<Tensor>& inputs,
                   FunctionLibraryRuntime* lib, OpKernelContext* ctx,
                   DoneCallback done) {
    FunctionLibraryRuntime::Options run_opts;
    run_opts.step_id = ctx->step_id();
    run_opts.step_container = ctx->step_container();
    run_opts.cancellation_manager = ctx->cancellation_manager();
    run_opts.stats_collector = ctx->stats_collector();
    run_opts.collective_executor = ctx->collective_executor();
    // TODO(akshayka): Consider selecting a runner on a per-device basis,
    // i.e., using device-specific threadpools when available.
    run_opts.runner = ctx->runner();
    run_opts.source_device = lib->device()->name();
    run_opts.allow_dead_tensors = true;
    // TODO(akshayka): Accommodate the multiple-worker scenario by adding the
    // constructed rendezvous to a rendezvous manager.
    Rendezvous* rendez = new IntraProcessRendezvous(lib->device_mgr());
    run_opts.rendezvous = rendez;

    std::vector<Tensor>* rets = new std::vector<Tensor>;
    lib->Run(run_opts, handle, inputs, rets,
             [rets, rendez, done, ctx](const Status& status) {
               if (!status.ok()) {
                 ctx->SetStatus(status);
               } else {
                 for (int i = 0; i < rets->size(); ++i) {
                   ctx->set_output(i, (*rets)[i]);
                 }
               }
               delete rets;
               rendez->Unref();
               done();
             });
  }

  NameAttrList func_;
  ConfigProto config_proto_;
  string executor_type_;
  mutex mu_;
  // Cache the handle per FLR because this kernel may be instantiated for
  // a stateful op, different invocations of it may use different FLRs.
  // Different device placements of PartitionedCallOp also use
  // different FLRs.
  gtl::FlatMap<FunctionLibraryRuntime*, FunctionLibraryRuntime::Handle> handles_
      GUARDED_BY(mu_);
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
