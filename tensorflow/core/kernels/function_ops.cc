/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/function_ops.h"

#include <deque>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/memory_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/gradients.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

static const char* const kGradientOp = FunctionLibraryDefinition::kGradientOp;

ArgOp::ArgOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
}

void ArgOp::Compute(OpKernelContext* ctx) {
  auto frame = ctx->call_frame();
  OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
  Tensor val;
  OP_REQUIRES_OK(ctx, frame->GetArg(index_, &val));
  OP_REQUIRES(ctx, val.dtype() == dtype_,
              errors::InvalidArgument("Type mismatch: actual ",
                                      DataTypeString(val.dtype()),
                                      " vs. expect ", DataTypeString(dtype_)));
  ctx->set_output(0, val);
}

RetvalOp::RetvalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
}

void RetvalOp::Compute(OpKernelContext* ctx) {
  const Tensor& val = ctx->input(0);
  OP_REQUIRES(ctx, val.dtype() == dtype_,
              errors::InvalidArgument("Type mismatch: actual ",
                                      DataTypeString(val.dtype()),
                                      " vs. expect ", DataTypeString(dtype_)));
  auto frame = ctx->call_frame();
  OP_REQUIRES(ctx, frame != nullptr, errors::Internal("no call frame"));
  OP_REQUIRES_OK(ctx, frame->SetRetval(index_, val));
}

REGISTER_SYSTEM_KERNEL_BUILDER(Name(kArgOp).Device(DEVICE_CPU), ArgOp);
REGISTER_SYSTEM_KERNEL_BUILDER(Name(kDeviceArgOp).Device(DEVICE_CPU), ArgOp);
REGISTER_SYSTEM_KERNEL_BUILDER(Name(kRetOp).Device(DEVICE_CPU), RetvalOp);
REGISTER_SYSTEM_KERNEL_BUILDER(Name(kDeviceRetOp).Device(DEVICE_CPU), RetvalOp);

#if TENSORFLOW_USE_SYCL
#define REGISTER(type)     \
  REGISTER_KERNEL_BUILDER( \
      Name(kArgOp).Device(DEVICE_SYCL).TypeConstraint<type>("T"), ArgOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER)
TF_CALL_bool(REGISTER) REGISTER_KERNEL_BUILDER(Name(kArgOp)
                                                   .Device(DEVICE_SYCL)
                                                   .HostMemory("output")
                                                   .TypeConstraint<int32>("T"),
                                               ArgOp);
#undef REGISTER
#define REGISTER(type)     \
  REGISTER_KERNEL_BUILDER( \
      Name(kRetOp).Device(DEVICE_SYCL).TypeConstraint<type>("T"), RetvalOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER)
TF_CALL_bool(REGISTER) REGISTER_KERNEL_BUILDER(Name(kRetOp)
                                                   .Device(DEVICE_SYCL)
                                                   .HostMemory("input")
                                                   .TypeConstraint<int32>("T"),
                                               RetvalOp);
#undef REGISTER
#endif

#define REGISTER(type)     \
  REGISTER_KERNEL_BUILDER( \
      Name(kArgOp).Device(DEVICE_GPU).TypeConstraint<type>("T"), ArgOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER)
TF_CALL_QUANTIZED_TYPES(REGISTER)
TF_CALL_bool(REGISTER) REGISTER_KERNEL_BUILDER(Name(kArgOp)
                                                   .Device(DEVICE_GPU)
                                                   .HostMemory("output")
                                                   .TypeConstraint<int32>("T"),
                                               ArgOp);
REGISTER_KERNEL_BUILDER(
    Name(kDeviceArgOp).Device(DEVICE_GPU).TypeConstraint<int32>("T"), ArgOp);
#undef REGISTER

REGISTER_KERNEL_BUILDER(Name(kArgOp)
                            .Device(DEVICE_GPU)
                            .HostMemory("output")
                            .TypeConstraint<ResourceHandle>("T"),
                        ArgOp);

REGISTER_KERNEL_BUILDER(Name(kArgOp)
                            .Device(DEVICE_GPU)
                            .HostMemory("output")
                            .TypeConstraint<tstring>("T"),
                        ArgOp);

REGISTER_KERNEL_BUILDER(
    Name(kArgOp).Device(DEVICE_GPU).TypeConstraint<Variant>("T"), ArgOp);

#define REGISTER(type)     \
  REGISTER_KERNEL_BUILDER( \
      Name(kRetOp).Device(DEVICE_GPU).TypeConstraint<type>("T"), RetvalOp);
TF_CALL_NUMBER_TYPES_NO_INT32(REGISTER)
TF_CALL_QUANTIZED_TYPES(REGISTER)
REGISTER(Variant)
TF_CALL_bool(REGISTER) REGISTER_KERNEL_BUILDER(Name(kRetOp)
                                                   .Device(DEVICE_GPU)
                                                   .HostMemory("input")
                                                   .TypeConstraint<int32>("T"),
                                               RetvalOp);
REGISTER_KERNEL_BUILDER(
    Name(kDeviceRetOp).Device(DEVICE_GPU).TypeConstraint<int32>("T"), RetvalOp);

REGISTER_KERNEL_BUILDER(Name(kRetOp)
                            .Device(DEVICE_GPU)
                            .TypeConstraint<ResourceHandle>("T")
                            .HostMemory("input"),
                        RetvalOp);

REGISTER_KERNEL_BUILDER(Name(kRetOp)
                            .Device(DEVICE_GPU)
                            .TypeConstraint<tstring>("T")
                            .HostMemory("input"),
                        RetvalOp);
#undef REGISTER

class PassOn : public OpKernel {
 public:
  explicit PassOn(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES(ctx, ctx->num_inputs() == ctx->num_outputs(),
                errors::Internal("#inputs != #outputs : ", ctx->num_inputs(),
                                 " vs. ", ctx->num_outputs()));
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      OP_REQUIRES(
          ctx, input_type(i) == output_type(i),
          errors::Internal("Input and output types for position ", i,
                           " do not match: ", DataTypeString(input_type(i)),
                           " vs. ", DataTypeString(output_type(i))));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      ctx->set_output(i, ctx->input(i));
    }
  }
};

REGISTER_SYSTEM_KERNEL_BUILDER(Name("_ListToArray").Device(DEVICE_CPU), PassOn);
REGISTER_SYSTEM_KERNEL_BUILDER(Name("_ArrayToList").Device(DEVICE_CPU), PassOn);

#define REGISTER_GPU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_ListToArray").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PassOn);                                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_ArrayToList").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      PassOn);

REGISTER_GPU_KERNELS(Eigen::half);
REGISTER_GPU_KERNELS(float);
REGISTER_GPU_KERNELS(double);

#undef REGISTER_GPU_KERNELS

REGISTER_KERNEL_BUILDER(Name("_ListToArray")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);
REGISTER_KERNEL_BUILDER(Name("_ArrayToList")
                            .Device(DEVICE_GPU)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ListToArray").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      PassOn);                                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_ArrayToList").Device(DEVICE_SYCL).TypeConstraint<type>("T"), \
      PassOn);

REGISTER_SYCL_KERNELS(float);
REGISTER_SYCL_KERNELS(double);

#undef REGISTER_SYCL_KERNELS

REGISTER_KERNEL_BUILDER(Name("_ListToArray")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);
REGISTER_KERNEL_BUILDER(Name("_ArrayToList")
                            .Device(DEVICE_SYCL)
                            .HostMemory("input")
                            .HostMemory("output")
                            .TypeConstraint<int32>("T"),
                        PassOn);
#endif  // TENSORFLOW_USE_SYCL

class SymbolicGradientOp : public AsyncOpKernel {
 public:
  explicit SymbolicGradientOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {}

  ~SymbolicGradientOp() override {}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    FunctionLibraryRuntime* lib = ctx->function_library();
    OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                      errors::Internal("No function library is provided."),
                      done);

    FunctionLibraryRuntime::Handle handle;
    OP_REQUIRES_OK_ASYNC(
        ctx, lib->Instantiate(kGradientOp, AttrSlice(def()), &handle), done);

    FunctionLibraryRuntime::Options opts;
    opts.rendezvous = ctx->rendezvous();
    opts.cancellation_manager = ctx->cancellation_manager();
    opts.runner = ctx->runner();
    opts.stats_collector = ctx->stats_collector();
    opts.step_container = ctx->step_container();
    opts.collective_executor = ctx->collective_executor();
    std::vector<Tensor> args;
    args.reserve(ctx->num_inputs());
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      args.push_back(ctx->input(i));
    }
    std::vector<Tensor>* rets = new std::vector<Tensor>;
    profiler::TraceMe trace_me(
        [&] {
          return absl::StrCat(
              "SymbolicGradientOp #parent_step_id=", ctx->step_id(),
              ",function_step_id=", opts.step_id, "#");
        },
        /*level=*/2);
    lib->Run(opts, handle, args, rets, [ctx, done, rets](const Status& status) {
      if (!status.ok()) {
        ctx->SetStatus(status);
      } else if (rets->size() != ctx->num_outputs()) {
        ctx->SetStatus(errors::InvalidArgument(
            "SymGrad expects to return ", ctx->num_outputs(),
            " tensor(s), but get ", rets->size(), " tensor(s) instead."));
      } else {
        for (size_t i = 0; i < rets->size(); ++i) {
          ctx->set_output(i, (*rets)[i]);
        }
      }
      delete rets;
      done();
    });
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(SymbolicGradientOp);
};

REGISTER_KERNEL_BUILDER(Name(kGradientOp).Device(DEVICE_CPU),
                        SymbolicGradientOp);
REGISTER_KERNEL_BUILDER(Name(kGradientOp).Device(DEVICE_GPU),
                        SymbolicGradientOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name(kGradientOp).Device(DEVICE_SYCL),
                        SymbolicGradientOp);

#endif  // TENSORFLOW_USE_SYCL

RemoteCallOp::RemoteCallOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(FunctionLibraryDefinition::kFuncAttr, &func_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_dtypes_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_dtypes_));
}

void RemoteCallOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  FunctionLibraryRuntime* lib = ctx->function_library();
  OP_REQUIRES_ASYNC(ctx, lib != nullptr,
                    errors::Internal("No function library is provided."), done);

  const string& source_device = lib->device()->name();
  const Tensor* target;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input("target", &target), done);
  string target_device;
  OP_REQUIRES_OK_ASYNC(
      ctx,
      DeviceNameUtils::CanonicalizeDeviceName(target->scalar<tstring>()(),
                                              source_device, &target_device),
      done);

  AttrValueMap attr_values = func_.attr();

  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;

  const auto* config = (ctx->function_library())
                           ? ctx->function_library()->config_proto()
                           : nullptr;
  if (config) {
    instantiate_opts.config_proto = *config;
  }

  instantiate_opts.target = target_device;

  FunctionTarget function_target = {target_device, lib};

  FunctionLibraryRuntime::Handle handle;
  {
    mutex_lock l(mu_);
    auto cached_entry = handle_cache_.find(function_target);
    if (cached_entry != handle_cache_.end()) {
      handle = cached_entry->second;
    } else {
      VLOG(1) << "Instantiating " << func_.name() << " on " << target_device;
      profiler::TraceMe activity(
          [&] {
            return strings::StrCat("RemoteCall: Instantiate: ", func_.name(),
                                   " on ", target_device);
          },
          profiler::TraceMeLevel::kInfo);
      OP_REQUIRES_OK_ASYNC(
          ctx,
          lib->Instantiate(func_.name(), AttrSlice(&attr_values),
                           instantiate_opts, &handle),
          done);
      auto insert_result = handle_cache_.insert({function_target, handle});
      CHECK(insert_result.second) << "Insert unsuccessful.";
      VLOG(1) << "Instantiated " << func_.name() << " on " << target_device
              << ", resulting in handle: " << handle << " flr: " << lib;
    }
  }

  OpInputList arguments;
  OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("args", &arguments), done);

  FunctionLibraryRuntime::Options opts;
  opts.runner = ctx->runner();
  opts.source_device = source_device;
  if (opts.source_device != target_device) {
    opts.remote_execution = true;
  }
  opts.create_rendezvous = true;
  std::vector<Tensor> args;
  args.reserve(arguments.size());
  for (const Tensor& argument : arguments) {
    args.push_back(argument);
  }
  for (const auto& dtype : input_dtypes_) {
    AllocatorAttributes arg_alloc_attrs;
    if (DataTypeAlwaysOnHost(dtype)) {
      arg_alloc_attrs.set_on_host(true);
    }
    opts.args_alloc_attrs.push_back(arg_alloc_attrs);
  }
  for (const auto& dtype : output_dtypes_) {
    AllocatorAttributes ret_alloc_attrs;
    if (DataTypeAlwaysOnHost(dtype)) {
      ret_alloc_attrs.set_on_host(true);
    }
    opts.rets_alloc_attrs.push_back(ret_alloc_attrs);
  }
  auto* rets = new std::vector<Tensor>;
  auto* activity = new profiler::TraceMe(
      [&] {
        return strings::StrCat("RemoteCall: Run: ", func_.name(), " on ",
                               target_device);
      },
      profiler::TraceMeLevel::kInfo);
  VLOG(1) << "Running " << func_.name() << " on " << target_device
          << " with handle: " << handle;
  profiler::TraceMe trace_me(
      [&] {
        return absl::StrCat("RemoteCallOp #parent_step_id=", ctx->step_id(),
                            ",function_step_id=", opts.step_id, "#");
      },
      /*level=*/2);
  lib->Run(opts, handle, args, rets,
           [rets, activity, done, ctx](const Status& status) {
             if (!status.ok()) {
               ctx->SetStatus(status);
             } else {
               for (size_t i = 0; i < rets->size(); ++i) {
                 ctx->set_output(i, (*rets)[i]);
               }
             }
             delete rets;
             delete activity;
             done();
           });
}

REGISTER_KERNEL_BUILDER(
    Name("RemoteCall").Device(DEVICE_CPU).HostMemory("target"), RemoteCallOp);
REGISTER_KERNEL_BUILDER(
    Name("RemoteCall").Device(DEVICE_GPU).HostMemory("target"), RemoteCallOp);
#if TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("RemoteCall").Device(DEVICE_SYCL).HostMemory("target"), RemoteCallOp);

#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
