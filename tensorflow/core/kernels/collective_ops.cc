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
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

static string CollectiveKey(OpKernelContext* ctx, int32_t group_key,
                            int32_t instance_key) {
  return strings::StrCat(group_key, ":", instance_key, ":",
                         ctx->frame_iter().frame_id, ":",
                         ctx->frame_iter().iter_id);
}

static std::unique_ptr<OpKernel> BuildOpKernel(OpKernelConstruction* c,
                                               const string& name,
                                               NodeDef* sub_node) {
  std::unique_ptr<OpKernel> k;
  if (name.empty() || name == "Id") return k;
  sub_node->set_name(name);
  sub_node->set_op(name);
  Status status;
  k = CreateOpKernel(c->device_type(), c->device(),
                     c->device()->GetAllocator(AllocatorAttributes()),
                     *sub_node, c->graph_def_version(), &status);
  if (!status.ok()) {
    c->CtxFailureWithWarning(errors::Internal(
        "Failed to build OpKernel for ", name, " : ", status.error_message()));
  }
  return k;
}

class CollectiveOpV1Kernel : public AsyncOpKernel {
 public:
  explicit CollectiveOpV1Kernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), name_(name()), col_params_(new CollectiveParams()) {}

  ~CollectiveOpV1Kernel() override { col_params_->Unref(); }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            name_),
        done);
    const CancellationToken token =
        c->cancellation_manager()->get_cancellation_token();
    const bool already_cancelled =
        !c->cancellation_manager()->RegisterCallback(token, [col_exec]() {
          // We must call StartAbort() within the callback. StartAbort() relies
          // on resources that may be deallocated if all execution of a graph is
          // finished.
          col_exec->StartAbort(errors::Cancelled("op cancelled"));
        });
    OP_REQUIRES_ASYNC(c, !already_cancelled,
                      errors::Cancelled("op cancelled ", name_), done);

    auto deregister_and_done = [c, token, done = std::move(done)]() {
      // Once done() is called, StartAbort() won't have any effect, so we
      // don't need to block on the deregistration. Also StartAbort() may call
      // done() and DeregisterCallback may deadlock.
      c->cancellation_manager()->TryDeregisterCallback(token);
      done();
    };
    ComputeAsyncImpl(c, col_exec, std::move(deregister_and_done));
  }

  // A string encoding instance, frame and iter to be handed off to
  // the implementation for use in generating RecvBuf keys.
  string GetCollectiveKey(OpKernelContext* c) {
    return CollectiveKey(c, col_params_->group.group_key,
                         col_params_->instance.instance_key);
  }

  // Returns false if calling invocation of ComputeAsync should return
  // immediately.
  bool CanProceedWithCompute(OpKernelContext* c, CollectiveExecutor* col_exec,
                             const DoneCallback& done) {
    if (col_params_->group.group_size > col_params_->group.members.size()) {
      // This is the first invocation: Finish initializing col_params_.
      // Schedule the `CompleteParamsAsync` call on a work queue that can handle
      // blocking work because it's not guaranteed that this call cannot block.
      c->collective_executor()->RunClosure([this, c, col_exec, done]() {
        VLOG(1) << "CollectiveOpKernel CompleteParams for collective "
                << col_params_->name << " device " << c->device()->name()
                << " group " << col_params_->group.group_key << " instance "
                << col_params_->instance.instance_key;
        col_exec->CompleteParamsAsync(
            c->device()->attributes(), col_params_, c->cancellation_manager(),
            [this, c, done](const Status& s) {
              if (s.ok()) {
                col_params_->instance.impl_details.dependencies = dependencies_;
                ComputeAsync(c, done);
              } else {
                c->SetStatus(s);
                done();
              }
            });
      });
      return false;
    }
    return true;
  }

 protected:
  virtual void ComputeAsyncImpl(OpKernelContext* c,
                                CollectiveExecutor* col_exec,
                                DoneCallback done) = 0;

  string name_;
  CollectiveParams* col_params_;
  std::vector<int32> dependencies_;
};

class CollectiveGatherOpKernel : public CollectiveOpV1Kernel {
 public:
  explicit CollectiveGatherOpKernel(OpKernelConstruction* c)
      : CollectiveOpV1Kernel(c) {
    col_params_->instance.type = GATHER_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_->group.group_size));
    OP_REQUIRES(
        c, col_params_->group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_->group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_->group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_->instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_->instance.data_type));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_->instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_->instance.impl_details.timeout_seconds));
    const NodeDef& real_node = c->def();
    col_params_->name = strings::StrCat(real_node.name(), ": Gather");
    col_params_->group.device_type = c->device_type();
  }

 protected:
  void ComputeAsyncImpl(OpKernelContext* c, CollectiveExecutor* col_exec,
                        DoneCallback done) override {
    auto output_shape = c->input(0).shape();
    OP_REQUIRES_ASYNC(c, output_shape.dims() > 0,
                      errors::InvalidArgument("input should have rank > 0, ",
                                              "recieved ", output_shape.dims()),
                      done);
    output_shape.set_dim(
        0, output_shape.dim_size(0) * col_params_->group.group_size);
    col_params_->instance.shape = output_shape;

    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_output(0, col_params_->instance.shape, &output), done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, col_params = col_params_, done](const Status& s) {
      VLOG(1) << "CollectiveGatherOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance "
              << col_params->instance.instance_key << " status " << s;
      col_params->Unref();
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveGatherOpKernel ExecuteAsync start for collective "
            << col_params_->name << " device " << c->device()->name()
            << " group " << col_params_->group.group_key << " instance "
            << col_params_->instance.instance_key;
    col_params_->Ref();
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveGatherOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveGather").Device(DEVICE_CPU),
                        CollectiveGatherOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveGather").Device(DEVICE_GPU),
                        CollectiveGatherOpKernel);

class CollectiveReduceOpKernel : public CollectiveOpV1Kernel {
 public:
  explicit CollectiveReduceOpKernel(OpKernelConstruction* c)
      : CollectiveOpV1Kernel(c) {
    col_params_->instance.type = REDUCTION_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_->group.group_size));
    OP_REQUIRES(
        c, col_params_->group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_->group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_->group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_->instance.instance_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("subdiv_offsets",
                      &col_params_->instance.impl_details.subdiv_offsets));
    string merge_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("merge_op", &merge_op_name));
    if (merge_op_name == "Max") {
      merge_op_name = "Maximum";
    } else if (merge_op_name == "Min") {
      merge_op_name = "Minimum";
    }
    string final_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("final_op", &final_op_name));
    OP_REQUIRES(c, final_op_name == "Id" || final_op_name == "Div",
                errors::InvalidArgument(
                    "final_op must be one of {\"Id\", \"Div\"} but got ",
                    final_op_name));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_->instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("wait_for", &dependencies_));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_->instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_->instance.impl_details.timeout_seconds));
    VLOG(2) << "CollectiveReduce instance "
            << col_params_->instance.instance_key << " merge_op "
            << merge_op_name << " final_op " << final_op_name
            << " communication_hint "
            << col_params_->instance.impl_details.communication_hint
            << " timeout "
            << col_params_->instance.impl_details.timeout_seconds;

    const NodeDef& real_node = c->def();
    col_params_->name = strings::StrCat(real_node.name(), ": Reduce(",
                                        merge_op_name, ",", final_op_name, ")");
    col_params_->group.device_type = c->device_type();

    // Find the OpKernels by name, type and device type.
    NodeDef sub_node;
    // The merge_op takes two inputs
    sub_node.add_input(real_node.input(0));
    sub_node.add_input(real_node.input(0));
    sub_node.set_device(real_node.device());
    SetAttrValue(col_params_->instance.data_type,
                 &(*sub_node.mutable_attr())["T"]);
    merge_op_ = BuildOpKernel(c, merge_op_name, &sub_node);
    final_op_ = BuildOpKernel(c, final_op_name, &sub_node);
    col_params_->merge_op = merge_op_.get();
    col_params_->final_op = final_op_.get();
  }

 protected:
  void ComputeAsyncImpl(OpKernelContext* c, CollectiveExecutor* col_exec,
                        DoneCallback done) override {
    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor, trying to reuse the input.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(c,
                           c->forward_input_or_allocate_output(
                               {0}, 0, c->input(0).shape(), &output),
                           done);
      col_params_->instance.shape = c->input(0).shape();
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, col_params = col_params_, done](const Status& s) {
      VLOG(1) << "CollectiveReduceOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance "
              << col_params->instance.instance_key << " status " << s;
      col_params->Unref();
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveReduceOpKernel ExecuteAsync start for collective "
            << col_params_->name << " device " << c->device()->name()
            << " group " << col_params_->group.group_key << " instance "
            << col_params_->instance.instance_key;
    col_params_->Ref();
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  std::unique_ptr<OpKernel> merge_op_;
  std::unique_ptr<OpKernel> final_op_;
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveReduceOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveReduce").Device(DEVICE_CPU),
                        CollectiveReduceOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveReduce").Device(DEVICE_GPU),
                        CollectiveReduceOpKernel);

class CollectiveBcastSendOpKernel : public CollectiveOpV1Kernel {
 public:
  explicit CollectiveBcastSendOpKernel(OpKernelConstruction* c)
      : CollectiveOpV1Kernel(c) {
    col_params_->instance.type = BROADCAST_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_->group.group_size));
    OP_REQUIRES(
        c, col_params_->group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_->group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_->group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_->instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_->instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &col_params_->instance.shape));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_->instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_->instance.impl_details.timeout_seconds));
    col_params_->is_source = true;
    col_params_->instance.impl_details.subdiv_offsets = {0};

    col_params_->name =
        strings::StrCat(name(), ": Broadcast(", col_params_->is_source, ")");
    col_params_->group.device_type = c->device_type();
  }

 protected:
  void ComputeAsyncImpl(OpKernelContext* c, CollectiveExecutor* col_exec,
                        DoneCallback done) override {
    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor, trying to reuse the input.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(c,
                           c->forward_input_or_allocate_output(
                               {0}, 0, col_params_->instance.shape, &output),
                           done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;
    OP_REQUIRES_ASYNC(
        c, col_params_->instance.shape.IsSameSize(c->input(0).shape()),
        errors::Internal("Declared shape of op ", col_params_->name,
                         " does not match shape of input"),
        done);

    auto actual_done = [c, col_params = col_params_, done](const Status& s) {
      VLOG(1) << "CollectiveBcastSendOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance "
              << col_params->instance.instance_key << " status " << s;
      col_params->Unref();
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveBcastSendOpKernel ExecuteAsync start for collective "
            << col_params_->name << " device " << c->device()->name()
            << " group " << col_params_->group.group_key << " instance "
            << col_params_->instance.instance_key;
    col_params_->Ref();
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveBcastSendOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSend").Device(DEVICE_CPU),
                        CollectiveBcastSendOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSend").Device(DEVICE_DEFAULT),
                        CollectiveBcastSendOpKernel);

class CollectiveBcastRecvOpKernel : public CollectiveOpV1Kernel {
 public:
  explicit CollectiveBcastRecvOpKernel(OpKernelConstruction* c)
      : CollectiveOpV1Kernel(c) {
    col_params_->instance.type = BROADCAST_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_->group.group_size));
    OP_REQUIRES(
        c, col_params_->group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_->group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_->group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_->instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_->instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &col_params_->instance.shape));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_->instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_->instance.impl_details.timeout_seconds));
    col_params_->is_source = false;
    col_params_->instance.impl_details.subdiv_offsets = {0};

    col_params_->name =
        strings::StrCat(name(), ": Broadcast(", col_params_->is_source, ")");
    col_params_->group.device_type = c->device_type();
  }

 protected:
  void ComputeAsyncImpl(OpKernelContext* c, CollectiveExecutor* col_exec,
                        DoneCallback done) override {
    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // No input, so must allocate output.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_output(0, col_params_->instance.shape, &output), done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, col_params = col_params_, done](const Status& s) {
      VLOG(1) << "CollectiveBcastRecvOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance_key "
              << col_params->instance.instance_key << " status  " << s;
      col_params->Unref();
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveBcastRecvOpKernel ExecuteAsync start for collective "
            << col_params_->name << " device " << c->device()->name()
            << " group " << col_params_->group.group_key << " instance "
            << col_params_->instance.instance_key;
    col_params_->Ref();
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveBcastRecvOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecv").Device(DEVICE_CPU),
                        CollectiveBcastRecvOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecv").Device(DEVICE_DEFAULT),
                        CollectiveBcastRecvOpKernel);

class CollectiveAssignGroupV2OpKernel : public OpKernel {
 public:
  explicit CollectiveAssignGroupV2OpKernel(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& group_assignment = context->input(0);
    const Tensor& device_index = context->input(1);
    const Tensor& base_key = context->input(2);

    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(device_index.shape()),
        errors::InvalidArgument(
            "device_index must be a scalar, but received tensor of shape: ",
            device_index.shape().DebugString()));

    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(group_assignment.shape()),
        errors::InvalidArgument("group_assignment must be a 2-d Tensor, but "
                                "received tensor of shape: ",
                                group_assignment.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(base_key.shape()),
                errors::InvalidArgument(
                    "base_key must be a scalar, but received tensor of shape: ",
                    base_key.shape().DebugString()));

    Tensor* group_key = nullptr;
    Tensor* group_size = nullptr;
    AllocatorAttributes attr;
    attr.set_on_host(true);
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                     &group_size, attr));

    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &group_key, attr));

    OP_REQUIRES_OK(
        context,
        ComputeGroupKey(group_assignment, device_index.scalar<int32_t>()(),
                        base_key.scalar<int32_t>()(), group_size, group_key));
  }

 private:
  static Status ComputeGroupKey(const Tensor& group_assignment,
                                const int32_t device_index,
                                const int32_t base_key, Tensor* group_size,
                                Tensor* group_key) {
    group_size->flat<int32_t>()(0) = group_assignment.dim_size(1);

    for (int group_id = 0; group_id < group_assignment.dim_size(0);
         group_id++) {
      int32_t key = static_cast<int32_t>(static_cast<uint32_t>(base_key) +
                                         static_cast<uint32_t>(group_id));
      if (key == 0) {
        return errors::InvalidArgument(
            "Using the reserved group_key = 0 is not allowed: group_id = ",
            group_id, ", base_key = ", base_key);
      }
      for (int color = 0; color < group_assignment.dim_size(1); color++) {
        const auto index = group_assignment.matrix<int32>()(group_id, color);
        if (index < 0 || index >= group_assignment.shape().num_elements()) {
          return errors::InvalidArgument("Not all items in group_assignment ",
                                         group_assignment.DebugString(),
                                         " is within [0, number of devices)");
        }
        if (index == device_index) {
          group_key->flat<int32_t>()(0) = key;
          VLOG(2) << " group_assignment = " << group_assignment.DebugString()
                  << " device_index = " << index
                  << " group_key = " << group_key->DebugString()
                  << " group_size = " << group_size->DebugString();
          return Status::OK();
        }
      }
    }
    return errors::InvalidArgument("device_index ", device_index,
                                   " is not found in group_assignment ",
                                   group_assignment.DebugString());
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveAssignGroupV2").Device(DEVICE_CPU),
                        CollectiveAssignGroupV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveAssignGroupV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("device_index")
                            .HostMemory("group_assignment")
                            .HostMemory("base_key")
                            .HostMemory("group_size")
                            .HostMemory("group_key"),
                        CollectiveAssignGroupV2OpKernel);

class CollectiveOpV2Kernel : public AsyncOpKernel {
 public:
  explicit CollectiveOpV2Kernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), name_(name()), device_type_(DEVICE_DEFAULT) {
    OP_REQUIRES_OK(c, c->GetAttr("T", &data_type_));
    OP_REQUIRES_OK(c, c->GetAttr("communication_hint", &communication_hint_));
    OP_REQUIRES_OK(c, c->GetAttr("timeout_seconds", &timeout_seconds_));
    device_type_ = c->device_type();
  }

 protected:
  // Fills common parts of CollectiveParams according to the Op, *excluding
  // output_shape*. Kernels should further work on the CollectiveParams if they
  // need to set additional fields.
  Status FillCollectiveParams(CollectiveParams* col_params,
                              CollectiveType collective_type,
                              const Tensor& group_size, const Tensor& group_key,
                              const Tensor& instance_key) {
    if (group_size.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input group_size, got ",
          group_size.shape().DebugString());
    }
    if (group_key.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input group_key, got ",
          group_key.shape().DebugString());
    }
    if (instance_key.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input instance_key, got ",
          instance_key.shape().DebugString());
    }
    col_params->name = name_;
    col_params->group.device_type = device_type_;
    col_params->group.group_size = group_size.unaligned_flat<int32>()(0);
    if (col_params->group.group_size <= 0) {
      return errors::InvalidArgument(
          "group_size must be positive integer but got ",
          col_params->group.group_size);
    }
    col_params->group.group_key = group_key.unaligned_flat<int32>()(0);
    col_params->instance.type = collective_type;
    col_params->instance.instance_key = instance_key.unaligned_flat<int32>()(0);
    col_params->instance.data_type = data_type_;
    col_params->instance.impl_details.communication_hint = communication_hint_;
    col_params->instance.impl_details.timeout_seconds = timeout_seconds_;
    return Status::OK();
  }

  // Runs a collective. The output tensor must be allocated before calling this
  // method. col_params must live until done is called.
  void Run(OpKernelContext* c, CollectiveParams* col_params,
           DoneCallback done) {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            name_),
        done);
    // Resolve the collective params.
    // Schedule the `CompleteParamsAsync` call on a work queue that can handle
    // blocking work because it's not guaranteed that this call cannot block.
    c->collective_executor()->RunClosure([c, done = std::move(done), col_params,
                                          col_exec]() {
      VLOG(1) << "Collective CompleteParams for " << col_params->name
              << " device " << c->device()->name() << " group "
              << col_params->group.group_key << " instance "
              << col_params->instance.instance_key;
      col_exec->CompleteParamsAsync(
          c->device()->attributes(), col_params, c->cancellation_manager(),
          [c, done = std::move(done), col_params, col_exec](const Status& s) {
            if (s.ok()) {
              auto actual_done = [c, col_params,
                                  done = std::move(done)](const Status& s) {
                VLOG(1) << "Collective ExecuteAsync done for "
                        << col_params->name << " device " << c->device()->name()
                        << " group " << col_params->group.group_key
                        << " instance " << col_params->instance.instance_key
                        << " status " << s;
                if (!s.ok()) {
                  c->SetStatus(s);
                }
                done();
              };
              VLOG(1) << "Collective ExecuteAsync start for "
                      << col_params->name << " device " << c->device()->name()
                      << " group " << col_params->group.group_key
                      << " instance " << col_params->instance.instance_key;
              col_exec->ExecuteAsync(
                  c, col_params,
                  CollectiveKey(c, col_params->group.group_key,
                                col_params->instance.instance_key),
                  actual_done);
            } else {
              c->SetStatus(s);
              done();
            }
          });
    });
  }

 protected:
  string name_;
  DataType data_type_ = DT_INVALID;
  string communication_hint_;
  float timeout_seconds_ = 0;
  DeviceType device_type_;
};

class CollectiveReduceV2OpKernel : public CollectiveOpV2Kernel {
 public:
  explicit CollectiveReduceV2OpKernel(OpKernelConstruction* c)
      : CollectiveOpV2Kernel(c) {
    string merge_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("merge_op", &merge_op_name));
    if (merge_op_name == "Max") {
      merge_op_name = "Maximum";
    } else if (merge_op_name == "Min") {
      merge_op_name = "Minimum";
    }
    string final_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("final_op", &final_op_name));
    OP_REQUIRES_OK(
        c, c->GetAttr("max_subdivs_per_device", &max_subdivs_per_device_));
    // Prepare OpKernels for reduction and final operations.
    // The merge_op takes two inputs
    NodeDef sub_node;
    sub_node.add_input(c->def().input(0));
    sub_node.add_input(c->def().input(0));
    sub_node.set_device(c->def().device());
    SetAttrValue(data_type_, &(*sub_node.mutable_attr())["T"]);
    merge_op_ = BuildOpKernel(c, merge_op_name, &sub_node);
    final_op_ = BuildOpKernel(c, final_op_name, &sub_node);
    name_ = strings::StrCat(c->def().name(), ": ReduceV2(", merge_op_name, ",",
                            final_op_name, ")");
    VLOG(2) << "CollectiveReduceV2 " << this << " name " << name_
            << " communication_hint " << communication_hint_;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
      done();
      col_params->Unref();
    };
    OP_REQUIRES_OK_ASYNC(c,
                         FillCollectiveParams(col_params, REDUCTION_COLLECTIVE,
                                              /*group_size*/ c->input(1),
                                              /*group_key*/ c->input(2),
                                              /*instance_key*/ c->input(3)),
                         done_with_cleanup);
    col_params->instance.shape = c->input(0).shape();
    col_params->merge_op = merge_op_.get();
    col_params->final_op = final_op_.get();
    VLOG(1) << "CollectiveReduceV2 group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    // Allocate the output tensor, trying to reuse the input.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         c->forward_input_or_allocate_output(
                             {0}, 0, col_params->instance.shape, &output),
                         done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }

 private:
  int max_subdivs_per_device_;
  std::unique_ptr<OpKernel> merge_op_;
  std::unique_ptr<OpKernel> final_op_;
};

REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV2").Device(DEVICE_CPU),
                        CollectiveReduceV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("instance_key"),
                        CollectiveReduceV2OpKernel);

class CollectiveGatherV2OpKernel : public CollectiveOpV2Kernel {
 public:
  explicit CollectiveGatherV2OpKernel(OpKernelConstruction* c)
      : CollectiveOpV2Kernel(c) {
    name_ = strings::StrCat(c->def().name(), ": GatherV2");
    VLOG(2) << "CollectiveGatherV2 " << this << " name " << name_
            << " communication_hint " << communication_hint_;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
      done();
      col_params->Unref();
    };
    OP_REQUIRES_OK_ASYNC(c,
                         FillCollectiveParams(col_params, GATHER_COLLECTIVE,
                                              /*group_size*/ c->input(1),
                                              /*group_key*/ c->input(2),
                                              /*instance_key*/
                                              c->input(3)),
                         done_with_cleanup);
    auto output_shape = c->input(0).shape();
    output_shape.set_dim(
        0, output_shape.dim_size(0) * col_params->group.group_size);
    col_params->instance.shape = output_shape;
    VLOG(1) << "CollectiveGatherV2 group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_output(0, col_params->instance.shape, &output),
        done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveGatherV2").Device(DEVICE_CPU),
                        CollectiveGatherV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveGatherV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("instance_key"),
                        CollectiveGatherV2OpKernel);

class CollectiveBcastSendV2OpKernel : public CollectiveOpV2Kernel {
 public:
  explicit CollectiveBcastSendV2OpKernel(OpKernelConstruction* c)
      : CollectiveOpV2Kernel(c) {
    const bool is_source = true;
    name_ = strings::StrCat(name(), ": Broadcast(", is_source, ")");
  }

 protected:
  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
      done();
      col_params->Unref();
    };
    OP_REQUIRES_OK_ASYNC(c,
                         FillCollectiveParams(col_params, BROADCAST_COLLECTIVE,
                                              /*group_size*/ c->input(1),
                                              /*group_key*/ c->input(2),
                                              /*instance_key*/ c->input(3)),
                         done_with_cleanup);
    col_params->is_source = true;
    col_params->instance.shape = c->input(0).shape();
    // Add a default value for subdiv offsets, which is the same as the default
    // value in the V1 op's attribute.
    col_params->instance.impl_details.subdiv_offsets.push_back(0);
    VLOG(1) << "CollectiveBcastSendV2 group_size "
            << col_params->group.group_size << " group_key "
            << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    // Allocate the output tensor, trying to reuse the input.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         c->forward_input_or_allocate_output(
                             {0}, 0, col_params->instance.shape, &output),
                         done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSendV2").Device(DEVICE_CPU),
                        CollectiveBcastSendV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSendV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("instance_key"),
                        CollectiveBcastSendV2OpKernel);

class CollectiveBcastRecvV2OpKernel : public CollectiveOpV2Kernel {
 public:
  explicit CollectiveBcastRecvV2OpKernel(OpKernelConstruction* c)
      : CollectiveOpV2Kernel(c) {
    const bool is_source = false;
    name_ = strings::StrCat(name(), ": Broadcast(", is_source, ")");
  }

 protected:
  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
      done();
      col_params->Unref();
    };
    OP_REQUIRES_OK_ASYNC(c,
                         FillCollectiveParams(col_params, BROADCAST_COLLECTIVE,
                                              /*group_size*/ c->input(0),
                                              /*group_key*/ c->input(1),
                                              /*instance_key*/ c->input(2)),
                         done_with_cleanup);
    col_params->is_source = false;
    TensorShape output_shape;
    OP_REQUIRES_OK_ASYNC(c, tensor::MakeShape(c->input(3), &output_shape),
                         done_with_cleanup);
    col_params->instance.shape = output_shape;
    // Add a default value for subdiv offsets, which is the same as the default
    // value in the V1 op's attribute.
    col_params->instance.impl_details.subdiv_offsets.push_back(0);
    VLOG(1) << "CollectiveBcastRecvV2 group_size "
            << col_params->group.group_size << " group_key "
            << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_output(0, col_params->instance.shape, &output),
        done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecvV2").Device(DEVICE_CPU),
                        CollectiveBcastRecvV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecvV2")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("instance_key")
                            .HostMemory("shape"),
                        CollectiveBcastRecvV2OpKernel);

/*
 * Resource for holding group for CollectiveOps.
 * This resource is returned from CollectiveInitializeCommunicatorOpKernel
 * It generates next instance key for the group for each collective operation.
 */
class CollectiveGroupResource : public ResourceBase {
 public:
  CollectiveGroupResource(int32 group_key, int32 rank, int32 group_size,
                          string communication_hint, float timeout_seconds)
      : group_key_(group_key),
        rank_(rank),
        group_size_(group_size),
        communication_hint_(communication_hint),
        timeout_seconds_(timeout_seconds) {}

  std::string DebugString() const override {
    return absl::StrFormat(
        "Collective Group with group_key = %d, group_size = %d, rank = %d",
        group_key_, group_size_, rank_);
  }

  int get_next_instance_key() {
    return instance_key_.fetch_add(1, std::memory_order_relaxed);
  }

  int32 group_key() const { return group_key_; }

  int32 rank() const { return rank_; }

  int32 group_size() const { return group_size_; }

  string communication_hint() const { return communication_hint_; }

  float timeout_seconds() const { return timeout_seconds_; }

 private:
  int32 group_key_, rank_, group_size_;
  string communication_hint_;
  std::atomic<int> instance_key_{0};
  float timeout_seconds_ = 0;
};

class CollectiveInitializeCommunicatorOpKernel : public AsyncOpKernel {
 public:
  explicit CollectiveInitializeCommunicatorOpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), device_type_(DEVICE_DEFAULT) {
    OP_REQUIRES_OK(c, c->GetAttr("communication_hint", &communication_hint_));
    OP_REQUIRES_OK(c, c->GetAttr("timeout_seconds", &timeout_seconds_));
    device_type_ = c->device_type();
  }

  Status CheckInputs(Tensor group_size_t, Tensor group_key_t) {
    if (group_size_t.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input group_size. "
          "It shoulbe a scalar, got tensor with shape ",
          group_size_t.shape().DebugString());
    }
    if (group_key_t.dims() > 0) {
      return errors::InvalidArgument(
          "Unexpected dimensions on input group_key, got ",
          group_key_t.shape().DebugString());
    }

    auto group_size = group_size_t.unaligned_flat<int32>()(0);
    if (group_size <= 0) {
      return errors::InvalidArgument(
          "group_size must be positive integer but got ", group_size);
    }
    return Status::OK();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto group_key_t = c->input(0);
    auto rank_t = c->input(1);
    auto group_size_t = c->input(2);

    OP_REQUIRES_OK_ASYNC(c, CheckInputs(group_size_t, group_key_t), done);

    auto group_size = group_size_t.unaligned_flat<int32>()(0);
    auto group_key = group_key_t.unaligned_flat<int32>()(0);
    auto rank = rank_t.unaligned_flat<int32>()(0);

    ResourceHandle resource_handle =
        MakeResourceHandle<CollectiveGroupResource>(
            c, "collective_op_group",
            absl::StrFormat("%d:r%04d", group_key, rank));

    Tensor* output_handle = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_output(0, TensorShape({}), &output_handle), done);
    output_handle->scalar<ResourceHandle>()() = resource_handle;

    CollectiveGroupResource* resource = new CollectiveGroupResource(
        group_key, rank, group_size, this->communication_hint_,
        this->timeout_seconds_);
    OP_REQUIRES_OK_ASYNC(
        c,
        CreateResource<CollectiveGroupResource>(c, resource_handle, resource),
        done);
    auto group_params = new CollGroupParams();
    group_params->device_type = device_type_;
    group_params->group_size = resource->group_size();
    group_params->group_key = resource->group_key();
    group_params->user_specified_rank = resource->rank();

    auto* col_exec = c->collective_executor();

    c->collective_executor()->RunClosure([c, done = std::move(done),
                                          group_params, col_exec]() {
      VLOG(1) << "Collective Group initialization for "
              << " device " << c->device()->name() << " group "
              << group_params->group_key;
      col_exec->CompleteGroupAsync(
          c->device()->attributes(), group_params, c->cancellation_manager(),
          [c, done = std::move(done), group_params](const Status& s) {
            if (s.ok()) {
              VLOG(1) << "Collective Group initialization done for device "
                      << c->device()->name() << " group "
                      << group_params->group_key << " status " << s;
            } else {
              c->SetStatus(s);
            }
            delete group_params;
            done();
          });
    });
  }

 private:
  string communication_hint_;
  DeviceType device_type_;
  float timeout_seconds_ = 0;
};

REGISTER_KERNEL_BUILDER(
    Name("CollectiveInitializeCommunicator").Device(DEVICE_CPU),
    CollectiveInitializeCommunicatorOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveInitializeCommunicator")
                            .Device(DEVICE_GPU)
                            .HostMemory("group_size")
                            .HostMemory("group_key")
                            .HostMemory("rank"),
                        CollectiveInitializeCommunicatorOpKernel);

class CollectiveOpV3Kernel : public AsyncOpKernel {
 public:
  explicit CollectiveOpV3Kernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), name_(name()), device_type_(DEVICE_DEFAULT) {
    OP_REQUIRES_OK(c, c->GetAttr("T", &data_type_));
    if (c->HasAttr("timeout_seconds")) {
      OP_REQUIRES_OK(c, c->GetAttr("timeout_seconds", &timeout_seconds_));
    } else {
      timeout_seconds_ = -1;
    }
    device_type_ = c->device_type();
  }

 protected:
  // Fills common parts of CollectiveParams according to the Op, *excluding
  // output_shape*. Kernels should further work on the CollectiveParams if they
  // need to set additional fields.
  Status FillCollectiveParams(CollectiveParams* col_params,
                              const Tensor& group_assignment,
                              CollectiveType collective_type,
                              CollectiveGroupResource* resource) {
    int64 group_id;
    int64 group_size;
    if (group_assignment.NumElements() == 0) {
      // No group assignments, perform collective as a single group.
      group_id = 0;
      group_size = resource->group_size();
    } else {
      return errors::Unimplemented("Group assignments are not supported yet.");
    }

    // Construct instance key with format:
    // <11 bits for group><21 bits for atomic incremented instance key>
    int32 instance_key = group_id << 21 | resource->get_next_instance_key();
    col_params->name = name_;
    col_params->group.device_type = device_type_;
    col_params->group.group_size = group_size;
    col_params->group.group_key = resource->group_key();
    col_params->group.user_specified_rank = resource->rank();
    col_params->instance.type = collective_type;
    col_params->instance.instance_key = instance_key;
    col_params->instance.data_type = data_type_;
    col_params->instance.impl_details.communication_hint =
        resource->communication_hint();
    col_params->instance.impl_details.timeout_seconds =
        timeout_seconds_ > 0 ? resource->timeout_seconds() : timeout_seconds_;
    col_params->run_group_initialization = false;
    return Status::OK();
  }

  // Runs a collective. The output tensor must be allocated before calling this
  // method. col_params must live until done is called.
  void Run(OpKernelContext* c, CollectiveParams* col_params,
           DoneCallback done) {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            name_),
        done);
    // Resolve the collective params.
    // Schedule the `CompleteParamsAsync` call on a work queue that can handle
    // blocking work because it's not guaranteed that this call cannot block.
    col_exec->RunClosure([c, done = std::move(done), col_params, col_exec]() {
      VLOG(1) << "Collective CompleteParams for " << col_params->name
              << " device " << c->device()->name() << " group "
              << col_params->group.group_key << " instance "
              << col_params->instance.instance_key;
      col_exec->CompleteParamsAsync(
          c->device()->attributes(), col_params, c->cancellation_manager(),
          [c, done = std::move(done), col_params, col_exec](const Status& s) {
            if (s.ok()) {
              auto actual_done = [c, col_params,
                                  done = std::move(done)](const Status& s) {
                VLOG(1) << "Collective ExecuteAsync done for "
                        << col_params->name << " device " << c->device()->name()
                        << " group " << col_params->group.group_key
                        << " instance " << col_params->instance.instance_key
                        << " status " << s;
                if (!s.ok()) {
                  c->SetStatus(s);
                }
                done();
              };
              VLOG(1) << "Collective ExecuteAsync start for "
                      << col_params->name << " device " << c->device()->name()
                      << " group " << col_params->group.group_key
                      << " instance " << col_params->instance.instance_key;
              col_exec->ExecuteAsync(
                  c, col_params,
                  CollectiveKey(c, col_params->group.group_key,
                                col_params->instance.instance_key),
                  actual_done);
            } else {
              c->SetStatus(s);
              done();
            }
          });
    });
  }

 protected:
  string name_;
  DataType data_type_ = DT_INVALID;
  DeviceType device_type_;
  float timeout_seconds_ = 0;
};

class CollectiveReduceV3OpKernel : public CollectiveOpV3Kernel {
 public:
  explicit CollectiveReduceV3OpKernel(OpKernelConstruction* c)
      : CollectiveOpV3Kernel(c) {
    string reduction;
    OP_REQUIRES_OK(c, c->GetAttr("reduction", &reduction));
    if (reduction == "Max") {
      reduction = "Maximum";
    } else if (reduction == "Min") {
      reduction = "Minimum";
    }
    // Prepare OpKernels for reduction and final operations.
    // The merge_op takes two inputs
    NodeDef sub_node;
    sub_node.add_input(c->def().input(0));
    sub_node.add_input(c->def().input(0));
    sub_node.set_device(c->def().device());
    SetAttrValue(data_type_, &(*sub_node.mutable_attr())["T"]);
    merge_op_ = BuildOpKernel(c, reduction, &sub_node);
    final_op_ = BuildOpKernel(c, "Id", &sub_node);
    name_ = strings::StrCat(c->def().name(), ": ReduceV3(", reduction, ")");
    VLOG(2) << "CollectiveReduceV3 " << this << " name " << name_;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
      done();
      col_params->Unref();
    };
    core::RefCountPtr<CollectiveGroupResource> resource;
    OP_REQUIRES_OK_ASYNC(c, LookupResource(c, HandleFromInput(c, 1), &resource),
                         done_with_cleanup);

    Tensor group_assignment = c->input(2);

    OP_REQUIRES_OK_ASYNC(
        c,
        FillCollectiveParams(col_params, group_assignment, REDUCTION_COLLECTIVE,
                             resource.get()),
        done_with_cleanup);
    col_params->instance.shape = c->input(0).shape();
    col_params->merge_op = merge_op_.get();
    col_params->final_op = final_op_.get();
    VLOG(1) << "CollectiveReduceV3 group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    // Allocate the output tensor, trying to reuse the input.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         c->forward_input_or_allocate_output(
                             {0}, 0, col_params->instance.shape, &output),
                         done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }

 private:
  std::unique_ptr<OpKernel> merge_op_;
  std::unique_ptr<OpKernel> final_op_;
};

REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV3").Device(DEVICE_CPU),
                        CollectiveReduceV3OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV3").Device(DEVICE_GPU),
                        CollectiveReduceV3OpKernel);

class CollectiveAllToAllV3OpKernel : public CollectiveOpV3Kernel {
 public:
  explicit CollectiveAllToAllV3OpKernel(OpKernelConstruction* c)
      : CollectiveOpV3Kernel(c) {
    name_ = strings::StrCat(c->def().name(), ": AllToAllV3");
    VLOG(2) << "CollectiveAllToAllV3 " << this << " name " << name_;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto col_params = new CollectiveParams();
    auto done_with_cleanup = [col_params, done = std::move(done)]() {
      done();
      col_params->Unref();
    };
    core::RefCountPtr<CollectiveGroupResource> resource;
    OP_REQUIRES_OK_ASYNC(c, LookupResource(c, HandleFromInput(c, 1), &resource),
                         done_with_cleanup);

    Tensor group_assignment = c->input(2);

    OP_REQUIRES_OK_ASYNC(
        c,
        FillCollectiveParams(col_params, group_assignment,
                             ALL_TO_ALL_COLLECTIVE, resource.get()),
        done_with_cleanup);
    col_params->instance.shape = c->input(0).shape();
    VLOG(1) << "CollectiveAllToAll group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;
    // Allocate the output tensor, trying to reuse the input.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c,
                         c->forward_input_or_allocate_output(
                             {0}, 0, col_params->instance.shape, &output),
                         done_with_cleanup);
    Run(c, col_params, std::move(done_with_cleanup));
  }
};

REGISTER_KERNEL_BUILDER(Name("CollectiveAllToAllV3").Device(DEVICE_CPU),
                        CollectiveAllToAllV3OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveAllToAllV3").Device(DEVICE_GPU),
                        CollectiveAllToAllV3OpKernel);
}  // namespace
}  // namespace tensorflow
