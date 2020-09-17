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
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {

static string CollectiveKey(OpKernelContext* ctx, int32 group_key,
                            int32 instance_key) {
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

class CollectiveOpKernel : public AsyncOpKernel {
 public:
  explicit CollectiveOpKernel(OpKernelConstruction* c) : AsyncOpKernel(c) {}

  // A string encoding instance, frame and iter to be handed off to
  // the implementation for use in generating RecvBuf keys.
  string GetCollectiveKey(OpKernelContext* c) {
    return CollectiveKey(c, col_params_.group.group_key,
                         col_params_.instance.instance_key);
  }

  // Returns false if calling invocation of ComputeAsync should return
  // immediately.
  bool CanProceedWithCompute(OpKernelContext* c, CollectiveExecutor* col_exec,
                             const DoneCallback& done) {
    if (col_params_.group.group_size >
        col_params_.instance.device_names.size()) {
      // This is the first invocation: Finish initializing col_params_.
      // Schedule the `CompleteParamsAsync` call on a work queue that can handle
      // blocking work because it's not guaranteed that this call cannot block.
      c->collective_executor()->RunClosure([this, c, done, col_exec]() {
        VLOG(1) << "CollectiveOpKernel CompleteParams for collective "
                << col_params_.name << " device " << c->device()->name()
                << " group " << col_params_.group.group_key << " instance "
                << col_params_.instance.instance_key;
        col_exec->CompleteParamsAsync(
            c->device()->attributes(), &col_params_, c->cancellation_manager(),
            [this, c, done](const Status& s) {
              if (s.ok()) {
                col_params_.instance.impl_details.dependencies = dependencies_;
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

  CollectiveParams col_params_;
  std::vector<int32> dependencies_;
};

class CollectiveGatherOpKernel : public CollectiveOpKernel {
 public:
  explicit CollectiveGatherOpKernel(OpKernelConstruction* c)
      : CollectiveOpKernel(c) {
    col_params_.instance.type = GATHER_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_.group.group_size));
    OP_REQUIRES(
        c, col_params_.group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_.group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_.group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_.instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_.instance.data_type));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_.instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_.instance.impl_details.timeout_seconds));
    const NodeDef& real_node = c->def();
    col_params_.name = strings::StrCat(real_node.name(), ": Gather");
    col_params_.group.device_type = c->device_type();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            col_params_.name),
        done);

    auto output_shape = c->input(0).shape();
    output_shape.set_dim(
        0, output_shape.dim_size(0) * col_params_.group.group_size);
    col_params_.instance.shape = output_shape;

    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_output(0, col_params_.instance.shape, &output), done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, group_key = col_params_.group.group_key,
                        instance_key = col_params_.instance.instance_key,
                        done](const Status& s) {
      VLOG(1) << "CollectiveGatherOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << group_key << " instance " << instance_key
              << " status " << s;
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveGatherOpKernel ExecuteAsync start for collective "
            << col_params_.name << " device " << c->device()->name()
            << " group " << col_params_.group.group_key << " instance "
            << col_params_.instance.instance_key;
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveGatherOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveGather").Device(DEVICE_CPU),
                        CollectiveGatherOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveGather").Device(DEVICE_GPU),
                        CollectiveGatherOpKernel);

class CollectiveReduceOpKernel : public CollectiveOpKernel {
 public:
  explicit CollectiveReduceOpKernel(OpKernelConstruction* c)
      : CollectiveOpKernel(c) {
    col_params_.instance.type = REDUCTION_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_.group.group_size));
    OP_REQUIRES(
        c, col_params_.group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_.group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_.group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_.instance.instance_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("subdiv_offsets",
                      &col_params_.instance.impl_details.subdiv_offsets));
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
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_.instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("wait_for", &dependencies_));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_.instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_.instance.impl_details.timeout_seconds));
    VLOG(2) << "CollectiveReduce instance " << col_params_.instance.instance_key
            << " merge_op " << merge_op_name << " final_op " << final_op_name
            << " communication_hint "
            << col_params_.instance.impl_details.communication_hint
            << " timeout " << col_params_.instance.impl_details.timeout_seconds;

    const NodeDef& real_node = c->def();
    col_params_.name = strings::StrCat(real_node.name(), ": Reduce(",
                                       merge_op_name, ",", final_op_name, ")");
    col_params_.group.device_type = c->device_type();

    // Find the OpKernels by name, type and device type.
    NodeDef sub_node;
    // The merge_op takes two inputs
    sub_node.add_input(real_node.input(0));
    sub_node.add_input(real_node.input(0));
    sub_node.set_device(real_node.device());
    SetAttrValue(col_params_.instance.data_type,
                 &(*sub_node.mutable_attr())["T"]);
    col_params_.merge_op = BuildOpKernel(c, merge_op_name, &sub_node);
    col_params_.final_op = BuildOpKernel(c, final_op_name, &sub_node);
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            col_params_.name),
        done);
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
      col_params_.instance.shape = c->input(0).shape();
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, group_key = col_params_.group.group_key,
                        instance_key = col_params_.instance.instance_key,
                        done](const Status& s) {
      VLOG(1) << "CollectiveReduceOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << group_key << " instance " << instance_key
              << " status " << s;
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveReduceOpKernel ExecuteAsync start for collective "
            << col_params_.name << " device " << c->device()->name()
            << " group " << col_params_.group.group_key << " instance "
            << col_params_.instance.instance_key;
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveReduceOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveReduce").Device(DEVICE_CPU),
                        CollectiveReduceOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveReduce").Device(DEVICE_GPU),
                        CollectiveReduceOpKernel);

class CollectiveBcastSendOpKernel : public CollectiveOpKernel {
 public:
  explicit CollectiveBcastSendOpKernel(OpKernelConstruction* c)
      : CollectiveOpKernel(c) {
    col_params_.instance.type = BROADCAST_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_.group.group_size));
    OP_REQUIRES(
        c, col_params_.group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_.group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_.group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_.instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_.instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &col_params_.instance.shape));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_.instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_.instance.impl_details.timeout_seconds));
    col_params_.is_source = true;
    col_params_.instance.impl_details.subdiv_offsets = {0};

    col_params_.name =
        strings::StrCat(name(), ": Broadcast(", col_params_.is_source, ")");
    col_params_.group.device_type = c->device_type();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            col_params_.name),
        done);
    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor, trying to reuse the input.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(c,
                           c->forward_input_or_allocate_output(
                               {0}, 0, col_params_.instance.shape, &output),
                           done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;
    OP_REQUIRES_ASYNC(
        c, col_params_.instance.shape.IsSameSize(c->input(0).shape()),
        errors::Internal("Declared shape of op ", col_params_.name,
                         " does not match shape of input"),
        done);

    auto actual_done = [c, group_key = col_params_.group.group_key,
                        instance_key = col_params_.instance.instance_key,
                        done](const Status& s) {
      VLOG(1) << "CollectiveBcastSendOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << group_key << " instance " << instance_key
              << " status " << s;
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveBcastSendOpKernel ExecuteAsync start for collective "
            << col_params_.name << " device " << c->device()->name()
            << " group " << col_params_.group.group_key << " instance "
            << col_params_.instance.instance_key;
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveBcastSendOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSend").Device(DEVICE_CPU),
                        CollectiveBcastSendOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSend").Device(DEVICE_GPU),
                        CollectiveBcastSendOpKernel);

class CollectiveBcastRecvOpKernel : public CollectiveOpKernel {
 public:
  explicit CollectiveBcastRecvOpKernel(OpKernelConstruction* c)
      : CollectiveOpKernel(c) {
    col_params_.instance.type = BROADCAST_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_.group.group_size));
    OP_REQUIRES(
        c, col_params_.group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_.group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_.group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_.instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_.instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &col_params_.instance.shape));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_.instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_.instance.impl_details.timeout_seconds));
    col_params_.is_source = false;
    col_params_.instance.impl_details.subdiv_offsets = {0};

    col_params_.name =
        strings::StrCat(name(), ": Broadcast(", col_params_.is_source, ")");
    col_params_.group.device_type = c->device_type();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            col_params_.name),
        done);
    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // No input, so must allocate output.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_output(0, col_params_.instance.shape, &output), done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, group_key = col_params_.group.group_key,
                        instance_key = col_params_.instance.instance_key,
                        done](const Status& s) {
      VLOG(1) << "CollectiveBcastRecvOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << group_key << " instance_key " << instance_key
              << " status  " << s;
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveBcastRecvOpKernel ExecuteAsync start for collective "
            << col_params_.name << " device " << c->device()->name()
            << " group " << col_params_.group.group_key << " instance "
            << col_params_.instance.instance_key;
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveBcastRecvOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecv").Device(DEVICE_CPU),
                        CollectiveBcastRecvOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecv").Device(DEVICE_GPU),
                        CollectiveBcastRecvOpKernel);

class CollectiveReduceV2OpKernel : public AsyncOpKernel {
 public:
  explicit CollectiveReduceV2OpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c) {
    col_params_ = std::make_shared<CollectiveParams>();
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_->instance.data_type));
    string merge_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("merge_op", &merge_op_name));
    OP_REQUIRES_OK(c, c->GetAttr("merge_op", &merge_op_name));
    if (merge_op_name == "Max") {
      merge_op_name = "Maximum";
    } else if (merge_op_name == "Min") {
      merge_op_name = "Minimum";
    }
    string final_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("final_op", &final_op_name));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_->instance.impl_details.communication_hint));
    // Prepare OpKernels for reduction and final operations.
    // The merge_op takes two inputs
    NodeDef sub_node;
    sub_node.add_input(c->def().input(0));
    sub_node.add_input(c->def().input(0));
    sub_node.set_device(c->def().device());
    SetAttrValue(col_params_->instance.data_type,
                 &(*sub_node.mutable_attr())["T"]);
    col_params_->merge_op = BuildOpKernel(c, merge_op_name, &sub_node);
    col_params_->final_op = BuildOpKernel(c, final_op_name, &sub_node);

    col_params_->name = strings::StrCat(c->def().name(), ": ReduceV2(",
                                        merge_op_name, ",", final_op_name, ")");
    col_params_->group.device_type = c->device_type();
    // Add a default value for subdiv offsets, which is the same as the default
    // value in the V1 op's attribute.
    col_params_->instance.impl_details.subdiv_offsets.push_back(0);
    VLOG(2) << "CollectiveReduceV2 " << this << " name " << col_params_->name
            << " communication_hint "
            << col_params_->instance.impl_details.communication_hint;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            col_params_->name),
        done);
    const Tensor& input = c->input(0);
    const Tensor& group_size = c->input(1);
    const Tensor& group_key = c->input(2);
    const Tensor& instance_key = c->input(3);
    OP_REQUIRES_ASYNC(
        c, group_size.dims() == 0,
        errors::Internal("Unexpected dimensions on input group_size"), done);
    OP_REQUIRES_ASYNC(
        c, group_key.dims() == 0,
        errors::Internal("Unexpected dimensions on input group_key"), done);
    OP_REQUIRES_ASYNC(
        c, instance_key.dims() == 0,
        errors::Internal("Unexpected dimensions on input instance_key"), done);

    auto col_params = std::make_shared<CollectiveParams>();
    col_params->name = col_params_->name;
    col_params->group.device_type = col_params_->group.device_type;
    col_params->group.group_size = group_size.unaligned_flat<int32>()(0);
    col_params->group.group_key = group_key.unaligned_flat<int32>()(0);
    col_params->instance.type = REDUCTION_COLLECTIVE;
    col_params->instance.instance_key = instance_key.unaligned_flat<int32>()(0);
    col_params->instance.data_type = col_params_->instance.data_type;
    col_params->instance.impl_details.communication_hint =
        col_params_->instance.impl_details.communication_hint;
    col_params->instance.impl_details.timeout_seconds = 0;
    col_params->instance.impl_details.subdiv_offsets =
        col_params_->instance.impl_details.subdiv_offsets;
    col_params->merge_op = std::move(col_params_->merge_op);
    col_params->final_op = std::move(col_params_->final_op);
    VLOG(1) << "CollectiveReduceV2 group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;

    // Allocate the output tensor, trying to reuse the input.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c, c->forward_input_or_allocate_output({0}, 0, input.shape(), &output),
        done);
    col_params->instance.shape = input.shape();

    // Store the updated params in this OpKernel.
    col_params_ = col_params;

    // Resolve the collective params.
    // Schedule the `CompleteParamsAsync` call on a work queue that can handle
    // blocking work because it's not guaranteed that this call cannot block.
    c->collective_executor()->RunClosure([c, done = std::move(done), col_params,
                                          col_exec]() {
      VLOG(1) << "CollectiveReduceV2 CompleteParams for collective "
              << col_params->name << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance "
              << col_params->instance.instance_key;
      col_exec->CompleteParamsAsync(
          c->device()->attributes(), col_params.get(),
          c->cancellation_manager(),
          [c, done = std::move(done), col_params, col_exec](const Status& s) {
            if (s.ok()) {
              auto actual_done = [c, group_key = col_params->group.group_key,
                                  instance_key =
                                      col_params->instance.instance_key,
                                  done = std::move(done)](const Status& s) {
                VLOG(1) << "CollectiveReduceV2 ExecuteAsync done for "
                           "collective "
                        << c->op_kernel().name() << " device "
                        << c->device()->name() << " group " << group_key
                        << " instance " << instance_key << " status " << s;
                OP_REQUIRES_OK_ASYNC(c, s, done);
                done();
              };
              VLOG(1) << "CollectiveReduceV2 ExecuteAsync start for "
                         "collective "
                      << col_params->name << " device " << c->device()->name()
                      << " group " << col_params->group.group_key
                      << " instance " << col_params->instance.instance_key;
              col_exec->ExecuteAsync(
                  c, *col_params,
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

 private:
  std::shared_ptr<CollectiveParams> col_params_;
};

REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV2").Device(DEVICE_CPU),
                        CollectiveReduceV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveReduceV2").Device(DEVICE_GPU),
                        CollectiveReduceV2OpKernel);

class CollectiveGatherV2OpKernel : public AsyncOpKernel {
 public:
  explicit CollectiveGatherV2OpKernel(OpKernelConstruction* c)
      : AsyncOpKernel(c), device_type_(DEVICE_DEFAULT) {
    OP_REQUIRES_OK(c, c->GetAttr("T", &data_type_));
    OP_REQUIRES_OK(c, c->GetAttr("communication_hint", &communication_hint_));
    OP_REQUIRES_OK(c, c->GetAttr("timeout_seconds", &timeout_seconds_));
    name_ = strings::StrCat(c->def().name(), ": GatherV2");
    device_type_ = c->device_type();
    VLOG(2) << "CollectiveGatherV2 " << this << " name " << name_
            << " communication_hint " << communication_hint_;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            name_),
        done);
    const Tensor& input = c->input(0);
    const Tensor& group_size = c->input(1);
    const Tensor& group_key = c->input(2);
    const Tensor& instance_key = c->input(3);
    OP_REQUIRES_ASYNC(c, group_size.dims() == 0,
                      errors::InvalidArgument(
                          "Unexpected dimensions on input group_size, got ",
                          group_size.shape().DebugString()),
                      done);
    OP_REQUIRES_ASYNC(c, group_key.dims() == 0,
                      errors::InvalidArgument(
                          "Unexpected dimensions on input group_key, got ",
                          group_key.shape().DebugString()),
                      done);
    OP_REQUIRES_ASYNC(c, instance_key.dims() == 0,
                      errors::InvalidArgument(
                          "Unexpected dimensions on input instance_key, got ",
                          instance_key.shape().DebugString()),
                      done);

    auto col_params = new CollectiveParams();
    col_params->name = name_;
    col_params->group.device_type = device_type_;
    col_params->group.group_size = group_size.unaligned_flat<int32>()(0);
    OP_REQUIRES(
        c, col_params->group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params->group.group_size));
    col_params->group.group_key = group_key.unaligned_flat<int32>()(0);
    col_params->instance.type = GATHER_COLLECTIVE;
    col_params->instance.instance_key = instance_key.unaligned_flat<int32>()(0);
    col_params->instance.data_type = data_type_;
    col_params->instance.impl_details.communication_hint = communication_hint_;
    col_params->instance.impl_details.timeout_seconds = timeout_seconds_;
    VLOG(1) << "CollectiveGatherV2 group_size " << col_params->group.group_size
            << " group_key " << col_params->group.group_key << " instance_key "
            << col_params->instance.instance_key;

    auto output_shape = input.shape();
    output_shape.set_dim(
        0, output_shape.dim_size(0) * col_params->group.group_size);
    col_params->instance.shape = output_shape;

    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_output(0, col_params->instance.shape, &output), done);

    auto done_with_cleanup = [col_params, done = std::move(done)]() {
      delete col_params;
      done();
    };

    // Resolve the collective params.
    // Schedule the `CompleteParamsAsync` call on a work queue that can handle
    // blocking work because it's not guaranteed that this call cannot block.
    c->collective_executor()->RunClosure([c,
                                          done = std::move(done_with_cleanup),
                                          col_params, col_exec]() {
      VLOG(1) << "CollectiveGatherV2 CompleteParams for collective "
              << col_params->name << " device " << c->device()->name()
              << " group " << col_params->group.group_key << " instance "
              << col_params->instance.instance_key;
      col_exec->CompleteParamsAsync(
          c->device()->attributes(), col_params, c->cancellation_manager(),
          [c, done = std::move(done), col_params, col_exec](const Status& s) {
            if (s.ok()) {
              auto actual_done = [c, group_key = col_params->group.group_key,
                                  instance_key =
                                      col_params->instance.instance_key,
                                  done = std::move(done)](const Status& s) {
                VLOG(1) << "CollectiveGatherV2 ExecuteAsync done for "
                           "collective "
                        << c->op_kernel().name() << " device "
                        << c->device()->name() << " group " << group_key
                        << " instance " << instance_key << " status " << s;
                OP_REQUIRES_OK_ASYNC(c, s, done);
                done();
              };
              VLOG(1) << "CollectiveGatherV2 ExecuteAsync start for "
                         "collective "
                      << col_params->name << " device " << c->device()->name()
                      << " group " << col_params->group.group_key
                      << " instance " << col_params->instance.instance_key;
              col_exec->ExecuteAsync(
                  c, *col_params,
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

 private:
  DataType data_type_;
  string communication_hint_;
  float timeout_seconds_;
  DeviceType device_type_;
  string name_;
};

REGISTER_KERNEL_BUILDER(Name("CollectiveGatherV2").Device(DEVICE_CPU),
                        CollectiveGatherV2OpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveGatherV2").Device(DEVICE_GPU),
                        CollectiveGatherV2OpKernel);

}  // namespace
}  // namespace tensorflow
