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
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {
class CollectiveOpKernel : public AsyncOpKernel {
 public:
  explicit CollectiveOpKernel(OpKernelConstruction* c) : AsyncOpKernel(c) {}

  // A string encoding instance, frame and iter to be handed off to
  // the implementation for use in generating RecvBuf keys.
  string GetCollectiveKey(OpKernelContext* c) {
    return strings::StrCat(col_params_.instance.instance_key, ":",
                           c->frame_iter().frame_id, ":",
                           c->frame_iter().iter_id);
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
            c->device()->name(), &col_params_, c->cancellation_manager(),
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

    auto actual_done = [c, done](const Status& s) {
      VLOG(1) << "CollectiveGatherOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
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
    VLOG(2) << "CollectiveReduce instance " << col_params_.instance.instance_key
            << " merge_op " << merge_op_name << " final_op " << final_op_name
            << " communication_hint "
            << col_params_.instance.impl_details.communication_hint;

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

  std::unique_ptr<OpKernel> BuildOpKernel(OpKernelConstruction* c,
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
      c->CtxFailureWithWarning(errors::Internal("Failed to build OpKernel for ",
                                                name, " : ",
                                                status.error_message()));
    }
    return k;
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

    auto actual_done = [c, done](const Status& s) {
      VLOG(1) << "CollectiveReduceOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
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

    auto actual_done = [c, done](const Status& s) {
      VLOG(1) << "CollectiveBcastSendOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
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

    auto actual_done = [c, done](const Status& s) {
      VLOG(1) << "CollectiveBcastRecvOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
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

}  // namespace
}  // namespace tensorflow
