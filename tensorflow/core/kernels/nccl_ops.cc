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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

#if GOOGLE_CUDA
#include "third_party/nccl/nccl.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/rccl/rccl.h"
#endif
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/nccl/nccl_manager.h"

namespace tensorflow {
namespace {

// Base class for all communicator ops that use nccl.
//
// About memory management and stream syncing:
// 1. The nccl communicator has a stream for each rank.
// 2. For input tensors to the communicator, the compute stream is passed to the
//    NcclManager which will do a needed
//    communicator_stream.ThenWaitFor(input_tensor_stream).
// 3. The done_callback of the async kernel is not called by the
//    NcclManager until after the communicator kernel is complete. This
//    is enough to a) keep the input tensor data valid for the lifetime of the
//    collective; and b) ensure the data in the output tensor is available
//    when the async op kernel's done callback is called.
class NcclAsyncOpBase : public AsyncOpKernel {
 public:
  explicit NcclAsyncOpBase(OpKernelConstruction* c) : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("num_devices", &num_devices_));
    OP_REQUIRES_OK(c, c->GetAttr("shared_name", &collective_prefix_));
  }

  string GetCollectiveKey(OpKernelContext* c) {
    return strings::StrCat(collective_prefix_, ";", c->step_id(), ";",
                           c->frame_iter().frame_id, ":",
                           c->frame_iter().iter_id);
  }

  int num_devices() const { return num_devices_; }

 private:
  int num_devices_;
  string collective_prefix_;

  TF_DISALLOW_COPY_AND_ASSIGN(NcclAsyncOpBase);
};

class NcclReduceOpBase : public NcclAsyncOpBase {
 public:
  explicit NcclReduceOpBase(OpKernelConstruction* c) : NcclAsyncOpBase(c) {
    string reduction;
    OP_REQUIRES_OK(c, c->GetAttr("reduction", &reduction));
    if (reduction == "min") {
      reduction_op_ = ncclMin;
    } else if (reduction == "max") {
      reduction_op_ = ncclMax;
    } else if (reduction == "sum") {
      reduction_op_ = ncclSum;
    } else if (reduction == "prod") {
      reduction_op_ = ncclProd;
    } else {
      OP_REQUIRES_OK(c,
                     errors::InvalidArgument("Invalid reduction: ", reduction));
    }
  }

  ncclRedOp_t reduction_op() const { return reduction_op_; }

 private:
  ncclRedOp_t reduction_op_;
};

// To execute a single all-reduce, this kernel is called once for each of the
// <k> devices in the communicator.
class NcclAllReduceOpKernel : public NcclReduceOpBase {
 public:
  explicit NcclAllReduceOpKernel(OpKernelConstruction* c)
      : NcclReduceOpBase(c) {}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    const Tensor* input = &c->input(0);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        c, c->forward_input_or_allocate_output({0}, 0, input->shape(), &output),
        done);
    auto actual_done = [c, done](Status s) {
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_gpu_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info,
        input, output, /*global_rank=*/-1,
        std::move(actual_done));
    NcclManager::instance()->AddToAllReduce(
        std::move(participant),
        {GetCollectiveKey(c),
         /*num_local_devices=*/num_devices(),
         /*num_global_devices=*/num_devices(),
         /*communicator_key=*/"", /*source_rank=*/-1},
        reduction_op());
  }
};
REGISTER_KERNEL_BUILDER(Name("NcclAllReduce").Device(DEVICE_GPU),
                        NcclAllReduceOpKernel);

// To execute a single reduce, this kernel is called once for all but one of the
// <k> devices in the communicator, and NcclReduceRecvKernel is called once for
// the remaining device.
class NcclReduceSendKernel : public NcclReduceOpBase {
 public:
  explicit NcclReduceSendKernel(OpKernelConstruction* c)
      : NcclReduceOpBase(c) {}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto actual_done = [c, done](Status s) {
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_gpu_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info,
        &c->input(0), /*output=*/nullptr, /*global_rank=*/-1,
        std::move(actual_done));
    NcclManager::instance()->AddReduceSend(
        std::move(participant),
        {GetCollectiveKey(c),
         /*num_local_devices=*/num_devices(),
         /*num_global_devices=*/num_devices(),
         /*communicator_key=*/"", /*source_rank=*/-1},
        reduction_op());
  }
};
REGISTER_KERNEL_BUILDER(Name("_NcclReduceSend").Device(DEVICE_GPU),
                        NcclReduceSendKernel);

// To execute a single reduce, this kernel is called once for one devices, and
// NcclReduceSendKernel is called for all other <k-1> devices in the
// communicator.
class NcclReduceRecvKernel : public NcclReduceOpBase {
 public:
  explicit NcclReduceRecvKernel(OpKernelConstruction* c)
      : NcclReduceOpBase(c) {}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    const Tensor* input = &c->input(0);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, input->shape(), &output),
                         done);

    auto actual_done = [c, done](Status s) {
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_gpu_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info,
        input, output, /*global_rank=*/-1,
        std::move(actual_done));
    NcclManager::instance()->AddReduceRecv(
        std::move(participant),
        {GetCollectiveKey(c),
         /*num_local_devices=*/num_devices(),
         /*num_global_devices=*/num_devices(),
         /*communicator_key=*/"", /*source_rank=*/-1},
        reduction_op());
  }

 private:
  ncclRedOp_t reduction_op_;
};
REGISTER_KERNEL_BUILDER(Name("_NcclReduceRecv").Device(DEVICE_GPU),
                        NcclReduceRecvKernel);

// To execute a single broadcast, this kernel is called once for one device, and
// NcclBroadcastRecvKernel is called for all other <k-1> devices in the
// communicator.
class NcclBroadcastSendKernel : public NcclAsyncOpBase {
 public:
  explicit NcclBroadcastSendKernel(OpKernelConstruction* c)
      : NcclAsyncOpBase(c) {}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    auto actual_done = [c, done](Status s) {
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_gpu_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info,
        &c->input(0), /*output=*/nullptr, /*global_rank=*/-1,
        std::move(actual_done));
    NcclManager::instance()->AddBroadcastSend(
        std::move(participant), {GetCollectiveKey(c),
                                 /*num_local_devices=*/num_devices(),
                                 /*num_global_devices=*/num_devices(),
                                 /*communicator_key=*/"", /*source_rank=*/-1});
  }
};
REGISTER_KERNEL_BUILDER(Name("_NcclBroadcastSend").Device(DEVICE_GPU),
                        NcclBroadcastSendKernel);

// To execute a single broadcast, this kernel is called once for all but one of
// the <k> devices in the communicator, and NcclBroadcastSendKernel is called
// once for the remaining device.
class NcclBroadcastRecvKernel : public NcclAsyncOpBase {
 public:
  explicit NcclBroadcastRecvKernel(OpKernelConstruction* c)
      : NcclAsyncOpBase(c) {}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    const Tensor& shape_t = c->input(0);
    TensorShape shape;
    OP_REQUIRES_OK_ASYNC(
        c, TensorShapeUtils::MakeShape(shape_t.vec<int32>(), &shape), done);
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, shape, &output), done);

    auto actual_done = [c, done](Status s) {
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };

    auto* compute_stream = c->op_device_context()->stream();
    auto* gpu_info = c->device()->tensorflow_gpu_device_info();
    auto participant = absl::make_unique<NcclManager::Participant>(
        compute_stream->parent(), compute_stream, gpu_info,
        /*input=*/nullptr, output, /*global_rank=*/-1,
        std::move(actual_done));
    NcclManager::instance()->AddBroadcastRecv(
        std::move(participant), {GetCollectiveKey(c),
                                 /*num_local_devices=*/num_devices(),
                                 /*num_global_devices=*/num_devices(),
                                 /*communicator_key=*/"", /*source_rank=*/-1});
  }
};
REGISTER_KERNEL_BUILDER(
    Name("_NcclBroadcastRecv").Device(DEVICE_GPU).HostMemory("shape"),
    NcclBroadcastRecvKernel);

// Define stub kernels for the ops that get replaced post placement.
class NcclStubKernel : public AsyncOpKernel {
 public:
  explicit NcclStubKernel(OpKernelConstruction* c) : AsyncOpKernel(c) {}
  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    c->SetStatus(errors::Unimplemented(
        "This op should be replaced during graph optimization."));
    done();
  }
};
REGISTER_KERNEL_BUILDER(Name("NcclBroadcast").Device(DEVICE_GPU),
                        NcclStubKernel);
REGISTER_KERNEL_BUILDER(Name("NcclReduce").Device(DEVICE_GPU), NcclStubKernel);

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
