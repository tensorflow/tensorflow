/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/sendrecv_ops.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

static string GetRendezvousKeyPrefix(const string& send_device,
                                     const string& recv_device,
                                     const uint64 send_device_incarnation,
                                     const string& tensor_name) {
  return strings::StrCat(send_device, ";",
                         strings::FpToString(send_device_incarnation), ";",
                         recv_device, ";", tensor_name);
}

static void GetRendezvousKey(const string& key_prefix,
                             const FrameAndIter& frame_iter, string* key) {
  key->clear();
  strings::StrAppend(key, key_prefix, ";", frame_iter.frame_id, ":",
                     frame_iter.iter_id);
}

SendOp::SendOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
  string send_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
  string recv_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
  uint64 send_device_incarnation;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("send_device_incarnation",
                        reinterpret_cast<int64*>(&send_device_incarnation)));
  string tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_prefix_ = GetRendezvousKeyPrefix(send_device, recv_device,
                                       send_device_incarnation, tensor_name);
  // The vast majority of Send nodes are outside any loop context, so
  // proactively cache the rendezvous key for the top-level.
  GetRendezvousKey(key_prefix_, {0, 0}, &parsed_key_.buf_);
  OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key_.buf_, &parsed_key_));
}

void SendOp::Compute(OpKernelContext* ctx) {
  OP_REQUIRES(
      ctx, ctx->rendezvous() != nullptr,
      errors::Internal("Op kernel context needs to provide a rendezvous."));

  // The device context may be passed between the Send/Recv
  // boundary, so that the device context used to produce the Tensor
  // is used when performing the copy on the recv side (which may be
  // a different device).
  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = ctx->input_alloc_attr(0);

  if (ctx->frame_iter() == FrameAndIter(0, 0)) {
    // Use the cached rendezvous key.
    VLOG(2) << "Send " << parsed_key_.buf_;
    OP_REQUIRES_OK(ctx,
                   ctx->rendezvous()->Send(parsed_key_, args, ctx->input(0),
                                           ctx->is_input_dead()));
  } else {
    Rendezvous::ParsedKey in_loop_parsed;
    GetRendezvousKey(key_prefix_, ctx->frame_iter(), &in_loop_parsed.buf_);
    VLOG(2) << "Send " << in_loop_parsed.buf_;
    OP_REQUIRES_OK(ctx,
                   Rendezvous::ParseKey(in_loop_parsed.buf_, &in_loop_parsed));

    OP_REQUIRES_OK(ctx,
                   ctx->rendezvous()->Send(in_loop_parsed, args, ctx->input(0),
                                           ctx->is_input_dead()));
  }
}

REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_CPU), SendOp);
REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_GPU), SendOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("_Send").Device(DEVICE_SYCL), SendOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostSend").Device(DEVICE_SYCL).HostMemory("tensor"), SendOp);
#endif // TENSORFLOW_USE_SYCL

REGISTER_KERNEL_BUILDER(Name("_HostSend").Device(DEVICE_CPU), SendOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostSend").Device(DEVICE_GPU).HostMemory("tensor"), SendOp);

RecvOp::RecvOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
  string send_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("send_device", &send_device));
  string recv_device;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("recv_device", &recv_device));
  uint64 send_device_incarnation;
  OP_REQUIRES_OK(
      ctx, ctx->GetAttr("send_device_incarnation",
                        reinterpret_cast<int64*>(&send_device_incarnation)));
  string tensor_name;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("tensor_name", &tensor_name));
  key_prefix_ = GetRendezvousKeyPrefix(send_device, recv_device,
                                       send_device_incarnation, tensor_name);
  // The vast majority of Recv nodes are outside any loop context, so
  // proactively cache the rendezvous key for the top-level.
  GetRendezvousKey(key_prefix_, {0, 0}, &parsed_key_.buf_);
  OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(parsed_key_.buf_, &parsed_key_));
}

void RecvOp::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  OP_REQUIRES(
      ctx, ctx->rendezvous() != nullptr,
      errors::Internal("Op kernel context needs to provide a rendezvous."));

  Rendezvous::Args args;
  args.device_context = ctx->op_device_context();
  args.alloc_attrs = ctx->output_alloc_attr(0);
  using namespace std::placeholders;
  Rendezvous::DoneCallback done_cb = std::bind(
      [ctx](DoneCallback done,
            // Begin unbound arguments.
            const Status& s, const Rendezvous::Args& send_args,
            const Rendezvous::Args& recv_args, const Tensor& val,
            bool is_dead) {
        ctx->SetStatus(s);
        if (s.ok()) {
          // 'ctx' allocates the output tensor of the expected type.
          // The runtime checks whether the tensor received here is
          // the same type.
          if (!is_dead) {
            ctx->set_output(0, val);
          }
          *ctx->is_output_dead() = is_dead;
        }
        done();
      },
      std::move(done), _1, _2, _3, _4, _5);

  if (ctx->frame_iter() == FrameAndIter(0, 0)) {
    VLOG(2) << "Recv " << parsed_key_.buf_;
    ctx->rendezvous()->RecvAsync(parsed_key_, args, std::move(done_cb));
  } else {
    Rendezvous::ParsedKey in_loop_parsed;
    GetRendezvousKey(key_prefix_, ctx->frame_iter(), &in_loop_parsed.buf_);
    VLOG(2) << "Recv " << in_loop_parsed.buf_;
    OP_REQUIRES_OK_ASYNC(
        ctx, Rendezvous::ParseKey(in_loop_parsed.buf_, &in_loop_parsed), done);

    ctx->rendezvous()->RecvAsync(in_loop_parsed, args, std::move(done_cb));
  }
}

REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_CPU), RecvOp);
REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_GPU), RecvOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("_Recv").Device(DEVICE_SYCL), RecvOp);
#endif // TENSORFLOW_USE_SYCL

REGISTER_KERNEL_BUILDER(Name("_HostRecv").Device(DEVICE_CPU), RecvOp);
REGISTER_KERNEL_BUILDER(
    Name("_HostRecv").Device(DEVICE_GPU).HostMemory("tensor"), RecvOp);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("_HostRecv").Device(DEVICE_SYCL).HostMemory("tensor"), RecvOp);
#endif // TENSORFLOW_USE_SYCL

}  // end namespace tensorflow
