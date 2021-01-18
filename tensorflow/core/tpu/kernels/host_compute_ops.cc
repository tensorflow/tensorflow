/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

// The RecvAtHost op is used to deliver data from the device at the start of a
// host compute block. Setting `device_ordinal_is_attr` to true and false
// will switch between using device ordinal as an attribute and a runtime value
// respectively. To minimize cloning of ops/functions, it may be necessary to
// have device ordinal be a runtime value.
template <bool device_ordinal_is_attr>
class RecvAtHostOp : public AsyncOpKernel {
 public:
  explicit RecvAtHostOp(OpKernelConstruction* ctx) : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
    int device_ordinal = 0;
    if (device_ordinal_is_attr) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal));
      OP_REQUIRES(
          ctx, device_ordinal >= 0,
          errors::Internal("RecvAtHost device_ordinal must be non negative"));
      OP_REQUIRES(ctx, ctx->num_inputs() == 1,
                  errors::Internal("RecvAtHost must have exactly one input"));
      OP_REQUIRES(ctx, ctx->input_type(0) == DT_STRING,
                  errors::Internal("RecvAtHost input must have string type"));
    } else {
      OP_REQUIRES(ctx, ctx->num_inputs() == 2,
                  errors::Internal("RecvAtHost must have exactly two inputs"));
      OP_REQUIRES(ctx, ctx->input_type(0) == DT_STRING,
                  errors::Internal("RecvAtHost input 0 must have string type"));
      OP_REQUIRES(ctx, ctx->input_type(1) == DT_INT64,
                  errors::Internal("RecvAtHost input 1 must have int64 type"));
    }

    DeviceNameUtils::ParsedName parsed_name;
    OP_REQUIRES(
        ctx,
        DeviceNameUtils::ParseFullName(ctx->device()->name(), &parsed_name),
        errors::Internal("Could not parse device name."));
    parsed_name.type = "CPU";
    parsed_name.id = 0;
    cpu_device_ = DeviceNameUtils::ParsedNameToString(parsed_name);
    if (device_ordinal_is_attr) {
      parsed_name.type = "TPU";
      parsed_name.id = device_ordinal;
      tpu_device_ = DeviceNameUtils::ParsedNameToString(parsed_name);
      VLOG(2) << "  tpu_device_ = " << tpu_device_;
      VLOG(2) << "  cpu_device_ = " << cpu_device_;
    }
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
    string tpu_device;
    if (!device_ordinal_is_attr) {
      const Tensor& device_ordinal_tensor = ctx->input(1);
      OP_REQUIRES_ASYNC(
          ctx, TensorShapeUtils::IsScalar(device_ordinal_tensor.shape()),
          errors::InvalidArgument("device_ordinal must be a scalar, not ",
                                  device_ordinal_tensor.shape().DebugString()),
          done);
      const int device_ordinal = device_ordinal_tensor.flat<int64>()(0);
      OP_REQUIRES_ASYNC(
          ctx, device_ordinal >= 0,
          errors::Internal("RecvAtHost device_ordinal must be non negative"),
          done);
      DeviceNameUtils::ParsedName parsed_name;
      OP_REQUIRES_ASYNC(
          ctx, DeviceNameUtils::ParseFullName(cpu_device_, &parsed_name),
          errors::Internal("Could not parse device name."), done);
      parsed_name.type = "TPU";
      parsed_name.id = device_ordinal;
      tpu_device = DeviceNameUtils::ParsedNameToString(parsed_name);
      VLOG(2) << "  tpu_device_ = " << tpu_device;
      VLOG(2) << "  cpu_device_ = " << cpu_device_;
    }

    const Tensor& input = ctx->input(0);
    VLOG(2) << input.DebugString();
    OP_REQUIRES_ASYNC(
        ctx,
        TensorShapeUtils::IsVector(input.shape()) &&
            input.shape().dim_size(0) == 3,
        errors::InvalidArgument("Input shape ", input.shape().DebugString(),
                                " is not a vector of length 3."),
        done);
    const string rendezvous_key_base = input.vec<tstring>()(1);
    OP_REQUIRES_ASYNC(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."),
        done);

    // Early return if there is no output to be received. Call `done()` to
    // unblock following execution.
    if (ctx->num_outputs() == 0) {
      done();
      return;
    }

    // Make all the parsed keys before starting any rendezvous->Recv calls to
    // avoid having to deal with an error case after some Recv have been
    // started.
    std::vector<string> rendezvous_key(ctx->num_outputs());
    std::vector<Rendezvous::ParsedKey> parsed_key(ctx->num_outputs());
    for (int i = 0; i < ctx->num_outputs(); ++i) {
      rendezvous_key[i] = Rendezvous::CreateKey(
          device_ordinal_is_attr ? tpu_device_ : tpu_device,
          /*src_incarnation=*/1, cpu_device_,
          strings::StrCat(rendezvous_key_base, key_, "_dtoh_", i),
          FrameAndIter(0, 0));

      OP_REQUIRES_OK_ASYNC(
          ctx, Rendezvous::ParseKey(rendezvous_key[i], &parsed_key[i]), done);
    }

    std::atomic_int_fast32_t* counter =
        new std::atomic_int_fast32_t(ctx->num_outputs());

    int num_outputs = ctx->num_outputs();
    for (int i = 0; i < num_outputs; ++i) {
      Rendezvous::Args args;
      args.device_context = ctx->op_device_context();
      args.alloc_attrs = ctx->output_alloc_attr(i);

      const string& key = rendezvous_key[i];
      VLOG(2) << "Recv " << key;
      ctx->rendezvous()->RecvAsync(
          parsed_key[i], args,
          [ctx, i, counter, key, done](const Status& s,
                                       const Rendezvous::Args& send_args,
                                       const Rendezvous::Args& recv_args,
                                       const Tensor& val, bool is_dead) {
            ctx->SetStatus(s);
            if (s.ok()) {
              ctx->set_output(i, val);
            }
            int previously_finished = counter->fetch_sub(1);
            VLOG(2) << "Processing Recv " << key << " " << s
                    << " previously finished " << previously_finished;
            if (previously_finished == 1) {
              delete counter;
              done();
            }
          });
    }
  }

 private:
  string key_;
  string tpu_device_;
  string cpu_device_;

  // RecvAtHostOp is neither copyable nor movable.
  RecvAtHostOp(const RecvAtHostOp&) = delete;
  RecvAtHostOp& operator=(const RecvAtHostOp&) = delete;
};

// The SendFromHost op is used to deliver data to the device at the end of a
// host compute block. Setting `device_ordinal_is_attr` to true and false will
// switch between using device ordinal as an attribute and a runtime value
// respectively. To minimize cloning of ops/functions, it may be necessary to
// have device ordinal be a runtime value.
template <bool device_ordinal_is_attr>
class SendFromHostOp : public OpKernel {
 public:
  explicit SendFromHostOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("key", &key_));
    int device_ordinal = 0;
    if (device_ordinal_is_attr) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("device_ordinal", &device_ordinal));
      OP_REQUIRES(
          ctx, device_ordinal >= 0,
          errors::Internal("SendFromHost device_ordinal must be non negative"));
      OP_REQUIRES(
          ctx, ctx->num_inputs() > 0,
          errors::Internal("SendFromHost must have at least one input"));
      OP_REQUIRES(
          ctx, ctx->input_type(ctx->num_inputs() - 1) == DT_STRING,
          errors::Internal("SendFromHost last input must have string type"));
    } else {
      OP_REQUIRES(
          ctx, ctx->num_inputs() > 1,
          errors::Internal("SendFromHost must have at least two inputs"));
      OP_REQUIRES(
          ctx, ctx->input_type(ctx->num_inputs() - 2) == DT_STRING,
          errors::Internal(
              "SendFromHost second to last input must have string type"));
      OP_REQUIRES(
          ctx, ctx->input_type(ctx->num_inputs() - 1) == DT_INT64,
          errors::Internal("SendFromHost last input must have int64 type"));
    }

    DeviceNameUtils::ParsedName parsed_name;
    OP_REQUIRES(
        ctx,
        DeviceNameUtils::ParseFullName(ctx->device()->name(), &parsed_name),
        errors::Internal("Could not parse device name."));
    parsed_name.type = "CPU";
    parsed_name.id = 0;
    cpu_device_ = DeviceNameUtils::ParsedNameToString(parsed_name);
    if (device_ordinal_is_attr) {
      parsed_name.type = "TPU";
      parsed_name.id = device_ordinal;
      tpu_device_ = DeviceNameUtils::ParsedNameToString(parsed_name);
      VLOG(2) << "  tpu_device_ = " << tpu_device_;
      VLOG(2) << "  cpu_device_ = " << cpu_device_;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    std::string tpu_device;
    if (!device_ordinal_is_attr) {
      const Tensor& device_ordinal_tensor = ctx->input(ctx->num_inputs() - 1);
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsScalar(device_ordinal_tensor.shape()),
          errors::InvalidArgument("device_ordinal must be a scalar, not ",
                                  device_ordinal_tensor.shape().DebugString()));
      const int device_ordinal = device_ordinal_tensor.flat<int64>()(0);
      OP_REQUIRES(
          ctx, device_ordinal >= 0,
          errors::Internal("SendFromHost device_ordinal must be non negative"));
      DeviceNameUtils::ParsedName parsed_name;
      OP_REQUIRES(ctx,
                  DeviceNameUtils::ParseFullName(cpu_device_, &parsed_name),
                  errors::Internal("Could not parse device name."));
      parsed_name.type = "TPU";
      parsed_name.id = device_ordinal;
      tpu_device = DeviceNameUtils::ParsedNameToString(parsed_name);
      VLOG(2) << "  tpu_device_ = " << tpu_device;
      VLOG(2) << "  cpu_device_ = " << cpu_device_;
    }

    const int num_send_inputs =
        ctx->num_inputs() - (device_ordinal_is_attr ? 1 : 2);
    const Tensor& key_input = ctx->input(num_send_inputs);
    OP_REQUIRES(ctx,
                TensorShapeUtils::IsVector(key_input.shape()) &&
                    key_input.shape().dim_size(0) == 3,
                errors::InvalidArgument("Key input shape ",
                                        key_input.shape().DebugString(),
                                        " is not a vector of length 3."));
    const string rendezvous_key_base = key_input.vec<tstring>()(1);
    OP_REQUIRES(
        ctx, ctx->rendezvous() != nullptr,
        errors::Internal("Op kernel context needs to provide a rendezvous."));

    for (int i = 0; i < num_send_inputs; ++i) {
      Rendezvous::Args args;
      args.device_context = ctx->op_device_context();
      args.alloc_attrs = ctx->input_alloc_attr(i);

      // TODO(misard) Fix this once we have replication.
      const string& rendezvous_key = Rendezvous::CreateKey(
          cpu_device_, /*src_incarnation=*/1,
          device_ordinal_is_attr ? tpu_device_ : tpu_device,
          strings::StrCat(rendezvous_key_base, key_, "_htod_", i),
          FrameAndIter(0, 0));

      Rendezvous::ParsedKey parsed_key;
      OP_REQUIRES_OK(ctx, Rendezvous::ParseKey(rendezvous_key, &parsed_key));
      VLOG(2) << "Send " << rendezvous_key;
      OP_REQUIRES_OK(
          ctx, ctx->rendezvous()->Send(parsed_key, args, ctx->input(i), false));
    }
  }

 private:
  string key_;
  string cpu_device_;
  string tpu_device_;

  // SendFromHostOp is neither copyable nor movable.
  SendFromHostOp(const SendFromHostOp&) = delete;
  SendFromHostOp& operator=(const SendFromHostOp&) = delete;
};

}  // anonymous namespace

// These ops execute on the CPU device and must specify a non-negative value for
// device_ordinal to indicate which TPU to send infeed to.
REGISTER_KERNEL_BUILDER(Name("_XlaRecvAtHost").Device(DEVICE_CPU),
                        RecvAtHostOp<true>);

REGISTER_KERNEL_BUILDER(Name("_XlaRecvAtHostV2").Device(DEVICE_CPU),
                        RecvAtHostOp<false>);

REGISTER_KERNEL_BUILDER(Name("_XlaSendFromHost").Device(DEVICE_CPU),
                        SendFromHostOp<true>);

REGISTER_KERNEL_BUILDER(Name("_XlaSendFromHostV2").Device(DEVICE_CPU),
                        SendFromHostOp<false>);

}  // namespace tensorflow
