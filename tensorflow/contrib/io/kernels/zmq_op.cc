/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/io_ops.cc.

#include <memory>
#include <zmq.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

class PollZmqOp : public OpKernel {
 public:
  explicit PollZmqOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("address", &address_));
    OP_REQUIRES_OK(context, context->GetAttr("timeout", &timeout_));
    // Create a context and connect a REQ socket
    context_ = zmq_ctx_new();
    OP_REQUIRES(context, context_ != nullptr,
                errors::Internal("Failed to initialize context."));
    socket_ = zmq_socket(context_, ZMQ_REQ);
    OP_REQUIRES(context, socket_ != nullptr,
                errors::Internal("Failed to initialize socket."));
    OP_REQUIRES(context,
                zmq_setsockopt(socket_, ZMQ_RCVTIMEO,
                               &timeout_, sizeof(timeout_)) == 0,
                errors::Internal("Failed to set timeout."));
    OP_REQUIRES(context, zmq_connect(socket_, &address_[0]) == 0,
                errors::Internal("Failed to connect to ", address_, "."));
  }

  ~PollZmqOp() {
    if (socket_ != nullptr) {
      zmq_close(socket_);
    }
    if (context_ != nullptr) {
      zmq_ctx_destroy(context_);
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input;
    OP_REQUIRES_OK(context, context->input("request", &input));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input->shape()),
                errors::InvalidArgument(
                    "Input message tensor must be scalar, but had shape: ",
                    input->shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("reply",
                                                      TensorShape({}), &output));

    // Get the message and response as string scalars
    const auto &request = input->scalar<string>()();
    auto &reply = output->scalar<string>()();

    // Lock for exclusive access (destructor releases)
    mutex_lock lock(mu_);

    // Prepare the message
    zmq_msg_t request_msg;
    OP_REQUIRES(context, zmq_msg_init_size(&request_msg, request.size()) == 0,
                errors::Internal("Failed to initialize request."));
    memmove(zmq_msg_data(&request_msg), &request[0], request.size());
    // Send the message (this also takes care of clean up)
    OP_REQUIRES(context, zmq_msg_send(&request_msg, socket_, 0) == request.size(),
                errors::Internal("Failed to send reply."));

    // Get the response
    zmq_msg_t reply_msg;
    OP_REQUIRES(context, zmq_msg_init(&reply_msg) == 0,
                errors::Internal("Failed to initialize reply."));
    OP_REQUIRES(context, zmq_msg_recv(&reply_msg, socket_, 0) != -1,
                errors::Internal("Failed to receive reply within ", timeout_, "ms."));
    // Copy the data and clean up
    reply.resize(zmq_msg_size(&reply_msg));
    memmove(&reply[0], zmq_msg_data(&reply_msg), zmq_msg_size(&reply_msg));
    OP_REQUIRES(context, zmq_msg_close(&reply_msg) == 0,
                errors::Internal("Failed to close reply message."));
  }

 private:
  string address_;
  int timeout_;
  void* context_ = nullptr;
  void* socket_ = nullptr;
  mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("PollZmq").Device(DEVICE_CPU), PollZmqOp);

}  // namespace tensorflow
