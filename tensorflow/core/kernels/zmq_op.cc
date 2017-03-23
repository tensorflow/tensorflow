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
#include <zmq.hpp>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

class PollZmqOp : public OpKernel {
 public:
  explicit PollZmqOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("address", &address_));
    socket_ = new zmq::socket_t(context_, ZMQ_REQ);
    socket_->connect(address_);
  }

  ~PollZmqOp() {
    delete socket_;
  }

  using OpKernel::OpKernel;
  void Compute(OpKernelContext* context) override {
    const Tensor* input;
    OP_REQUIRES_OK(context, context->input("message", &input));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input->shape()),
                errors::InvalidArgument(
                    "Input message tensor must be scalar, but had shape: ",
                    input->shape().DebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output("response",
                                                      TensorShape({}), &output));

    // Get the message and response as string scalars
    auto message = input->scalar<string>()();
    auto &response = output->scalar<string>()();

    // Send the request
    zmq::message_t request (message.size());
    memmove(request.data(), &message[0], message.size());
    socket_->send(request);

    //  Get the reply
    zmq::message_t reply;
    socket_->recv(&reply);
    response.resize(reply.size());
    memmove(&response[0], reply.data(), reply.size());
  }

 private:
  string address_;
  zmq::context_t context_;
  zmq::socket_t* socket_;
};

REGISTER_KERNEL_BUILDER(Name("PollZmq").Device(DEVICE_CPU), PollZmqOp);

}  // namespace tensorflow
