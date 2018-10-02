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

#include "tensorflow/compiler/jit/xla_device_ops.h"

#include <memory>

#include "tensorflow/compiler/jit/xla_device_context.h"
#include "tensorflow/compiler/jit/xla_tensor.h"

namespace tensorflow {

XlaDeviceDummyOp::XlaDeviceDummyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

void XlaDeviceDummyOp::Compute(OpKernelContext* ctx) {
  LOG(FATAL) << "Attempted to execute Op " << name() << " type "
             << type_string() << " on an XLA device. This should never happen.";
}

XlaAssignVariableOp::XlaAssignVariableOp(OpKernelConstruction* c)
    : AsyncOpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("dtype", &dtype_));
}

void XlaAssignVariableOp::ComputeAsync(OpKernelContext* context,
                                       DoneCallback done) {
  OP_REQUIRES_ASYNC(context, dtype_ == context->input(1).dtype(),
                    errors::InvalidArgument(
                        "Variable and value dtypes don't match; respectively, ",
                        dtype_, " and ", context->input(1).dtype()),
                    done);
  Var* variable = nullptr;
  OP_REQUIRES_OK_ASYNC(
      context,
      LookupOrCreateResource<Var>(
          context, HandleFromInput(context, 0), &variable,
          [this, context](Var** ptr) {
            *ptr = new Var(dtype_);
            PersistentTensor unused;
            Tensor* tmp;
            AllocatorAttributes attr;
            TF_RETURN_IF_ERROR(context->allocate_persistent(
                dtype_, context->input(1).shape(), &unused, &tmp, attr));
            *(*ptr)->tensor() = *tmp;
            return Status::OK();
          }),
      done);
  core::ScopedUnref s(variable);

  OP_REQUIRES_ASYNC(context, variable->tensor()->dtype() == dtype_,
                    errors::InvalidArgument(
                        "Trying to assign variable with wrong dtype. Expected ",
                        DataTypeString(variable->tensor()->dtype()), " got ",
                        DataTypeString(dtype_)),
                    done);

  const Tensor& value = context->input(1);
  AllocatorAttributes attr;

  // Copying is unnecessary if we are the last user of the value tensor, we can
  // just adopt the input tensor's buffer instead.
  std::unique_ptr<Tensor> input_alias = context->forward_input(
      1, /*output_index=*/OpKernelContext::Params::kNoReservation, dtype_,
      value.shape(), DEVICE_MEMORY, attr);
  mutex_lock ml(*variable->mu());
  variable->is_initialized = true;
  if (input_alias) {
    *variable->tensor() = *input_alias;
    done();
    return;
  }

  // Need to copy, but maybe we can re-use variable's buffer?
  if (!XlaTensor::RefCountIsOne(*variable->tensor()) ||
      !variable->tensor()->shape().IsSameSize(value.shape())) {
    // Copy to new buffer
    PersistentTensor unused;
    Tensor* tmp;
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_persistent(dtype_, value.shape(),
                                                      &unused, &tmp, attr),
                         done);
    *variable->tensor() = *tmp;
  }

  XlaDeviceContext* device_context =
      static_cast<XlaDeviceContext*>(context->op_device_context());

  variable->Ref();
  device_context->CopyDeviceTensorToDevice(
      value, variable->tensor(), [context, variable, done](Status status) {
        variable->Unref();
        context->SetStatus(status);
        done();
      });
}

}  // namespace tensorflow
