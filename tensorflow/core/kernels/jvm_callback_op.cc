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

#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/kernels/jvm_callback_op.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {
  // Given the 'call', prepares the inputs as a JNI long array that is appropriate for calling the registry.
  jlongArray MakeInputs(JVMCall* call) {
    unsigned long n = call->inputs.size();
    jlongArray inputs = call->env->NewLongArray(static_cast<jsize>(n));
    jlong* inputs_array = call->env->GetLongArrayElements(inputs, nullptr);
    for (int64 i = 0; i < n; ++i) {
      const Tensor& t = call->inputs[i];
      TFE_TensorHandle* tensor = TFE_NewTensorHandle(t);
      inputs_array[i] = reinterpret_cast<jlong>(tensor);
    }
    call->env->ReleaseLongArrayElements(inputs, inputs_array, 0);
    return inputs;
  }

  // Process the return values by converting them back to TensorFlow tensors and adding them to the call outputs.
  void ProcessOutputs(JVMCall* call, jlongArray call_outputs, TF_Status* status) {
    call->outputs.clear();
    jsize n = call->env->GetArrayLength(call_outputs);
    jlong* outputs_array = call->env->GetLongArrayElements(call_outputs, nullptr);
    for (int i = 0; i < n; ++i) {
      auto* h = require_handle<TFE_TensorHandle>(call->env, outputs_array[i], "output");
      if (h == nullptr) {
        status->status = errors::InvalidArgument("Could not obtain tensor handle to one of the outputs.");
        return;
      }
      const Tensor* t = TFE_TensorHandleUnderlyingTensorInHostMemory(h, status);
      if (!status->status.ok()) return;
      call->outputs.push_back(*t);
    }
    call->env->ReleaseLongArrayElements(call_outputs, outputs_array, 0);
  }

  // Calls the registered JVM function through the registry.
  Status CallJVMFunction(JVMCall* call) {
    // Prepare the call arguments.
    jlongArray call_inputs = MakeInputs(call);

    // Invoke the registry 'call' method.
    auto outputs = (jlongArray) call->env->CallStaticObjectMethod(
        call->registry, call->call_method_id, call->id, call_inputs);
    if (outputs == nullptr) {
      return errors::Unknown("Failed to run JVM callback function.");
    }

    // Process the return values and convert them back to TensorFlow tensors.
    auto* status = new TF_Status;
    ProcessOutputs(call, outputs, status);
    return status->status;
  }
}  // namespace

class JVMCallbackOp : public OpKernel {
public:
  explicit JVMCallbackOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("id", &id_));
    std::string jvm_pointer;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("jvm_pointer", &jvm_pointer));
    jvm_ = pointerFromString<JavaVM*>(jvm_pointer);
    JNIEnv* env;
    jint status = jvm_->AttachCurrentThread((void **) &env, nullptr);
    assert(status == JNI_OK);
    string registry_class_name_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("registry_class_name", &registry_class_name_));
    registry_ = env->FindClass(registry_class_name_.c_str());
    call_method_id_ = env->GetStaticMethodID(registry_, "call", "(I[J)[J");
    if (call_method_id_ == nullptr) {
      ctx->CtxFailure(errors::InvalidArgument("Missing JVM registry 'call' method."));
      return;
    }
  }

  void Compute(OpKernelContext* ctx) override {
    JNIEnv* env;
    jint status = jvm_->AttachCurrentThread((void**) &env, nullptr);
    assert(status == JNI_OK);

    JVMCall call;
    call.env = env;
    call.registry = registry_;
    call.call_method_id = call_method_id_;
    call.id = id_;
    for (int i = 0; i < ctx->num_inputs(); ++i) {
      call.inputs.push_back(ctx->input(i));
    }

    Status s = CallJVMFunction(&call);
    status = jvm_->DetachCurrentThread();
    assert(status == JNI_OK);

    OP_REQUIRES_OK(ctx, s);

    OP_REQUIRES(ctx, static_cast<int32>(call.outputs.size()) == ctx->num_outputs(),
                errors::InvalidArgument(id_, " returns ", call.outputs.size(),
                                        " values, but expects to see ",
                                        ctx->num_outputs(), " values."));
    for (size_t i = 0; i < call.outputs.size(); ++i) {
      const auto& t = call.outputs[i];
      OP_REQUIRES(
          ctx, t.dtype() == output_type(i),
          errors::InvalidArgument(i, "-th value returned by ", id_, " is ",
                                  DataTypeString(t.dtype()), ", but expects ",
                                  DataTypeString(output_type(i))));
      ctx->set_output(i, t);
    }
  }

private:
  int id_;
  JavaVM* jvm_;
  jclass registry_;
  jmethodID call_method_id_;

  TF_DISALLOW_COPY_AND_ASSIGN(JVMCallbackOp);
};

REGISTER_KERNEL_BUILDER(Name("JVMCallback").Device(DEVICE_CPU), JVMCallbackOp);
REGISTER_KERNEL_BUILDER(Name("JVMCallbackStateless").Device(DEVICE_CPU), JVMCallbackOp);

}  // namespace tensorflow
