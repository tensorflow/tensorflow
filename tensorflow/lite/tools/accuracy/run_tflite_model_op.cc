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

#include <memory>
#include <vector>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/tools/accuracy/utils.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace {
Status ValidateInputsMatch(const OpInputList& input_tensors,
                           const tflite::Interpreter& interpreter) {
  std::vector<int> tflite_tensor_indices = interpreter.inputs();
  if (tflite_tensor_indices.size() != input_tensors.size()) {
    return errors::InvalidArgument(
        "size mismatch, interpreter size: ", tflite_tensor_indices.size(),
        " actual: ", input_tensors.size());
  }

  for (int i = 0; i < input_tensors.size(); i++) {
    const TfLiteTensor* tflite_tensor =
        interpreter.tensor(tflite_tensor_indices[i]);
    if (tflite_tensor == nullptr) {
      return errors::InvalidArgument("Tensor is null at index: ", i);
    }

    const Tensor& tensor = input_tensors[i];
    auto i_type = metrics::utils::GetTFDataType(tflite_tensor->type);
    auto i_shape = metrics::utils::GetTFLiteTensorShape(*tflite_tensor);
    if (i_type != tensor.dtype()) {
      return errors::InvalidArgument("Data types mismatch for tensors: ", i,
                                     " expected: ", i_type,
                                     " got: ", tensor.dtype());
    }

    if (i_shape != tensor.shape()) {
      return errors::InvalidArgument("Data shapes mismatch for tensors: ", i,
                                     " expected: ", i_shape,
                                     " got: ", tensor.shape());
    }
  }

  return Status::OK();
}

}  // namespace

class RunTFLiteModelOp : public OpKernel {
 public:
  explicit RunTFLiteModelOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    string model_file_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_file_path", &model_file_path));
    model_ = tflite::FlatBufferModel::BuildFromFile(model_file_path.data());
    OP_REQUIRES(ctx, model_,
                errors::InvalidArgument(
                    "Model loading failed. Invalid model file path: ",
                    model_file_path));
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    OP_REQUIRES(ctx, interpreter_,
                errors::Internal("Interpreter creation failed."));
  }

  void Compute(OpKernelContext* context) override {
    OpInputList input_tensors;
    OP_REQUIRES_OK(context, context->input_list("model_input", &input_tensors));

    OP_REQUIRES_OK(context, ValidateInputsMatch(input_tensors, *interpreter_));
    OpOutputList output_tensors;
    OP_REQUIRES_OK(context,
                   context->output_list("model_output", &output_tensors));
    auto tfl_outputs = interpreter_->outputs();
    OP_REQUIRES(context, output_tensors.size() == tfl_outputs.size(),
                errors::InvalidArgument(
                    "Invalid output size, expected: ", tfl_outputs.size(),
                    " got: ", output_tensors.size()));
    for (int i = 0; i < output_tensors.size(); i++) {
      DataType tfl_type = metrics::utils::GetTFDataType(
          interpreter_->tensor(tfl_outputs[i])->type);
      DataType otype = output_tensors.expected_output_dtype(i);
      OP_REQUIRES(
          context, tfl_type == otype,
          errors::InvalidArgument("Invalid data type for output at index: ", i,
                                  " expected: ", tfl_type, " got: ", otype));
    }

    auto allocation_status = interpreter_->AllocateTensors();
    OP_REQUIRES(context, allocation_status == kTfLiteOk,
                errors::Internal("Unable to allocate tensors."));
    for (int i = 0; i < input_tensors.size(); i++) {
      const int tfl_index = interpreter_->inputs()[i];
      TfLiteTensor* tflite_tensor = interpreter_->tensor(tfl_index);
      auto tensor_bytes = input_tensors[i].tensor_data();
      OP_REQUIRES(context, tflite_tensor->bytes == tensor_bytes.size(),
                  errors::InvalidArgument(
                      "Size mismatch, expected: ", tflite_tensor->bytes,
                      " got: ", tensor_bytes.size()));
      std::memcpy(tflite_tensor->data.raw, tensor_bytes.data(),
                  tensor_bytes.size());
    }
    auto invocation_status = interpreter_->Invoke();
    OP_REQUIRES(context, invocation_status == kTfLiteOk,
                errors::Internal("Interpreter invocation failed."));
    for (int i = 0; i < output_tensors.size(); i++) {
      auto tfl_tensor = interpreter_->tensor(tfl_outputs[i]);
      TensorShape shape = metrics::utils::GetTFLiteTensorShape(*tfl_tensor);
      Tensor* output = nullptr;
      OP_REQUIRES_OK(context, output_tensors.allocate(i, shape, &output));
      auto tensor_bytes = output->tensor_data();
      OP_REQUIRES(context, tensor_bytes.size() == tfl_tensor->bytes,
                  errors::Internal("Invalid size"));
      std::memcpy(const_cast<char*>(tensor_bytes.data()), tfl_tensor->data.raw,
                  tfl_tensor->bytes);
    }
  }

 private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

REGISTER_KERNEL_BUILDER(Name("RunTFLiteModel").Device(DEVICE_CPU),
                        RunTFLiteModelOp);

REGISTER_OP("RunTFLiteModel")
    .Input("model_input: input_type")
    .Output("model_output: output_type")
    .Attr("model_file_path: string")
    .Attr("input_type : list(type)")
    .Attr("output_type: list(type)")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // TODO(shashishekhar): Infer the correct shape based on output_type and
      // maybe another attribute.
      return shape_inference::UnknownShape(c);
    });

}  // namespace tensorflow
