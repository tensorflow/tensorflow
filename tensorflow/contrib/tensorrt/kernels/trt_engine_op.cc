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
#include "tensorflow/contrib/tensorrt/kernels/trt_engine_op.h"
#include <cuda_runtime_api.h>
#include <sstream>
#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"


namespace tensorflow {
static ::tensorflow::tensorrt::Logger gLogger;

using namespace nvinfer1;

namespace tensorrt {

TRTEngineOp::TRTEngineOp(OpKernelConstruction* context) : OpKernel(context) {
  // read serialized_engine
  std::string serialized_engine;
  OP_REQUIRES_OK(context,
                 context->GetAttr("serialized_engine", &serialized_engine));

  // register input output node name in trt_sub_graph
  OP_REQUIRES_OK(context, context->GetAttr("input_nodes", &input_nodes_));
  OP_REQUIRES_OK(context, context->GetAttr("output_nodes", &output_nodes_));

  // TODO(samikama) runtime should be taken from a resourcemanager as well.
  //  Only engine should be in the op and context and runtime should be taken
  //  from resourcemanager
  IRuntime* infer = createInferRuntime(gLogger);
  trt_engine_ptr_.reset(infer->deserializeCudaEngine(
      serialized_engine.c_str(), serialized_engine.size(), nullptr));

  trt_execution_context_ptr_.reset(trt_engine_ptr_->createExecutionContext());
  // runtime is safe to delete after engine creation
  infer->destroy();
  std::stringstream oss;

}

void TRTEngineOp::Compute(OpKernelContext* context) {
  int nbBindings = context->num_inputs() + context->num_outputs();
  // TODO(jjsjann123) multiple input/output
  std::vector<void*> buffers(nbBindings);

  size_t bindingIndex;
  int nbBatch = 0;
  bool valid = true;
  for (int i = 0; i < context->num_inputs(); i++) {
    // Grab the input tensor
    bindingIndex = trt_engine_ptr_->getBindingIndex(input_nodes_[i].c_str());

    const Tensor& input_tensor = context->input(i);
    const TensorShape& input_shape = input_tensor.shape();
    if (i == 0) {
      nbBatch = input_shape.dim_size(0);
    } else if (nbBatch != input_shape.dim_size(0)) {
      valid = false;
      break;
    }
    switch (trt_engine_ptr_->getBindingDataType(bindingIndex)) {
      case nvinfer1::DataType::kFLOAT:
        buffers[bindingIndex] = (void*)(input_tensor.flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(FATAL) << "half size is not supported yet!";
        break;
      case nvinfer1::DataType::kINT8:
        LOG(FATAL) << "int8 is not supported yet!";
        break;
    }
  }

  // Might want a different way to inform the user of batch size inconsistency
  if (!valid) LOG(WARNING) << "input data inconsistent batch size";

  for (int i = 0; i < static_cast<int>(output_nodes_.size()); i++) {
    // This is bad that we have to reallocate output buffer every run.
    // Create an output tensor
    bindingIndex = trt_engine_ptr_->getBindingIndex(output_nodes_[i].c_str());
    Tensor* output_tensor = NULL;

    TensorShape output_shape;
    if (bindingIndex != -1) {
      auto dims = trt_engine_ptr_->getBindingDimensions(bindingIndex);
      std::vector<int> trt_shape(dims.nbDims + 1);
      trt_shape[0] = nbBatch;
      for (int j = 0; j < dims.nbDims; j++) trt_shape[j + 1] = dims.d[j];
      TensorShapeUtils::MakeShape(trt_shape.data(), trt_shape.size(),
                                  &output_shape);
    } else {
      LOG(FATAL) << "output node not found, at " << output_nodes_[i];
      break;
    }

    OP_REQUIRES_OK(context,
                   context->allocate_output(i, output_shape, &output_tensor));
    switch (trt_engine_ptr_->getBindingDataType(bindingIndex)) {
      case nvinfer1::DataType::kFLOAT:
        buffers[bindingIndex] =
            reinterpret_cast<void*>(output_tensor->flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(FATAL) << "half size is not supported yet!";
        break;
      case nvinfer1::DataType::kINT8:
        LOG(FATAL) << "int8 is not supported yet!";
        break;
    }
  }
  // copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->CudaStreamMemberHack()));

  // execution handled by TF since we are getting stream from TF.
  trt_execution_context_ptr_->enqueue(nbBatch, &buffers[0], *stream, nullptr);
}

REGISTER_KERNEL_BUILDER(Name("TRTEngineOp").Device(DEVICE_GPU), TRTEngineOp);
}  // namespace tensorrt
}  // namespace tensorflow
