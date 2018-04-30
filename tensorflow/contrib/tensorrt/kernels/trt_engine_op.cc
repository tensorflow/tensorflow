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
#include "tensorflow/contrib/tensorrt/plugin/trt_plugin_factory.h"

#include "tensorflow/contrib/tensorrt/log/trt_logger.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT
#include "cuda/include/cuda_runtime_api.h"

namespace tensorflow {
static ::tensorflow::tensorrt::Logger logger;
using IRuntime = nvinfer1::IRuntime;
using Dims = nvinfer1::Dims;

namespace tensorrt {

TRTEngineOp::TRTEngineOp(OpKernelConstruction* context) : OpKernel(context) {
  // read serialized_engine
  string serialized_engine;
  OP_REQUIRES_OK(context,
                 context->GetAttr("serialized_engine", &serialized_engine));

  // register input output node name in trt_sub_graph
  OP_REQUIRES_OK(context, context->GetAttr("input_nodes", &input_nodes_));
  OP_REQUIRES_OK(context, context->GetAttr("output_nodes", &output_nodes_));

  // TODO(samikama) runtime should be taken from a resourcemanager as well.
  // Only engine should be in the op and context and runtime should be taken
  // from resourcemanager
  // TODO(jie): cudaSetDevice make sure trt engine is allocated on the same
  // gpu where the input/output is also located.
  int gpu_id = context->device()->tensorflow_gpu_device_info()->gpu_id;
  cudaSetDevice(gpu_id);
  int device;
  cudaGetDevice(&device);
  if (gpu_id != device) LOG(FATAL) << "set device failed!";

  // TODO(samikama) runtime should be taken from a resourcemanager as well.
  // Only engine should be in the op and context and runtime should be taken
  // from resourcemanager

  IRuntime* infer = nvinfer1::createInferRuntime(logger);
  trt_engine_ptr_.reset(infer->deserializeCudaEngine(
      serialized_engine.c_str(), serialized_engine.size(),
      PluginFactoryTensorRT::GetInstance()));
  trt_execution_context_ptr_.reset(trt_engine_ptr_->createExecutionContext());
  // Runtime is safe to delete after engine creation
  infer->destroy();
}

void TRTEngineOp::Compute(OpKernelContext* context) {
  int num_binding = context->num_inputs() + context->num_outputs();
  std::vector<void*> buffers(num_binding);

  size_t binding_index;
  int num_batch = 0;
  for (int i = 0; i < context->num_inputs(); i++) {
    // Grab the input tensor
    binding_index = trt_engine_ptr_->getBindingIndex(input_nodes_[i].c_str());

    const Tensor& input_tensor = context->input(i);
    const TensorShape& input_shape = input_tensor.shape();
    if (i == 0) {
      num_batch = input_shape.dim_size(0);
      if (num_batch > trt_engine_ptr_->getMaxBatchSize()) {
        LOG(FATAL) << "input tensor batch larger than max_batch_size: "
                   << trt_engine_ptr_->getMaxBatchSize();
      }
    } else if (num_batch != input_shape.dim_size(0)) {
      LOG(FATAL) << "input data inconsistent batch size";
      break;
    }
    auto dtype = trt_engine_ptr_->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] = (void*)(input_tensor.flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(FATAL) << "half size is not supported yet!";
        break;
      case nvinfer1::DataType::kINT8:
        LOG(FATAL) << "int8 is not supported yet!";
        break;
      default:
        LOG(FATAL) << "Unknown data type: " << int(dtype);
        break;
    }
  }

  for (int i = 0; i < static_cast<int>(output_nodes_.size()); i++) {
    // This is bad that we have to reallocate output buffer every run.
    // Create an output tensor
    binding_index = trt_engine_ptr_->getBindingIndex(output_nodes_[i].c_str());
    Tensor* output_tensor = nullptr;

    TensorShape output_shape;
    if (binding_index != -1) {
      auto dims = trt_engine_ptr_->getBindingDimensions(binding_index);
      std::vector<int> trt_shape(dims.nbDims + 1);
      trt_shape[0] = num_batch;
      for (int j = 0; j < dims.nbDims; j++) trt_shape[j + 1] = dims.d[j];
      OP_REQUIRES_OK(context,
                     TensorShapeUtils::MakeShape(
                         trt_shape.data(), trt_shape.size(), &output_shape));
    } else {
      LOG(FATAL) << "output node not found, at " << output_nodes_[i];
      break;
    }

    OP_REQUIRES_OK(context,
                   context->allocate_output(i, output_shape, &output_tensor));
    auto dtype = trt_engine_ptr_->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] =
            reinterpret_cast<void*>(output_tensor->flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        LOG(FATAL) << "half size is not supported yet!";
        break;
      case nvinfer1::DataType::kINT8:
        LOG(FATAL) << "int8 is not supported yet!";
        break;
      default:
        LOG(FATAL) << "Unknown data type: " << int(dtype);
        break;
    }
  }
  // copied from cuda_kernel_helper since it seems only valid in *.cu.cc files
  const cudaStream_t* stream = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->CudaStreamMemberHack()));

  // TODO(jie): trt enqueue does not return error
  auto ret = trt_execution_context_ptr_->enqueue(num_batch, &buffers[0],
                                                 *stream, nullptr);
  VLOG(2) << "enqueue returns: " << ret;
  // sync should be done by TF.
}

REGISTER_KERNEL_BUILDER(Name("TRTEngineOp").Device(DEVICE_GPU), TRTEngineOp);

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
