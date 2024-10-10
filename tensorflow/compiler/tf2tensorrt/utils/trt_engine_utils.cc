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

#include "tensorflow/compiler/tf2tensorrt/utils/trt_engine_utils.h"

#include <string>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_execution_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/profiler/lib/traceme.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

using absl::StrCat;

ExecutionContext ExecutionContext::Create(nvinfer1::ICudaEngine* cuda_engine) {
  bool has_int32_output = false;
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
  for (int i = 0; i < cuda_engine->getNbBindings(); i++) {
    if (!cuda_engine->bindingIsInput(i) &&
        cuda_engine->getBindingDataType(i) == nvinfer1::DataType::kINT32) {
#else
  for (int i = 0; i < cuda_engine->getNbIOTensors(); i++) {
    const char* tensor_name = cuda_engine->getIOTensorName(i);
    if (cuda_engine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT &&
        cuda_engine->getTensorDataType(tensor_name) == nvinfer1::DataType::kINT32) {
#endif
      has_int32_output = true;
      break;
    }
  }
  if (!IS_TRT_VERSION_GE(8, 0, 0, 0) && has_int32_output) {
    // TODO(nvbugs/3390469): Remove this workaround when the bug is fixed.
    nvinfer1::IExecutionContext* execution_context =
        cuda_engine->createExecutionContext();
    return ExecutionContext(execution_context, true);
  }

  nvinfer1::IExecutionContext* execution_context =
      cuda_engine->createExecutionContextWithoutDeviceMemory();
  return ExecutionContext(execution_context, false);
}

Status GetTrtBindingShape(const nvinfer1::ICudaEngine* cuda_engine,
                          const nvinfer1::IExecutionContext* execution_context,
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
                          int binding_index,
#else
                          const char* tensor_name,
#endif
                          bool use_implicit_batch, int batch_size,
                          TensorShape& shape) {
  tensorflow::profiler::TraceMe activity(
      "getBindingDimensions", tensorflow::profiler::TraceMeLevel::kInfo);
  nvinfer1::Dims dims =
      use_implicit_batch
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
          ? cuda_engine->getBindingDimensions(binding_index)
          : execution_context->getBindingDimensions(binding_index);
#else
          ? cuda_engine->getTensorShape(tensor_name)
          : execution_context->getTensorShape(tensor_name);
#endif
  if (!use_implicit_batch) {
    if (dims.nbDims == -1) {
      return errors::Internal(
          "Binding index out of range. This can happen if profile is not set, "
          "or the network is invalid for the current profile.");
    }
  }
  TF_RETURN_IF_ERROR(DimsAdapter(dims).TensorShape(
      &shape,
      use_implicit_batch ? std::optional<int>(batch_size) : std::nullopt));
  return OkStatus();
}

#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
Status SetupBindings(nvinfer1::ICudaEngine* cuda_engine, const Tensor& tensor,
                     std::vector<void*>& buffers, int binding_index) {
#else
Status SetupBindings(nvinfer1::ICudaEngine* cuda_engine,
                     nvinfer1::IExecutionContext* execution_context,
                     const Tensor& tensor, const char* tensor_name) {
#endif
  tensorflow::profiler::TraceMe activity(
      "SetBindingPointers", tensorflow::profiler::TraceMeLevel::kInfo);
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
  const auto dtype = cuda_engine->getBindingDataType(binding_index);
#else
  const auto dtype = cuda_engine->getTensorDataType(tensor_name);
#endif
  VLOG(2) << "<<<<<<<<< SetupBindings with dtype = " << (int)dtype;
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
      buffers[binding_index] = const_cast<float*>(tensor.flat<float>().data());
#else
      execution_context->setTensorAddress(
          tensor_name, const_cast<float*>(tensor.flat<float>().data()));
#endif
      break;
    case nvinfer1::DataType::kHALF:
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
      buffers[binding_index] =
          const_cast<Eigen::half*>(tensor.flat<Eigen::half>().data());
#else
      execution_context->setTensorAddress(
          tensor_name,
          const_cast<Eigen::half*>(tensor.flat<Eigen::half>().data()));
#endif
      break;
    case nvinfer1::DataType::kINT8:
      return errors::Internal("INT8 inputs are not supported yet!");
    case nvinfer1::DataType::kINT32:
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
      buffers[binding_index] = const_cast<int32*>(tensor.flat<int32>().data());
#else
      execution_context->setTensorAddress(
          tensor_name, const_cast<int32*>(tensor.flat<int32>().data()));
#endif
      break;
#if IS_TRT_VERSION_GE(8, 2, 0, 0)
    case nvinfer1::DataType::kBOOL:
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
      buffers[binding_index] = const_cast<bool*>(tensor.flat<bool>().data());
#else
      execution_context->setTensorAddress(
          tensor_name, const_cast<bool*>(tensor.flat<bool>().data()));
#endif
      break;
#endif
#if IS_TRT_VERSION_GE(8, 5, 0, 0)
    case nvinfer1::DataType::kUINT8:
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
      buffers[binding_index] = const_cast<uint8*>(tensor.flat<uint8>().data());
#else   // IS_TRT_VERSION_GE(10, 0, 0, 0)
      execution_context->setTensorAddress(
          tensor_name, const_cast<uint8*>(tensor.flat<uint8>().data()));
#endif  // !IS_TRT_VERSION_GE(10, 0, 0, 0)
      break;
#endif
#if IS_TRT_VERSION_GE(8, 6, 0, 0)
    case nvinfer1::DataType::kFP8:
      return errors::Internal("FP8 inputs are not supported yet!");
#endif
#if IS_TRT_VERSION_GE(10, 0, 0, 0)
    case nvinfer1::DataType::kBF16:
      return errors::Internal("BF16 inputs are not supported yet!");
    case nvinfer1::DataType::kINT64:
      return errors::Internal("INT64 inputs are not supported yet!");
    case nvinfer1::DataType::kINT4:
      return errors::Internal("INT4 inputs are not supported yet!");
#endif  // IS_TRT_VERSION_GE(10, 0, 0, 0)
    default:
      return errors::Internal("Unknown TRT data type: ",
                              static_cast<int>(dtype));
  }
  return OkStatus();
}

// Sets up bindings.
Status SetTrtEngineInputs(nvinfer1::ICudaEngine* cuda_engine,
                          nvinfer1::IExecutionContext* execution_context,
                          const int trt_profile_idx,
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
                          std::vector<void*>& buffers,
#endif
                          bool use_implicit_batch, int num_batch,
                          const TrtShapeOptimizationProfile& profiles,
                          OpKernelContext* ctx, const DataVec* input_vec) {
  tensorflow::profiler::TraceMe activity(
      "SetTrtEngineInputs", tensorflow::profiler::TraceMeLevel::kInfo);
  int n_inputs = ctx ? ctx->num_inputs() : (input_vec ? input_vec->size() : 0);
  // Setup engine inputs.
  for (int i = 0; i < n_inputs; i++) {
    const Tensor& input_tensor = ctx ? ctx->input(i) : input_vec->at(i).tensor;
    const TensorShape& input_shape = input_tensor.shape();

    // Skip resource inputs.
    if (input_tensor.dtype() == DataType::DT_RESOURCE) {
      continue;
    }

    const string input_name =
        ctx ? StrCat(IONamePrefixes::kInputPHName, i) : input_vec->at(i).name;
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
    int binding_index;
    Status status = GetTrtBindingIndex(input_name.c_str(), trt_profile_idx,
                                       cuda_engine, &binding_index);
    if (IS_TRT_VERSION_GE(8, 0, 0, 0)) {
      TF_RETURN_IF_ERROR(status);
    } else if (!status.ok()) {
      // Before TRT 8, an input tensor can be pruned if it is not used by the
      // network (e.g. only its shape is used, but the shape is already defined
      // by the optimization profile by setting min=max). nvbugs/3153064
      VLOG(2) << "Skipping pruned input " << input_name;
      continue;
    }
#endif  // !IS_TRT_VERSION_GE(10, 0, 0, 0)

    if (use_implicit_batch && ctx) {
      // Ensure all inputs have the same batch size
      if (num_batch != input_shape.dim_size(0)) {
        const string msg =
            StrCat("Input data has inconsistent batch size: ", num_batch,
                   " vs ", input_shape.dim_size(0));
        return errors::NotFound(msg);
      }
    }
    // Set known input dimensions. This is necessary because TRT network
    // could be made with dynamic dimensions.
    if (!use_implicit_batch) {
      TF_RETURN_IF_ERROR(profiles.SetInputShapeBinding(i,
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
                                                       binding_index,
#else
                                                       input_name.c_str(),
#endif
                                                       cuda_engine,
                                                       execution_context));

#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
      if (cuda_engine->isExecutionBinding(binding_index))
#else
      if (true)
#endif
      {
        tensorflow::profiler::TraceMe activity(
            "SetTrtEngineInputs::setBindingDimensions",
            tensorflow::profiler::TraceMeLevel::kInfo);
        auto adap = DimsAdapter::Create(input_shape);
        TRT_ENSURE_OK(adap);
        nvinfer1::Dims trt_dims = adap->AsTrtDims();
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
        if (execution_context->getBindingDimensions(binding_index) !=
            trt_dims) {
          VLOG(2) << "Setting binding dimensions for idx " << binding_index;
          bool ret =
              execution_context->setBindingDimensions(binding_index, trt_dims);
          if (!ret) {
            VLOG(2) << "Error setting engine input " << binding_index << " "
                    << DebugString(trt_dims);
            return errors::Internal(
                "Binding dimension does not fit selected profile.");
          }
        }
#else   // IS_TRT_VERSION_GE(10, 0, 0, 0)
        if (execution_context->getTensorShape(input_name.c_str()) != trt_dims) {
          VLOG(2) << "Setting binding dimensions for input " << input_name;
          bool ret =
              execution_context->setInputShape(input_name.c_str(), trt_dims);
          if (!ret) {
            VLOG(2) << "Error setting engine input " << input_name << " "
                    << DebugString(trt_dims);
            return errors::Internal(
                "Binding dimension does not fit selected profile.");
          }
        }
#endif  // IS_TRT_VERSION_GE(10, 0, 0, 0)
      }
    }
    // Setup input bindings.
    TF_RETURN_IF_ERROR(
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
        SetupBindings(cuda_engine, input_tensor, buffers, binding_index)
#else
        SetupBindings(cuda_engine, execution_context, input_tensor,
                      input_name.c_str())
#endif
    );
  }

  // Ensure all network dynamic dimensions (if any) are set in execution
  // context.
  if (!execution_context->allInputDimensionsSpecified()) {
    return errors::Internal(
        "Failed to set dimensions for all dynamic input tensors");
  }
  if (!execution_context->allInputShapesSpecified()) {
    return errors::Internal(
        "Failed to set dimensions for all shape input tensors.");
  }
  return OkStatus();
}

Status SetTrtEngineOutputs(nvinfer1::ICudaEngine* cuda_engine,
                           nvinfer1::IExecutionContext* execution_context,
                           int trt_profile_idx,
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
                           std::vector<void*>& buffers,
#endif
                           bool use_implicit_batch, int batch_size,
                           OpKernelContext* ctx, DataVec* outputs) {
  tensorflow::profiler::TraceMe activity(
      "SetTrtEngineOutputs", tensorflow::profiler::TraceMeLevel::kInfo);
  // Either one of ctx or outpus should be specified
  int n_outputs = ctx ? ctx->num_outputs() : (outputs ? outputs->size() : 0);
  for (int i = 0; i < n_outputs; i++) {
    const string output_name =
        ctx ? StrCat(IONamePrefixes::kOutputPHName, i) : outputs->at(i).name;
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
    int binding_index;
    TF_RETURN_IF_ERROR(GetTrtBindingIndex(output_name.c_str(), trt_profile_idx,
                                          cuda_engine, &binding_index));
#endif

    // Get TRT output shapes for allocating output memory.
    TensorShape output_shape;
    TF_RETURN_IF_ERROR(GetTrtBindingShape(cuda_engine, execution_context,
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
                                          binding_index,
#else
                                          output_name.c_str(),
#endif
                                          use_implicit_batch, batch_size,
                                          output_shape));

    // Allocate output tensor of TRTEngineOp.
    Tensor* output_tensor = nullptr;
    if (ctx) {
      tensorflow::profiler::TraceMe activity(
          "AllocateOutput", tensorflow::profiler::TraceMeLevel::kInfo);
      TF_RETURN_IF_ERROR(ctx->allocate_output(i, output_shape, &output_tensor));
    } else {
      // This path is used for unit tests. The tensor is already allocated.
      // Its shape is not necessarily set correctly, we fix that.
      VLOG(2) << "Applying shape " << output_shape.DebugString()
              << " on output.";
      output_tensor = &(outputs->at(i).tensor);
      bool status = output_tensor->CopyFrom(*output_tensor, output_shape);
      if (!status) {
        return errors::Internal(
            "Buffer size (", output_tensor->NumElements(),
            ") do not match while reshaping output tensors to shape ",
            output_shape.DebugString());
      }
    }

    // Set up output bindings.
    TF_RETURN_IF_ERROR(
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
        SetupBindings(cuda_engine, *output_tensor, buffers, binding_index));
#else
        SetupBindings(cuda_engine, execution_context, *output_tensor,
                      output_name.c_str()));
#endif
  }
  return OkStatus();
}

Status TrtEnqueue(nvinfer1::IExecutionContext* execution_context,
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
                  std::vector<void*>& buffers,
#endif
                  cudaStream_t stream,
                  bool use_implicit_batch, int batch_size) {
  tensorflow::profiler::TraceMe activity(
      "TrtEnqueue", tensorflow::profiler::TraceMeLevel::kInfo);
  bool ret = false;
  if (use_implicit_batch) {
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
    ret = execution_context->enqueue(batch_size, &buffers[0], stream, nullptr);
    VLOG(1) << "Called IExecutionContext::enqueue";
#else
    return errors::Internal("Implicit batch is not supported with TensorRT >=10");
#endif
  } else {
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
    ret = execution_context->enqueueV2(&buffers[0], stream, nullptr);
    VLOG(1) << "Called IExecutionContext::enqueueV2";
#else
    ret = execution_context->enqueueV3(stream);
    VLOG(1) << "Called IExecutionContext::enqueueV3";
#endif
  }
  if (!ret) {
    return errors::Internal("Failed to enqueue batch for TRT engine");
  }
  // Synchronization will be done by TF.
  return OkStatus();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
