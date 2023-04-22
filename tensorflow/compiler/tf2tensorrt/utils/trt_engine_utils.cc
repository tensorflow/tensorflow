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

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_execution_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

using absl::StrCat;

ExecutionContext ExecutionContext::Create(nvinfer1::ICudaEngine* cuda_engine) {
  nvinfer1::IExecutionContext* execution_context =
      cuda_engine->createExecutionContextWithoutDeviceMemory();
  return ExecutionContext(execution_context);
}

Status GetTrtBindingShape(const nvinfer1::ICudaEngine* cuda_engine,
                          const nvinfer1::IExecutionContext* execution_context,
                          int binding_index, bool use_implicit_batch,
                          int batch_size, TensorShape& shape) {
  nvinfer1::Dims dims;
  if (use_implicit_batch) {
    dims = cuda_engine->getBindingDimensions(binding_index);
  } else {
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    // Get dims from context instead of engine in explicit batch mode because
    // the engine might have dynamic shapes.
    dims = execution_context->getBindingDimensions(binding_index);
    if (dims.nbDims == -1) {
      // Invalid dimensions. There can be multiple reasons for this. If we have
      // incompatible input shapes (network invalid for the current profile)
      // that can trigger this error.
      return errors::Internal(
          "Binding index out of range. This can happen if profile is not set, "
          "or the network is invalid for the current profile.");
    }
#else
    return errors::Internal(
        "Explicit batch mode is only supported with TensorRT 6 and above.");
#endif
  }
  TF_RETURN_IF_ERROR(TrtDimsToTensorShape(
      dims, &shape,
      use_implicit_batch ? absl::optional<int>(batch_size) : absl::nullopt));
  return Status::OK();
}

Status GetTrtBindingIndex(const char* tensor_name, int profile_index,
                          const nvinfer1::ICudaEngine* cuda_engine,
                          int* binding_index) {
  // If the engine has been built for K profiles, the first getNbBindings() / K
  // bindings are used by profile number 0, the following getNbBindings() / K
  // bindings are used by profile number 1 etc.
  //
  // GetBindingIndex(tensor_name) returns the binding index for the progile 0.
  // We can also consider it as a "binding_index_within_profile".
  *binding_index = cuda_engine->getBindingIndex(tensor_name);
  if (*binding_index == -1) {
    const string msg = StrCat("Input node ", tensor_name, " not found");
    LOG(ERROR) << msg;
    return errors::NotFound(msg);
  }
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
  int n_profiles = cuda_engine->getNbOptimizationProfiles();
#else
  int n_profiles = 1;
#endif
  // If we have more then one optimization profile, then we need to shift the
  // binding index according to the following formula:
  // binding_index_within_engine = binding_index_within_profile +
  //                               profile_index * bindings_per_profile
  const int bindings_per_profile = cuda_engine->getNbBindings() / n_profiles;
  *binding_index = *binding_index + profile_index * bindings_per_profile;
  return Status::OK();
}

Status SetTrtEngineInputs(nvinfer1::ICudaEngine* cuda_engine,
                          nvinfer1::IExecutionContext* execution_context,
                          const int trt_profile_idx,
                          std::vector<void*>& buffers, bool use_implicit_batch,
                          int num_batch,
                          const TrtShapeOptimizationProfile& profiles,
                          OpKernelContext* ctx, const DataVec* input_vec) {
  int n_inputs = ctx ? ctx->num_inputs() : (input_vec ? input_vec->size() : 0);
  // Setup engine inputs.
  for (int i = 0; i < n_inputs; i++) {
    const string input_name =
        ctx ? StrCat(IONamePrefixes::kInputPHName, i) : input_vec->at(i).name;
    int binding_index;
    TF_RETURN_IF_ERROR(GetTrtBindingIndex(input_name.c_str(), trt_profile_idx,
                                          cuda_engine, &binding_index));
    const Tensor& input_tensor = ctx ? ctx->input(i) : input_vec->at(i).tensor;
    const TensorShape& input_shape = input_tensor.shape();

    if (use_implicit_batch && ctx) {
      // Ensure all inputs have the same batch size
      if (num_batch != input_shape.dim_size(0)) {
        const string msg =
            StrCat("Input data has inconsistent batch size: ", num_batch,
                   " vs ", input_shape.dim_size(0));
        return errors::NotFound(msg);
      }
    }
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    // Set known input dimensions. This is necessary because TRT network
    // could be made with dynamic dimensions.
    if (!use_implicit_batch) {
      TF_RETURN_IF_ERROR(profiles.SetInputShapeBinding(
          i, binding_index, cuda_engine, execution_context));

      if (cuda_engine->isExecutionBinding(binding_index)) {
        nvinfer1::Dims trt_dims;
        TF_RETURN_IF_ERROR(TensorShapeToTrtDims(input_shape, false, &trt_dims));
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
    }
#endif
    // Setup input bindings.
    auto dtype = cuda_engine->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] =
            const_cast<float*>(input_tensor.flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        buffers[binding_index] =
            const_cast<Eigen::half*>(input_tensor.flat<Eigen::half>().data());
        break;
      case nvinfer1::DataType::kINT8:
        return errors::Internal("INT8 inputs are not supported yet!");
      case nvinfer1::DataType::kINT32:
        buffers[binding_index] =
            const_cast<int32*>(input_tensor.flat<int32>().data());
        break;
      default:
        return errors::Internal("Unknown TRT data type: ",
                                static_cast<int>(dtype));
    }
  }

#if IS_TRT_VERSION_GE(6, 0, 0, 0)
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
#endif
  return Status::OK();
}

Status SetTrtEngineOutputs(nvinfer1::ICudaEngine* cuda_engine,
                           nvinfer1::IExecutionContext* execution_context,
                           int trt_profile_idx, std::vector<void*>& buffers,
                           bool use_implicit_batch, int batch_size,
                           OpKernelContext* ctx, DataVec* outputs) {
  // Either one of ctx or outpus should be specified
  int n_outputs = ctx ? ctx->num_outputs() : (outputs ? outputs->size() : 0);
  for (int i = 0; i < n_outputs; i++) {
    const string output_name =
        ctx ? StrCat(IONamePrefixes::kOutputPHName, i) : outputs->at(i).name;
    int binding_index;
    TF_RETURN_IF_ERROR(GetTrtBindingIndex(output_name.c_str(), trt_profile_idx,
                                          cuda_engine, &binding_index));

    // Get TRT output shapes for allocating output memory.
    TensorShape output_shape;
    TF_RETURN_IF_ERROR(GetTrtBindingShape(cuda_engine, execution_context,
                                          binding_index, use_implicit_batch,
                                          batch_size, output_shape));

    // Allocate output tensor of TRTEngineOp.
    Tensor* output_tensor = nullptr;
    if (ctx) {
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

    // Setup output bindings.
    auto dtype = cuda_engine->getBindingDataType(binding_index);
    switch (dtype) {
      case nvinfer1::DataType::kFLOAT:
        buffers[binding_index] =
            const_cast<float*>(output_tensor->flat<float>().data());
        break;
      case nvinfer1::DataType::kHALF:
        buffers[binding_index] =
            const_cast<Eigen::half*>(output_tensor->flat<Eigen::half>().data());
        break;
      case nvinfer1::DataType::kINT8:
        return errors::Internal("int8 is not supported yet!");
      case nvinfer1::DataType::kINT32:
        buffers[binding_index] =
            const_cast<int32*>(output_tensor->flat<int32>().data());
        break;
      default:
        return errors::Internal("Unknown TRT data type: ",
                                static_cast<int>(dtype));
    }
  }
  return Status::OK();
}

Status TrtEnqueue(nvinfer1::IExecutionContext* execution_context,
                  std::vector<void*>& buffers, cudaStream_t stream,
                  bool use_implicit_batch, int batch_size) {
  bool ret = false;
  if (use_implicit_batch) {
    ret = execution_context->enqueue(batch_size, &buffers[0], stream, nullptr);
    VLOG(1) << "Called IExecutionContext::enqueue";
  } else {
#if IS_TRT_VERSION_GE(6, 0, 0, 0)
    ret = execution_context->enqueueV2(&buffers[0], stream, nullptr);
    VLOG(1) << "Called IExecutionContext::enqueueV2";
#else
    return errors::Internal(
        "Explicit batch mode is only supported with TensorRT 6 and above.");
#endif
  }
  if (!ret) {
    return errors::Internal("Failed to enqueue batch for TRT engine");
  }
  // Synchronization will be done by TF.
  return Status::OK();
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
