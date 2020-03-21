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

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <xnnpack.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace xnnpack {
namespace {

// Forward declaration.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
 public:
  explicit Delegate(const TfLiteXNNPackDelegateOptions* options) {
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
    if (options != nullptr && options->num_threads > 1) {
      threadpool_.reset(
          pthreadpool_create(static_cast<size_t>(options->num_threads)));
    }
#endif
  }

  TfLiteDelegate* tflite_delegate() { return &delegate_; }

  pthreadpool_t threadpool() {
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    return nullptr;
#else
    return threadpool_.get();
#endif
  }

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
  // Thread pool with smart-pointer for lifetime management.
  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool_{
      nullptr, &pthreadpool_destroy};
#endif
};

class Subgraph {
 public:
  static Subgraph* Create(TfLiteContext* context,
                          const TfLiteDelegateParams* params,
                          pthreadpool_t threadpool) {
    // Convert subgraph inputs and outputs to hash sets for faster lookup.
    const std::unordered_set<int> inputs(
        &params->input_tensors->data[0],
        &params->input_tensors->data[params->input_tensors->size]);
    const std::unordered_set<int> outputs(
        &params->output_tensors->data[0],
        &params->output_tensors->data[params->output_tensors->size]);
    std::unordered_set<int> externals(outputs);

    TfLiteIntArray* execution_plan;
    if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
      return nullptr;
    }

    xnn_subgraph_t subgraph_ptr = nullptr;
    xnn_status status = xnn_create_subgraph(
        /*external_value_ids=*/context->tensors_size, /*flags=*/0,
        &subgraph_ptr);
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "failed to create XNNPACK subgraph");
      return nullptr;
    }

    // Smart pointer to automatically release subgraph on exit.
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
        subgraph_ptr, &xnn_delete_subgraph);

    // Detect which tensors are used as inputs or outputs of any subgraph nodes.
    // -1 denotes tensor not used in the subgraph. These indexes will be
    // filtered out and removed later.
    std::vector<int> tensors(context->tensors_size, -1);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context,
                                          params->nodes_to_replace->data[i],
                                          &node, &registration) != kTfLiteOk) {
        return nullptr;
      }

      for (int k = 0; k < node->inputs->size; k++) {
        const int t = node->inputs->data[k];
        tensors[t] = t;
      }
      for (int k = 0; k < node->outputs->size; k++) {
        const int t = node->outputs->data[k];
        tensors[t] = t;
      }
    }
    // Filter out and remove -1 (unused) indexes.
    tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                 [](int i) { return i < 0; }),
                  tensors.end());
    std::sort(tensors.begin(), tensors.end());

    // XNNPACK Value IDs for TFLite tensors
    std::vector<uint32_t> xnnpack_tensors(tensors.back() + 1);
    for (int t : tensors) {
      if (context->tensors[t].type != kTfLiteFloat32) {
        TF_LITE_KERNEL_LOG(
            context,
            "unsupported datatype (%s) of tensor %d in XNNPACK delegate",
            TfLiteTypeGetName(context->tensors[t].type), t);
        return nullptr;
      }

      uint32_t flags = 0;
      const void* data = nullptr;
      if (context->tensors[t].allocation_type == kTfLiteMmapRo) {
        data = context->tensors[t].data.raw_const;
      }
      if (inputs.count(t) != 0) {
        flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
        if (data == nullptr) {
          externals.insert(t);
        }
      }
      if (outputs.count(t) != 0) {
        flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
      }

      std::vector<size_t> dims(
          &context->tensors[t].dims->data[0],
          &context->tensors[t].dims->data[context->tensors[t].dims->size]);

      const xnn_status status = xnn_define_tensor_value(
          subgraph.get(), xnn_datatype_fp32, dims.size(), dims.data(), data,
          static_cast<uint32_t>(t), flags, &xnnpack_tensors[t]);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(context,
                           "failed to create XNNPACK Value for tensor %d", t);
        return nullptr;
      }
    }

    // Create XNNPACK nodes for TFLite delegate nodes
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context,
                                          params->nodes_to_replace->data[i],
                                          &node, &registration) != kTfLiteOk) {
        return nullptr;
      }

      if (VisitNode(subgraph.get(), context, registration, node, i,
                    xnnpack_tensors) != kTfLiteOk) {
        return nullptr;
      }
    }

    xnn_runtime_t runtime_ptr = nullptr;
    status = xnn_create_runtime_v2(subgraph.get(), threadpool, /*flags=*/0,
                                   &runtime_ptr);
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "failed to create XNNPACK runtime");
      return nullptr;
    }

    return new Subgraph(runtime_ptr, std::move(externals));
  }

  TfLiteStatus Prepare(TfLiteContext* context) { return kTfLiteOk; }

  TfLiteStatus Invoke(TfLiteContext* context) {
    if (first_run_) {
      std::vector<xnn_external_value> external_values;
      for (int t : externals_) {
        xnn_external_value value = {0};
        value.id = static_cast<uint32_t>(t);
        value.data = context->tensors[t].data.raw;
        external_values.push_back(value);
      }

      const xnn_status status = xnn_setup_runtime(
          runtime_.get(), external_values.size(), external_values.data());
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(context, "failed to setup XNNPACK runtime");
        return kTfLiteError;
      }

      first_run_ = false;
    }

    const xnn_status status = xnn_invoke_runtime(runtime_.get());
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "failed to invoke XNNPACK runtime");
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CalculatePadding(TfLiteContext* context,
                                       TfLitePadding padding, uint32_t* flags,
                                       int node_index) {
    switch (padding) {
      case kTfLitePaddingSame: {
        *flags = XNN_FLAG_TENSORFLOW_SAME_PADDING;
        return kTfLiteOk;
      }
      case kTfLitePaddingValid:
        *flags = 0;
        return kTfLiteOk;
      default:
        if (context != nullptr) {
          TF_LITE_KERNEL_LOG(context, "invalid padding mode (%d) in node #%d",
                             static_cast<int>(padding), node_index);
        }
        return kTfLiteError;
    }
  }

  static TfLiteStatus ConvertActivationToOutputRange(
      TfLiteContext* context, int node_index, TfLiteFusedActivation activation,
      float* output_min, float* output_max) {
    switch (activation) {
      case kTfLiteActNone:
        *output_min = -std::numeric_limits<float>::infinity();
        *output_max = +std::numeric_limits<float>::infinity();
        return kTfLiteOk;
      case kTfLiteActRelu:
        *output_min = 0.0f;
        *output_max = +std::numeric_limits<float>::infinity();
        return kTfLiteOk;
      case kTfLiteActRelu1:
        *output_min = -1.0f;
        *output_max = +1.0f;
        return kTfLiteOk;
      case kTfLiteActRelu6:
        *output_min = 0.0f;
        *output_max = 6.0f;
        return kTfLiteOk;
      case kTfLiteActTanh:
        if (context != nullptr) {
          TF_LITE_KERNEL_LOG(context,
                             "unsupported fused activation (Tanh) in node #%d",
                             node_index);
        }
        return kTfLiteError;
      case kTfLiteActSignBit:
        if (context != nullptr) {
          TF_LITE_KERNEL_LOG(context,
                             "unsupported fused activation (Sign) in node #%d",
                             node_index);
        }
        return kTfLiteError;
      case kTfLiteActSigmoid:
        if (context != nullptr) {
          TF_LITE_KERNEL_LOG(
              context, "unsupported fused activation (Sigmoid) in node #%d",
              node_index);
        }
        return kTfLiteError;
      default:
        if (context != nullptr) {
          TF_LITE_KERNEL_LOG(context,
                             "invalid fused activation (%d) in node #%d",
                             static_cast<int>(activation), node_index);
        }
        return kTfLiteError;
    }
  }

  static TfLiteStatus CheckConvolutionParams(TfLiteContext* context,
                                             const TfLiteConvParams* params,
                                             int node_index) {
    if (params->stride_width <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                           params->stride_width, node_index);
      }
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                           params->stride_height, node_index);
      }
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "invalid dilation width factor %d in node #%d",
                           params->dilation_width_factor, node_index);
      }
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "invalid dilation height factor %d in node #%d",
                           params->dilation_height_factor, node_index);
      }
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckDepthwiseConvolutionParams(
      TfLiteContext* context, const TfLiteDepthwiseConvParams* params,
      int output_channels, int node_index) {
    if (params->stride_width <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                           params->stride_width, node_index);
      }
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                           params->stride_height, node_index);
      }
      return kTfLiteError;
    }

    if (params->depth_multiplier <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "invalid depth multiplier %d in node #%d",
                           params->depth_multiplier, node_index);
      }
      return kTfLiteError;
    }
    if (output_channels % params->depth_multiplier != 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "depth multiplier %d is incompatible with "
                           "number of output channels %d in node #%d",
                           params->depth_multiplier, output_channels,
                           node_index);
      }
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "invalid dilation width factor %d in node #%d",
                           params->dilation_width_factor, node_index);
      }
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "invalid dilation height factor %d in node #%d",
                           params->dilation_height_factor, node_index);
      }
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckPoolingParams(TfLiteContext* context,
                                         const TfLitePoolParams* params,
                                         int node_index) {
    if (params->stride_width <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                           params->stride_width, node_index);
      }
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                           params->stride_height, node_index);
      }
      return kTfLiteError;
    }

    if (params->filter_width <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "invalid filter width %d in node #%d",
                           params->filter_width, node_index);
      }
      return kTfLiteError;
    }
    if (params->filter_height <= 0) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "invalid filter height %d in node #%d",
                           params->filter_height, node_index);
      }
      return kTfLiteError;
    }
    if (params->filter_width == 1 && params->filter_height == 1) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context, "meaningless 1x1 pooling in node #%d",
                           node_index);
      }
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckNumInputsAndOutputs(TfLiteContext* context,
                                               TfLiteNode* node,
                                               int expected_num_inputs,
                                               int expected_num_outputs,
                                               int node_index) {
    if (node->inputs->size != expected_num_inputs) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "unexpected number of inputs (%d != %d) in node #%d",
                           node->inputs->size, expected_num_inputs, node_index);
      }
      return kTfLiteError;
    }
    if (node->outputs->size != expected_num_outputs) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(
            context, "unexpected number of output (%d != %d) in node #%d",
            node->outputs->size, expected_num_outputs, node_index);
      }
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorFloatType(TfLiteContext* context,
                                           const TfLiteTensor& tensor,
                                           int tensor_index, int node_index) {
    if (tensor.type != kTfLiteFloat32) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(
            context, "unsupported type %s in tensor #%d in node #%d",
            TfLiteTypeGetName(tensor.type), tensor_index, node_index);
      }
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorShape(TfLiteContext* context,
                                       const TfLiteTensor& tensor,
                                       int expected_num_dims,
                                       int tensor_index) {
    if (tensor.dims->size != expected_num_dims) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(
            context,
            "unexpected number of shape dimensions (%d != %d) in tensor #%d",
            tensor.dims->size, expected_num_dims, tensor_index);
      }
      return kTfLiteError;
    }
    for (int i = 0; i < tensor.dims->size; i++) {
      if (tensor.dims->data[i] <= 0) {
        TF_LITE_KERNEL_LOG(context, "invalid dimension #%d (%d) in tensor #%d",
                           i, tensor.dims->data[i], tensor_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckSlopeTensorShape(TfLiteContext* context,
                                            const TfLiteTensor& tensor,
                                            int tensor_index, int node_index) {
    if (tensor.dims->size < 1) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "unexpected number of shape dimensions (%d) in "
                           "tensor #%d in node #%d: "
                           "expected at least a 1D tensor",
                           tensor.dims->size, tensor_index, node_index);
      }
      return kTfLiteError;
    }
    // Validate that all non-channel dimensions (if any) are exactly 1.
    for (int i = 0; i < tensor.dims->size - 1; i++) {
      if (tensor.dims->data[i] != 1) {
        if (context != nullptr) {
          TF_LITE_KERNEL_LOG(context,
                             "unexpected value %d of shape dimension #%d in "
                             "tensor #%d in node #%d: "
                             "expected 1 for non-channel dimensions",
                             tensor.dims[i], i, tensor_index, node_index);
        }
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorNonDynamicAllocation(
      TfLiteContext* context, const TfLiteTensor& tensor, int tensor_index,
      int node_index) {
    // TODO(b/149120844): remove checks once dynamic tensors are supported
    if (tensor.allocation_type == kTfLiteDynamic) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "invalid allocation type in tensor #%d in node #%d: "
                           "expected non-dynamic tensor",
                           tensor_index, node_index);
      }
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorStaticAllocation(TfLiteContext* context,
                                                  const TfLiteTensor& tensor,
                                                  int tensor_index,
                                                  int node_index) {
    if (tensor.allocation_type != kTfLiteMmapRo ||
        tensor.data.raw_const == nullptr) {
      if (context != nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "invalid allocation type in tensor #%d in node #%d: "
                           "expected static read-only tensor",
                           tensor_index, node_index);
      }
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitNode(xnn_subgraph_t subgraph, TfLiteContext* context,
                                TfLiteRegistration* registration,
                                TfLiteNode* node, int node_index,
                                const std::vector<uint32_t>& xnnpack_tensors) {
    // TFLite context used for logging purposes. When we create a new node
    // (subgraph is non-null), logging context is the same as context, and error
    // messages are passed to TFLite. When we detect supported operations
    // (subgraph is null), logging context is null, and error messages are
    // supressed.
    TfLiteContext* logging_context = subgraph == nullptr ? nullptr : context;
    switch (registration->builtin_code) {
      case kTfLiteBuiltinAdd: {
        const TfLiteAddParams* add_params =
            static_cast<const TfLiteAddParams*>(node->builtin_data);

        return VisitAddNode(subgraph, logging_context, node_index, node,
                            context->tensors, add_params, xnnpack_tensors);
      }
      case kTfLiteBuiltinAveragePool2d: {
        const TfLitePoolParams* pool_params =
            static_cast<const TfLitePoolParams*>(node->builtin_data);

        return VisitAveragePool2DNode(subgraph, logging_context, node_index,
                                      node, context->tensors, pool_params,
                                      xnnpack_tensors);
      }
      case kTfLiteBuiltinConv2d: {
        const TfLiteConvParams* conv_params =
            static_cast<const TfLiteConvParams*>(node->builtin_data);

        return VisitConv2DNode(subgraph, logging_context, node_index, node,
                               context->tensors, conv_params, xnnpack_tensors);
      }
      case kTfLiteBuiltinDepthwiseConv2d: {
        const TfLiteDepthwiseConvParams* dwconv_params =
            static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);

        return VisitDepthwiseConv2DNode(subgraph, logging_context, node_index,
                                        node, context->tensors, dwconv_params,
                                        xnnpack_tensors);
      }
      case kTfLiteBuiltinHardSwish:
        return VisitHardSwishNode(subgraph, logging_context, node_index, node,
                                  context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinLogistic:
        return VisitLogisticNode(subgraph, logging_context, node_index, node,
                                 context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinMaxPool2d: {
        const TfLitePoolParams* pool_params =
            static_cast<const TfLitePoolParams*>(node->builtin_data);

        return VisitMaxPool2DNode(subgraph, logging_context, node_index, node,
                                  context->tensors, pool_params,
                                  xnnpack_tensors);
      }
      case kTfLiteBuiltinMul: {
        const TfLiteMulParams* mul_params =
            static_cast<const TfLiteMulParams*>(node->builtin_data);

        return VisitMulNode(subgraph, logging_context, node_index, node,
                            context->tensors, mul_params, xnnpack_tensors);
      }
      case kTfLiteBuiltinPrelu:
        return VisitPreluNode(subgraph, logging_context, node_index, node,
                              context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinRelu:
        return VisitReluNode(
            subgraph, logging_context, node_index, node, context->tensors, 0.0f,
            std::numeric_limits<float>::infinity(), xnnpack_tensors);
      case kTfLiteBuiltinReluN1To1:
        return VisitReluNode(subgraph, logging_context, node_index, node,
                             context->tensors, -1.0f, 1.0f, xnnpack_tensors);
      case kTfLiteBuiltinRelu6:
        return VisitReluNode(subgraph, logging_context, node_index, node,
                             context->tensors, 0.0f, 6.0f, xnnpack_tensors);
      case kTfLiteBuiltinSoftmax: {
        const TfLiteSoftmaxParams* softmax_params =
            static_cast<const TfLiteSoftmaxParams*>(node->builtin_data);

        return VisitSoftmaxNode(subgraph, logging_context, node_index, node,
                                context->tensors, softmax_params,
                                xnnpack_tensors);
      }
      default:
        return kTfLiteError;
    }
  }

  static TfLiteStatus VisitAddNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteAddParams* add_params,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input1_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input2_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    if (add_params != nullptr) {
      TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
          logging_context, node_index, add_params->activation, &output_min,
          &output_max));
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_add2(
          subgraph, output_min, output_max,
          /*input1_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*input2_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate ADD node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitAveragePool2DNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLitePoolParams* pool_params,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    TF_LITE_ENSURE_STATUS(
        CheckPoolingParams(logging_context, pool_params, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, &flags, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, pool_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_average_pooling_2d(
          subgraph,
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          static_cast<uint32_t>(pool_params->filter_height),
          static_cast<uint32_t>(pool_params->filter_width),
          static_cast<uint32_t>(pool_params->stride_height),
          static_cast<uint32_t>(pool_params->stride_width), output_min,
          output_max,
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], flags);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate AVERAGE_POOL_2D node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitConv2DNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteConvParams* conv_params,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckConvolutionParams(logging_context, conv_params, node_index));

    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 3, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           node->inputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& filter_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, filter_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, filter_tensor, 4,
                                           node->inputs->data[1]));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, filter_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& bias_tensor = tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, filter_tensor, node->inputs->data[2], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                           node->inputs->data[2]));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, bias_tensor, node->inputs->data[2], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           node->outputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    const int output_channels = filter_tensor.dims->data[0];
    const int kernel_height = filter_tensor.dims->data[1];
    const int kernel_width = filter_tensor.dims->data[2];
    const int input_channels = filter_tensor.dims->data[3];

    uint32_t flags;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, conv_params->padding, &flags, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, conv_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_convolution_2d(
          subgraph,
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0, static_cast<uint32_t>(kernel_height),
          static_cast<uint32_t>(kernel_width),
          static_cast<uint32_t>(conv_params->stride_height),
          static_cast<uint32_t>(conv_params->stride_width),
          static_cast<uint32_t>(conv_params->dilation_height_factor),
          static_cast<uint32_t>(conv_params->dilation_width_factor),
          /*groups=*/1, static_cast<size_t>(input_channels),
          static_cast<size_t>(output_channels), output_min, output_max,
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*filter_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*bias_id=*/xnnpack_tensors[node->inputs->data[2]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], flags);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate CONV_2D node #%d", node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitDepthwiseConv2DNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteDepthwiseConvParams* dwconv_params,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 3, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           node->inputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& filter_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, filter_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, filter_tensor, 4,
                                           node->inputs->data[1]));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, filter_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& bias_tensor = tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, filter_tensor, node->inputs->data[2], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                           node->inputs->data[2]));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, bias_tensor, node->inputs->data[2], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           node->outputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    const int kernel_height = filter_tensor.dims->data[1];
    const int kernel_width = filter_tensor.dims->data[2];
    const int output_channels = filter_tensor.dims->data[3];

    TF_LITE_ENSURE_STATUS(CheckDepthwiseConvolutionParams(
        logging_context, dwconv_params, output_channels, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, dwconv_params->padding, &flags, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, dwconv_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_depthwise_convolution_2d(
          subgraph,
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0, static_cast<uint32_t>(kernel_height),
          static_cast<uint32_t>(kernel_width),
          static_cast<uint32_t>(dwconv_params->stride_height),
          static_cast<uint32_t>(dwconv_params->stride_width),
          static_cast<uint32_t>(dwconv_params->dilation_height_factor),
          static_cast<uint32_t>(dwconv_params->dilation_width_factor),
          static_cast<uint32_t>(dwconv_params->depth_multiplier),
          /*input_channels=*/
          static_cast<uint32_t>(output_channels /
                                dwconv_params->depth_multiplier),
          output_min, output_max,
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*filter_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*bias_id=*/xnnpack_tensors[node->inputs->data[2]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], flags);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate DEPTHWISE_CONV_2D node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitHardSwishNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_hardswish(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate HARD_SWISH node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitLogisticNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_sigmoid(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate SIGMOID node #%d", node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMaxPool2DNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLitePoolParams* pool_params,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    TF_LITE_ENSURE_STATUS(
        CheckPoolingParams(logging_context, pool_params, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, &flags, node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, pool_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_max_pooling_2d(
          subgraph,
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          static_cast<uint32_t>(pool_params->filter_height),
          static_cast<uint32_t>(pool_params->filter_width),
          static_cast<uint32_t>(pool_params->stride_height),
          static_cast<uint32_t>(pool_params->stride_width),
          /*dilation_height=*/1,
          /*dilation_width=*/1, output_min, output_max,
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], flags);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate MAX_POOL_2D node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMulNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteMulParams* mul_params,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const TfLiteTensor& input1_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input1_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input1_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& input2_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input2_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input2_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    if (mul_params != nullptr) {
      TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
          logging_context, node_index, mul_params->activation, &output_min,
          &output_max));
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_multiply2(
          subgraph, output_min, output_max,
          /*input1_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*input2_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate MUL node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitPreluNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           node->inputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& slope_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, slope_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckSlopeTensorShape(
        logging_context, slope_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, slope_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           node->outputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_prelu(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*slope_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate PRELU node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitReluNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors, float output_min,
      float output_max, const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_clamp(
          subgraph, output_min, output_max,
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate RELU node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSoftmaxNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteSoftmaxParams* params,
      const std::vector<uint32_t>& xnnpack_tensors) {
    if (params->beta != 1.0f) {
      if (logging_context != nullptr) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "unsupported beta value %.7f in SOFTMAX node #%d",
                           params->beta, node_index);
      }
      return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_softmax(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate SOFTMAX node #%d", node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

 private:
  Subgraph(xnn_runtime_t runtime, std::unordered_set<int>&& externals)
      : runtime_(runtime, &xnn_delete_runtime), externals_(externals) {}

  // XNNPACK Runtime (subgraph + workspace) with smart-pointer for lifetime
  // management.
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{
      nullptr, &xnn_delete_runtime};
  // TFLite Tensor IDs == XNNPACK Value IDs of input/output tensors for the
  // delegated subgraph.
  std::unordered_set<int> externals_;
  bool first_run_{true};
};

TfLiteIntArray* GetOpsToReplace(TfLiteContext* context) {
  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Unable to get graph execution plan.");
    return nullptr;
  }

  TfLiteIntArray* nodes_to_replace = TfLiteIntArrayCreate(execution_plan->size);
  nodes_to_replace->size = 0;
  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];

    // Check if TFLite nodes can be delegated to XNNPACK
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         node_index);
      continue;  // Soft error (skip this node).
    }

    if (Subgraph::VisitNode(/*subgraph=*/nullptr, context, registration, node,
                            node_index, std::vector<uint32_t>()) != kTfLiteOk) {
      // Non-delegatable node is not an error.
      continue;
    }

    nodes_to_replace->data[nodes_to_replace->size++] = node_index;
  }
  return nodes_to_replace;
}

void* SubgraphInit(TfLiteContext* context, const char* buffer, size_t length) {
  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);

  pthreadpool_t threadpool =
      static_cast<::tflite::xnnpack::Delegate*>(params->delegate->data_)
          ->threadpool();

  return static_cast<void*>(Subgraph::Create(context, params, threadpool));
}

TfLiteStatus SubgraphPrepare(TfLiteContext* context, TfLiteNode* node) {
  return static_cast<Subgraph*>(node->user_data)->Prepare(context);
}

TfLiteStatus SubgraphInvoke(TfLiteContext* context, TfLiteNode* node) {
  return static_cast<Subgraph*>(node->user_data)->Invoke(context);
}

void SubgraphFree(TfLiteContext* context, void* buffer) {
  if (buffer != nullptr) {
    delete static_cast<Subgraph*>(buffer);
  }
}

const TfLiteRegistration kSubgraphRegistration = {
    /*.init=*/SubgraphInit,
    /*.free=*/SubgraphFree,
    /*.prepare=*/SubgraphPrepare,
    /*.invoke=*/SubgraphInvoke,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/0,
    /*.custom_name=*/"TfLiteXNNPackDelegate",
    /*.version=*/2,
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* ops_to_replace = GetOpsToReplace(context);
  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kSubgraphRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite

TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault() {
  TfLiteXNNPackDelegateOptions options = {0};
  return options;
}

TfLiteDelegate* TfLiteXNNPackDelegateCreate(
    const TfLiteXNNPackDelegateOptions* options) {
  xnn_status status = xnn_initialize(/*allocator=*/nullptr);
  if (status != xnn_status_success) {
    return nullptr;
  }

  auto* xnnpack_delegate = new ::tflite::xnnpack::Delegate(options);
  return xnnpack_delegate ? xnnpack_delegate->tflite_delegate() : nullptr;
}

void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate) {
  if (delegate != nullptr) {
    delete static_cast<::tflite::xnnpack::Delegate*>(delegate->data_);
  }
}
