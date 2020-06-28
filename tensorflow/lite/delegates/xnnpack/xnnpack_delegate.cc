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
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fp16.h>
#include <xnnpack.h>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/tools/optimize/sparsity/format_converter.h"

namespace tflite {
namespace xnnpack {
namespace {

// Forward declaration.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
  friend class Subgraph;

 public:
  explicit Delegate(const TfLiteXNNPackDelegateOptions* options) {
#if !defined(__EMSCRIPTEN__) || defined(__EMSCRIPTEN_PTHREADS__)
    if (options != nullptr && options->num_threads > 1) {
      threadpool_.reset(
          pthreadpool_create(static_cast<size_t>(options->num_threads)));
    }
#endif
    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Created TensorFlow Lite XNNPACK delegate for CPU.");
  }

  TfLiteIntArray* PrepareOpsToDelegate(TfLiteContext* context);
  TfLiteDelegate* tflite_delegate() { return &delegate_; }

  pthreadpool_t threadpool() const {
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

  // Unpacked data for quasi-static tensors, i.e. tensors produced by
  // dequantizing or unpacking static buffers.
  std::vector<char> static_unpacked_data_;
  // Mapping from a tensor index for a quasi-static tensor to the offset to
  // its unpacked data within static_unpacked_data_.
  std::unordered_map<int, size_t> static_unpacked_data_map_;
  // Set of indices of nodes which unpack static data, e.g. Dequantize
  // operators which convert FP16 static weights to FP32. These nodes are simply
  // ignored in the delegate implementation, because their outputs are
  // pre-unpacked in DelegatePrepare.
  std::unordered_set<int> static_unpack_nodes_;
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
                          const Delegate* delegate) {
    // Convert subgraph inputs and outputs to hash sets for faster lookup.
    const std::unordered_set<int> inputs(
        &params->input_tensors->data[0],
        &params->input_tensors->data[params->input_tensors->size]);
    std::unordered_set<int> outputs;
    for (int o = 0; o < params->output_tensors->size; o++) {
      const int output_tensor_idx = params->output_tensors->data[o];
      // Exclude quasi-static tensors which may have become subgraph outputs
      // after partitioning.
      if (delegate->static_unpacked_data_map_.count(output_tensor_idx) == 0) {
        outputs.insert(output_tensor_idx);
      }
    }
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
      const int node_index = params->nodes_to_replace->data[i];
      if (delegate->static_unpack_nodes_.count(node_index)) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      switch (registration->builtin_code) {
        case kTfLiteBuiltinMean:
        case kTfLiteBuiltinPad:
          // Ignore the second input (static padding, or axes), because it is
          // represented as parameters of the XNNPACK operator rather than
          // extra input.
          {
            const int t = node->inputs->data[0];
            tensors[t] = t;
          }
          break;
        default:
          // All other operators: process all inputs
          for (int k = 0; k < node->inputs->size; k++) {
            const int t = node->inputs->data[k];
            tensors[t] = t;
          }
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
      } else {
        // Check for quasi-static data.
        const auto it = delegate->static_unpacked_data_map_.find(t);
        if (it != delegate->static_unpacked_data_map_.end()) {
          data = delegate->static_unpacked_data_.data() + it->second;
        }
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

    // Create a set of quasi-static tensors for VisitNode function
    std::unordered_set<int> quasi_static_tensors;
    for (const std::pair<const int, size_t>& entry :
         delegate->static_unpacked_data_map_) {
      quasi_static_tensors.insert(entry.first);
    }

    // Create XNNPACK nodes for TFLite delegate nodes
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];
      if (delegate->static_unpack_nodes_.count(node_index)) {
        // The node unpacks static input and can be skipped because its input
        // was pre-unpacked in DelegatePrepare.
        continue;
      }

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      if (VisitNode(subgraph.get(), context, registration, node, node_index,
                    quasi_static_tensors, xnnpack_tensors) != kTfLiteOk) {
        return nullptr;
      }
    }

    xnn_runtime_t runtime_ptr = nullptr;
    status = xnn_create_runtime_v2(subgraph.get(), delegate->threadpool(),
                                   /*flags=*/0, &runtime_ptr);
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
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid padding mode (%d) in node #%d",
                                 static_cast<int>(padding), node_index);
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
      case kTfLiteActReluN1To1:
        *output_min = -1.0f;
        *output_max = +1.0f;
        return kTfLiteOk;
      case kTfLiteActRelu6:
        *output_min = 0.0f;
        *output_max = 6.0f;
        return kTfLiteOk;
      case kTfLiteActTanh:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Tanh) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSignBit:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sign) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSigmoid:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sigmoid) in node #%d",
            node_index);
        return kTfLiteError;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid fused activation (%d) in node #%d",
                                 static_cast<int>(activation), node_index);
        return kTfLiteError;
    }
  }

  static TfLiteStatus CheckConvolutionParams(TfLiteContext* context,
                                             const TfLiteConvParams* params,
                                             int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation width factor %d in node #%d",
                               params->dilation_width_factor, node_index);
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation height factor %d in node #%d",
                               params->dilation_height_factor, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckDepthwiseConvolutionParams(
      TfLiteContext* context, const TfLiteDepthwiseConvParams* params,
      int output_channels, int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->depth_multiplier <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid depth multiplier %d in node #%d",
                               params->depth_multiplier, node_index);
      return kTfLiteError;
    }
    if (output_channels % params->depth_multiplier != 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "depth multiplier %d is incompatible with "
                               "number of output channels %d in node #%d",
                               params->depth_multiplier, output_channels,
                               node_index);
      return kTfLiteError;
    }

    if (params->dilation_width_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation width factor %d in node #%d",
                               params->dilation_width_factor, node_index);
      return kTfLiteError;
    }
    if (params->dilation_height_factor <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "invalid dilation height factor %d in node #%d",
                               params->dilation_height_factor, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckMediaPipeTransposedConvolutionParams(
      TfLiteContext* context, const TfLiteTransposeConvParams* params,
      int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckMediaPipePoolParams(TfLiteContext* context,
                                               const TfLitePoolParams* params,
                                               int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }
    if (params->filter_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter width %d in node #%d",
                               params->filter_width, node_index);
      return kTfLiteError;
    }
    if (params->filter_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter height %d in node #%d",
                               params->filter_height, node_index);
      return kTfLiteError;
    }
    if (params->filter_width != params->stride_width) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "filter width %d does not match stride width %d in node #%d",
          params->filter_width, params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->filter_height != params->stride_height) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "filter height %d does not match stride height %d in node #%d",
          params->filter_height, params->stride_height, node_index);
      return kTfLiteError;
    }
    switch (params->activation) {
      case kTfLiteActNone:
        break;
      case kTfLiteActRelu:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Relu) in node #%d",
            node_index);
        return kTfLiteOk;
      case kTfLiteActReluN1To1:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (ReluMinus1To1) in node #%d",
            node_index);
        return kTfLiteOk;
      case kTfLiteActRelu6:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Relu6) in node #%d",
            node_index);
        return kTfLiteOk;
      case kTfLiteActTanh:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Tanh) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSignBit:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sign) in node #%d",
            node_index);
        return kTfLiteError;
      case kTfLiteActSigmoid:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "unsupported fused activation (Sigmoid) in node #%d",
            node_index);
        return kTfLiteError;
      default:
        TF_LITE_MAYBE_KERNEL_LOG(
            context, "invalid fused activation (%d) in node #%d",
            static_cast<int>(params->activation), node_index);
        return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckFullyConnectedParams(
      TfLiteContext* context, const TfLiteFullyConnectedParams* params,
      int node_index) {
    if (params->weights_format != kTfLiteFullyConnectedWeightsFormatDefault) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unsupported non-default weights format in node #%d",
          node_index);
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus CheckPoolingParams(TfLiteContext* context,
                                         const TfLitePoolParams* params,
                                         int node_index) {
    if (params->stride_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride width %d in node #%d",
                               params->stride_width, node_index);
      return kTfLiteError;
    }
    if (params->stride_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid stride height %d in node #%d",
                               params->stride_height, node_index);
      return kTfLiteError;
    }

    if (params->filter_width <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter width %d in node #%d",
                               params->filter_width, node_index);
      return kTfLiteError;
    }
    if (params->filter_height <= 0) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "invalid filter height %d in node #%d",
                               params->filter_height, node_index);
      return kTfLiteError;
    }
    if (params->filter_width == 1 && params->filter_height == 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context, "meaningless 1x1 pooling in node #%d",
                               node_index);
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
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of inputs (%d != %d) in node #%d",
          node->inputs->size, expected_num_inputs, node_index);
      return kTfLiteError;
    }
    if (node->outputs->size != expected_num_outputs) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unexpected number of output (%d != %d) in node #%d",
          node->outputs->size, expected_num_outputs, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorType(TfLiteContext* context,
                                      const TfLiteTensor& tensor,
                                      TfLiteType expected_type,
                                      int tensor_index, int node_index) {
    if (tensor.type != expected_type) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context, "unsupported type %s in tensor #%d in node #%d",
          TfLiteTypeGetName(tensor.type), tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorFloatType(TfLiteContext* context,
                                           const TfLiteTensor& tensor,
                                           int tensor_index, int node_index) {
    return CheckTensorType(context, tensor, kTfLiteFloat32, tensor_index,
                           node_index);
  }

  static TfLiteStatus CheckTensorShape(TfLiteContext* context,
                                       const TfLiteTensor& tensor,
                                       int min_num_dims, int max_num_dims,
                                       int tensor_index) {
    if (min_num_dims == max_num_dims) {
      if (tensor.dims->size != min_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported number of shape dimensions (%d) in tensor #%d: "
            "%d dimensions expected",
            tensor.dims->size, tensor_index, min_num_dims);
        return kTfLiteError;
      }
    } else {
      if (tensor.dims->size < min_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported number of shape dimensions (%d) in tensor #%d: "
            "at least %d dimensions expected",
            tensor.dims->size, tensor_index, min_num_dims);
        return kTfLiteError;
      }
      if (tensor.dims->size > max_num_dims) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unsupported number of shape dimensions (%d) in tensor #%d: "
            "at most %d dimensions expected",
            tensor.dims->size, tensor_index, max_num_dims);
        return kTfLiteError;
      }
    }
    for (int i = 0; i < tensor.dims->size; i++) {
      if (tensor.dims->data[i] <= 0) {
        TF_LITE_MAYBE_KERNEL_LOG(context,
                                 "invalid num of elements (%d) in "
                                 "dimension #%d in tensor #%d",
                                 tensor.dims->data[i], i, tensor_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorShape(TfLiteContext* context,
                                       const TfLiteTensor& tensor,
                                       int expected_num_dims,
                                       int tensor_index) {
    return CheckTensorShape(context, tensor, expected_num_dims,
                            expected_num_dims, tensor_index);
  }

  static TfLiteStatus CheckSlopeTensorShape(TfLiteContext* context,
                                            const TfLiteTensor& tensor,
                                            int tensor_index, int node_index) {
    if (tensor.dims->size < 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "tensor #%d in node #%d: "
                               "expected at least a 1D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    // Validate that all non-channel dimensions (if any) are exactly 1.
    for (int i = 0; i < tensor.dims->size - 1; i++) {
      if (tensor.dims->data[i] != 1) {
        TF_LITE_MAYBE_KERNEL_LOG(
            context,
            "unexpected value %d of shape dimension #%d in "
            "tensor #%d in node #%d: "
            "expected 1 for non-channel dimensions",
            tensor.dims[i], i, tensor_index, node_index);
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckPaddingsTensorShape(TfLiteContext* context,
                                               const TfLiteTensor& tensor,
                                               int expected_rows,
                                               int tensor_index,
                                               int node_index) {
    if (tensor.dims->size != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "padding tensor #%d in node #%d: "
                               "expected a 2D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    if (tensor.dims->data[0] != expected_rows) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of rows (%d) in "
                               "padding tensor #%d in node #%d: "
                               "%d rows expected",
                               tensor.dims->size, tensor_index, node_index,
                               expected_rows);
      return kTfLiteError;
    }
    if (tensor.dims->data[1] != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of columns (%d) in "
                               "padding tensor #%d in node #%d: "
                               "2 columns expected",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckAxesTensorShape(TfLiteContext* context,
                                           const TfLiteTensor& tensor,
                                           int tensor_index, int node_index) {
    if (tensor.dims->size != 1) {
      TF_LITE_MAYBE_KERNEL_LOG(context,
                               "unexpected number of shape dimensions (%d) in "
                               "axes tensor #%d in node #%d: "
                               "expected a 1D tensor",
                               tensor.dims->size, tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus CheckTensorNonDynamicAllocation(
      TfLiteContext* context, const TfLiteTensor& tensor, int tensor_index,
      int node_index) {
    // TODO(b/149120844): remove checks once dynamic tensors are supported
    if (tensor.allocation_type == kTfLiteDynamic) {
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in node #%d: "
          "expected non-dynamic tensor",
          tensor_index, node_index);
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
      TF_LITE_MAYBE_KERNEL_LOG(
          context,
          "invalid allocation type in tensor #%d in node #%d: "
          "expected static read-only tensor",
          tensor_index, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static TfLiteStatus VisitNode(
      xnn_subgraph_t subgraph, TfLiteContext* context,
      TfLiteRegistration* registration, TfLiteNode* node, int node_index,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::vector<uint32_t>& xnnpack_tensors) {
    // TFLite context used for logging purposes. When we create a new node
    // (subgraph is non-null), logging context is the same as context, and error
    // messages are passed to TFLite. When we detect supported operations
    // (subgraph is null), logging context is null, and error messages are
    // supressed.
    TfLiteContext* logging_context = subgraph == nullptr ? nullptr : context;
    switch (registration->builtin_code) {
      case kTfLiteBuiltinAbs:
        return VisitAbsNode(subgraph, logging_context, node_index, node,
                            context->tensors, xnnpack_tensors);
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
      case kTfLiteBuiltinCeil:
        return VisitCeilNode(subgraph, logging_context, node_index, node,
                             context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinConv2d: {
        const TfLiteConvParams* conv_params =
            static_cast<const TfLiteConvParams*>(node->builtin_data);

        return VisitConv2DNode(subgraph, logging_context, node_index, node,
                               context->tensors, conv_params,
                               quasi_static_tensors, xnnpack_tensors);
      }
      case kTfLiteBuiltinDepthwiseConv2d: {
        const TfLiteDepthwiseConvParams* dwconv_params =
            static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);

        return VisitDepthwiseConv2DNode(subgraph, logging_context, node_index,
                                        node, context->tensors, dwconv_params,
                                        quasi_static_tensors, xnnpack_tensors);
      }
      case kTfLiteBuiltinDiv: {
        const TfLiteDivParams* div_params =
            static_cast<const TfLiteDivParams*>(node->builtin_data);

        return VisitDivNode(subgraph, logging_context, node_index, node,
                            context->tensors, div_params, xnnpack_tensors);
      }
      case kTfLiteBuiltinFullyConnected: {
        const TfLiteFullyConnectedParams* fc_params =
            static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

        return VisitFullyConnectedNode(subgraph, logging_context, node_index,
                                       node, context->tensors, fc_params,
                                       quasi_static_tensors, xnnpack_tensors);
      }
      case kTfLiteBuiltinFloor:
        return VisitFloorNode(subgraph, logging_context, node_index, node,
                              context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinHardSwish:
        return VisitHardSwishNode(subgraph, logging_context, node_index, node,
                                  context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinLeakyRelu: {
        const TfLiteLeakyReluParams* leaky_relu_params =
            static_cast<const TfLiteLeakyReluParams*>(node->builtin_data);

        return VisitLeakyReluNode(subgraph, logging_context, node_index, node,
                                  context->tensors, leaky_relu_params,
                                  xnnpack_tensors);
      }
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
      case kTfLiteBuiltinMaximum:
        return VisitMaximumNode(subgraph, logging_context, node_index, node,
                                context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinMean: {
        const TfLiteReducerParams* reducer_params =
            static_cast<const TfLiteReducerParams*>(node->builtin_data);

        return VisitMeanNode(subgraph, logging_context, node_index, node,
                             context->tensors, reducer_params, xnnpack_tensors);
      }
      case kTfLiteBuiltinMinimum:
        return VisitMinimumNode(subgraph, logging_context, node_index, node,
                                context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinMul: {
        const TfLiteMulParams* mul_params =
            static_cast<const TfLiteMulParams*>(node->builtin_data);

        return VisitMulNode(subgraph, logging_context, node_index, node,
                            context->tensors, mul_params, xnnpack_tensors);
      }
      case kTfLiteBuiltinNeg:
        return VisitNegNode(subgraph, logging_context, node_index, node,
                            context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinPad:
        return VisitPadNode(subgraph, logging_context, node_index, node,
                            context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinPrelu:
        return VisitPreluNode(subgraph, logging_context, node_index, node,
                              context->tensors, quasi_static_tensors,
                              xnnpack_tensors);
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
      case kTfLiteBuiltinRound:
        return VisitRoundNode(subgraph, logging_context, node_index, node,
                              context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinSoftmax: {
        const TfLiteSoftmaxParams* softmax_params =
            static_cast<const TfLiteSoftmaxParams*>(node->builtin_data);

        return VisitSoftmaxNode(subgraph, logging_context, node_index, node,
                                context->tensors, softmax_params,
                                xnnpack_tensors);
      }
      case kTfLiteBuiltinSquare:
        return VisitSquareNode(subgraph, logging_context, node_index, node,
                               context->tensors, xnnpack_tensors);
      case kTfLiteBuiltinSquaredDifference:
        return VisitSquaredDifferenceNode(subgraph, logging_context, node_index,
                                          node, context->tensors,
                                          xnnpack_tensors);
      case kTfLiteBuiltinSub: {
        const TfLiteSubParams* sub_params =
            static_cast<const TfLiteSubParams*>(node->builtin_data);

        return VisitSubNode(subgraph, logging_context, node_index, node,
                            context->tensors, sub_params, xnnpack_tensors);
      }
      case kTfLiteBuiltinCustom: {
        if (strcmp(registration->custom_name, "Convolution2DTransposeBias") ==
            0) {
          TfLiteTransposeConvParams deconv_params = {kTfLitePaddingUnknown};
          std::memcpy(&deconv_params, node->custom_initial_data,
                      node->custom_initial_data_size);

          return VisitMediaPipeDeconvolutionNode(
              subgraph, context, node_index, node, context->tensors,
              &deconv_params, quasi_static_tensors, xnnpack_tensors);
        } else if (strcmp(registration->custom_name,
                          "MaxPoolingWithArgmax2D") == 0) {
          TfLitePoolParams pool_params = {kTfLitePaddingUnknown};
          std::memcpy(&pool_params, node->custom_initial_data,
                      node->custom_initial_data_size);

          return VisitMediaPipeMaxPoolingNode(subgraph, context, node_index,
                                              node, context->tensors,
                                              &pool_params, xnnpack_tensors);
        } else if (strcmp(registration->custom_name, "MaxUnpooling2D") == 0) {
          TfLitePoolParams pool_params = {kTfLitePaddingUnknown};
          std::memcpy(&pool_params, node->custom_initial_data,
                      node->custom_initial_data_size);

          return VisitMediaPipeUnpoolingNode(subgraph, context, node_index,
                                             node, context->tensors,
                                             &pool_params, xnnpack_tensors);
        }
        return kTfLiteError;
      }
      default:
        return kTfLiteError;
    }
  }

  static TfLiteStatus VisitAbsNode(
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
      const xnn_status status = xnn_define_abs(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate ABS node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
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

  static TfLiteStatus VisitCeilNode(
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
      const xnn_status status = xnn_define_ceiling(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate CEIL node #%d",
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
      const std::unordered_set<int>& quasi_static_tensors,
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
    if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, node->inputs->data[1], node_index));
    }

    const TfLiteTensor& bias_tensor = tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, filter_tensor, node->inputs->data[2], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                           node->inputs->data[2]));
    if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, bias_tensor, node->inputs->data[2], node_index));
    }

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
      const std::unordered_set<int>& quasi_static_tensors,
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
    if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, node->inputs->data[1], node_index));
    }

    const TfLiteTensor& bias_tensor = tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, filter_tensor, node->inputs->data[2], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                           node->inputs->data[2]));
    if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, bias_tensor, node->inputs->data[2], node_index));
    }

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

  static TfLiteStatus VisitDivNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteDivParams* div_params,
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
    if (div_params != nullptr) {
      TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
          logging_context, node_index, div_params->activation, &output_min,
          &output_max));
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_divide(
          subgraph, output_min, output_max,
          /*input1_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*input2_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate DIV node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitFullyConnectedNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteFullyConnectedParams* fc_params,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckFullyConnectedParams(logging_context, fc_params, node_index));

    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 3, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& filter_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, filter_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, filter_tensor, 2,
                                           node->inputs->data[1]));
    if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, node->inputs->data[1], node_index));
    }

    const TfLiteTensor& bias_tensor = tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, filter_tensor, node->inputs->data[2], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                           node->inputs->data[2]));
    if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, bias_tensor, node->inputs->data[2], node_index));
    }

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    const int32_t output_channels = filter_tensor.dims->data[0];
    const int32_t input_channels = filter_tensor.dims->data[1];

    if (input_tensor.dims->size == 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unexpected number of shape dimensions %d in tensor #%d",
          input_tensor.dims->size, node->inputs->data[0]);
      return kTfLiteError;
    }

    int32_t num_input_elements = 1;
    for (int i = 0; i < input_tensor.dims->size; i++) {
      if (input_tensor.dims->data[i] <= 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context, "invalid dimension #%d (%d) in tensor #%d", i,
            input_tensor.dims->data[i], node->inputs->data[0]);
        return kTfLiteError;
      }
      num_input_elements *= input_tensor.dims->data[i];
    }

    if (fc_params->keep_num_dims) {
      TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor,
                                             input_tensor.dims->size,
                                             node->outputs->data[0]));

      for (int i = 0; i < input_tensor.dims->size - 1; i++) {
        if (input_tensor.dims->data[i] != output_tensor.dims->data[i]) {
          TF_LITE_MAYBE_KERNEL_LOG(
              logging_context,
              "mismatch in shape dimension %d (%d != %d) in input and output "
              "tensors of FULLY_CONNECTED operator #%d",
              i, input_tensor.dims->data[i], output_tensor.dims->data[i],
              node_index);
          return kTfLiteError;
        }
      }
    } else {
      if (num_input_elements % input_channels != 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "number of elements in input tensor #%d in FULLY_CONNECTED "
            "operator is not divisible by input channels (%d)",
            node->inputs->data[0], input_channels);
        return kTfLiteError;
      }

      TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 2,
                                             node->outputs->data[0]));

      if (output_tensor.dims->data[0] != num_input_elements / input_channels) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "batch size %d in output tensor #%d in FULLY_CONNECTED operator "
            "does not match batch size %d in reshaped input tensor #%d",
            output_tensor.dims->data[0], node->outputs->data[0],
            num_input_elements / input_channels, node->inputs->data[0]);
        return kTfLiteError;
      }
    }

    if (output_tensor.dims->data[output_tensor.dims->size - 1] !=
        output_channels) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "number of channels %d in output tensor #%d does not match output "
          "channels %d in filter tensor #%d",
          output_tensor.dims->data[output_tensor.dims->size - 1],
          node->outputs->data[0], output_channels, node->inputs->data[1]);
      return kTfLiteError;
    }

    float output_min = -std::numeric_limits<float>::infinity();
    float output_max = +std::numeric_limits<float>::infinity();
    TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
        logging_context, node_index, fc_params->activation, &output_min,
        &output_max));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_fully_connected(
          subgraph, output_min, output_max,
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*filter_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*bias_id=*/xnnpack_tensors[node->inputs->data[2]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]],
          /*flags=*/fc_params->keep_num_dims ? 0
                                             : XNN_FLAG_TENSORFLOW_RESHAPE_2D);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate FULLY_CONNECTED node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitFloorNode(
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
      const xnn_status status = xnn_define_floor(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate FLOOR node #%d",
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

  static TfLiteStatus VisitLeakyReluNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteLeakyReluParams* leaky_relu_params,
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
      const xnn_status status = xnn_define_leaky_relu(
          subgraph, leaky_relu_params->alpha,
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate LEAKY_RELU node #%d",
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
                           "failed to delegate LOGISTIC node #%d", node_index);
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

  static TfLiteStatus VisitMaximumNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
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

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_maximum2(
          subgraph, /*input1_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*input2_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate MAXIMUM node #%d", node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMeanNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteReducerParams* reducer_params,
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

    const TfLiteTensor& axes_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, axes_tensor,
                                          kTfLiteInt32, node->inputs->data[1],
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckAxesTensorShape(
        logging_context, axes_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, axes_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           node->outputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    if (!reducer_params->keep_dims) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unsupported MEAN reduction without keep_dims attributes in node %d",
          node_index);
      return kTfLiteError;
    }

    if (axes_tensor.dims->data[0] != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context,
          "unsupported MEAN reduction along %d axes in node %d",
          axes_tensor.dims->data[0], node_index);
      return kTfLiteError;
    }

    const int32_t* axes_data =
        reinterpret_cast<const int32_t*>(axes_tensor.data.data);
    if (std::min(axes_data[0], axes_data[1]) != 1 ||
        std::max(axes_data[0], axes_data[1]) != 2) {
      TF_LITE_MAYBE_KERNEL_LOG(logging_context,
                               "unsupported MEAN reduction along non-spatial "
                               "axes %d and %d in node %d",
                               std::min(axes_data[0], axes_data[1]),
                               std::max(axes_data[0], axes_data[1]),
                               node_index);
      return kTfLiteError;
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_global_average_pooling_2d(
          subgraph,
          /*output_min=*/-std::numeric_limits<float>::infinity(),
          /*output_max=*/+std::numeric_limits<float>::infinity(),
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate MEAN node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMediaPipeDeconvolutionNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteTransposeConvParams* deconv_params,
      const std::unordered_set<int>& quasi_static_tensors,
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
    if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, filter_tensor, node->inputs->data[1], node_index));
    }

    const TfLiteTensor& bias_tensor = tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, filter_tensor, node->inputs->data[2], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, bias_tensor, 1,
                                           node->inputs->data[2]));
    if (quasi_static_tensors.count(node->inputs->data[2]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, bias_tensor, node->inputs->data[2], node_index));
    }

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

    TF_LITE_ENSURE_STATUS(CheckMediaPipeTransposedConvolutionParams(
        logging_context, deconv_params, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, deconv_params->padding, &flags, node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_deconvolution_2d(
          subgraph,
          /*padding_top=*/0,
          /*padding_right=*/0,
          /*padding_bottom=*/0,
          /*padding_left=*/0,
          /*adjustment_height=*/0,
          /*adjustment_width=*/0, static_cast<uint32_t>(kernel_height),
          static_cast<uint32_t>(kernel_width),
          static_cast<uint32_t>(deconv_params->stride_height),
          static_cast<uint32_t>(deconv_params->stride_width),
          /*dilation_height=*/1,
          /*dilation_width=*/1,
          /*groups=*/1,
          /*group_input_channels=*/input_channels,
          /*group_output_channels=*/output_channels,
          /*output_min=*/-std::numeric_limits<float>::infinity(),
          /*output_max=*/+std::numeric_limits<float>::infinity(),
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*filter_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*bias_id=*/xnnpack_tensors[node->inputs->data[2]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], flags);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context,
            "failed to delegate Convolution2DTransposeBias node #%d",
            node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMediaPipeMaxPoolingNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLitePoolParams* pool_params,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 1, 2, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 4,
                                           node->inputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& output_value_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloatType(logging_context, output_value_tensor,
                             node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_value_tensor,
                                           4, node->outputs->data[0]));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(logging_context, output_value_tensor,
                                        node->outputs->data[0], node_index));

    const TfLiteTensor& output_index_tensor = tensors[node->outputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_index_tensor,
                                           4, node->outputs->data[1]));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(logging_context, output_index_tensor,
                                        node->outputs->data[1], node_index));

    TF_LITE_ENSURE_STATUS(
        CheckMediaPipePoolParams(logging_context, pool_params, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, &flags, node_index));

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_argmax_pooling_2d(
          subgraph,
          /*input_padding_top=*/0,
          /*input_padding_right=*/0,
          /*input_padding_bottom=*/0,
          /*input_padding_left=*/0,
          static_cast<uint32_t>(pool_params->filter_height),
          static_cast<uint32_t>(pool_params->filter_width),
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_value_id=*/xnnpack_tensors[node->outputs->data[0]],
          /*output_index_id=*/xnnpack_tensors[node->outputs->data[1]], flags);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(
            logging_context,
            "failed to delegate CUSTOM(MaxPoolingWithArgmax2D) node #%d",
            node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMediaPipeUnpoolingNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLitePoolParams* pool_params,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const TfLiteTensor& input_value_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(
        CheckTensorFloatType(logging_context, input_value_tensor,
                             node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_value_tensor,
                                           4, node->inputs->data[0]));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(logging_context, input_value_tensor,
                                        node->inputs->data[0], node_index));

    const TfLiteTensor& input_index_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_index_tensor,
                                           4, node->inputs->data[1]));
    TF_LITE_ENSURE_STATUS(
        CheckTensorNonDynamicAllocation(logging_context, input_index_tensor,
                                        node->inputs->data[1], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 4,
                                           node->outputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    TF_LITE_ENSURE_STATUS(
        CheckMediaPipePoolParams(logging_context, pool_params, node_index));

    uint32_t flags = 0;
    TF_LITE_ENSURE_STATUS(CalculatePadding(
        logging_context, pool_params->padding, &flags, node_index));
    if (flags != 0) {
      TF_LITE_MAYBE_KERNEL_LOG(
          logging_context, "invalid padding mode (%d) in node #%d",
          static_cast<int>(pool_params->padding), node_index);
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_unpooling_2d(
          subgraph,
          /*padding_top=*/0,
          /*padding_right=*/0,
          /*padding_bottom=*/0,
          /*padding_left=*/0, static_cast<uint32_t>(pool_params->filter_height),
          static_cast<uint32_t>(pool_params->filter_width),
          /*input_value_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*input_index_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]],
          /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate CUSTOM(MaxUnpooling2D) node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMinimumNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
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

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_minimum2(
          subgraph, /*input1_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*input2_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate MINIMUM node #%d", node_index);
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

  static TfLiteStatus VisitNegNode(
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
      const xnn_status status = xnn_define_negate(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate NEG node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitPadNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 1,
                                           XNN_MAX_TENSOR_DIMS,
                                           node->inputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& paddings_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorType(logging_context, paddings_tensor,
                                          kTfLiteInt32, node->inputs->data[1],
                                          node_index));
    TF_LITE_ENSURE_STATUS(CheckPaddingsTensorShape(
        logging_context, paddings_tensor, input_tensor.dims->size,
        node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
        logging_context, paddings_tensor, node->inputs->data[1], node_index));

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 1,
                                           XNN_MAX_TENSOR_DIMS,
                                           node->outputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, output_tensor, node->outputs->data[0], node_index));

    const int32_t* paddings_data =
        reinterpret_cast<const int32_t*>(paddings_tensor.data.data);
    for (int i = 0; i < paddings_tensor.dims->size; i++) {
      const int32_t pre_padding = paddings_data[i * 2 + 0];
      if (pre_padding < 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "invalid pre-padding %d for dimension #%d in node %d", pre_padding,
            i, node_index);
        return kTfLiteError;
      }

      const int32_t post_padding = paddings_data[i * 2 + 1];
      if (post_padding < 0) {
        TF_LITE_MAYBE_KERNEL_LOG(
            logging_context,
            "invalid post-padding %d for dimension #%d in node %d", pre_padding,
            i, node_index);
        return kTfLiteError;
      }
    }

    if (subgraph != nullptr) {
      std::array<size_t, XNN_MAX_TENSOR_DIMS> pre_paddings{};
      std::array<size_t, XNN_MAX_TENSOR_DIMS> post_paddings{};
      for (int i = 0; i < paddings_tensor.dims->data[0]; i++) {
        pre_paddings[i] = static_cast<size_t>(paddings_data[i * 2 + 0]);
        post_paddings[i] = static_cast<size_t>(paddings_data[i * 2 + 1]);
      }

      const xnn_status status = xnn_define_static_constant_pad(
          subgraph, pre_paddings.data(), post_paddings.data(),
          /*padding_value=*/0.0f,
          /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate PAD node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitPreluNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const std::unordered_set<int>& quasi_static_tensors,
      const std::vector<uint32_t>& xnnpack_tensors) {
    TF_LITE_ENSURE_STATUS(
        CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

    const TfLiteTensor& input_tensor = tensors[node->inputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, input_tensor, node->inputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, input_tensor, 1,
                                           XNN_MAX_TENSOR_DIMS,
                                           node->inputs->data[0]));
    TF_LITE_ENSURE_STATUS(CheckTensorNonDynamicAllocation(
        logging_context, input_tensor, node->inputs->data[0], node_index));

    const TfLiteTensor& slope_tensor = tensors[node->inputs->data[1]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, slope_tensor, node->inputs->data[1], node_index));
    TF_LITE_ENSURE_STATUS(CheckSlopeTensorShape(
        logging_context, slope_tensor, node->inputs->data[1], node_index));
    if (quasi_static_tensors.count(node->inputs->data[1]) == 0) {
      TF_LITE_ENSURE_STATUS(CheckTensorStaticAllocation(
          logging_context, slope_tensor, node->inputs->data[1], node_index));
    }

    const TfLiteTensor& output_tensor = tensors[node->outputs->data[0]];
    TF_LITE_ENSURE_STATUS(CheckTensorFloatType(
        logging_context, output_tensor, node->outputs->data[0], node_index));
    TF_LITE_ENSURE_STATUS(CheckTensorShape(logging_context, output_tensor, 1,
                                           XNN_MAX_TENSOR_DIMS,
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

  static TfLiteStatus VisitRoundNode(
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
      const xnn_status status = xnn_define_bankers_rounding(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate ROUND node #%d",
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

  static TfLiteStatus VisitSquareNode(
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
      const xnn_status status = xnn_define_square(
          subgraph, /*input_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate SQUARE node #%d", node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSquaredDifferenceNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
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

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_squared_difference(
          subgraph, /*input1_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*input2_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context,
                           "failed to delegate SQUARED_DIFFERENCE node #%d",
                           node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitSubNode(
      xnn_subgraph_t subgraph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteSubParams* sub_params,
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
    if (sub_params != nullptr) {
      TF_LITE_ENSURE_STATUS(ConvertActivationToOutputRange(
          logging_context, node_index, sub_params->activation, &output_min,
          &output_max));
    }

    if (subgraph != nullptr) {
      const xnn_status status = xnn_define_subtract(
          subgraph, output_min, output_max,
          /*input1_id=*/xnnpack_tensors[node->inputs->data[0]],
          /*input2_id=*/xnnpack_tensors[node->inputs->data[1]],
          /*output_id=*/xnnpack_tensors[node->outputs->data[0]], /*flags=*/0);
      if (status != xnn_status_success) {
        TF_LITE_KERNEL_LOG(logging_context, "failed to delegate SUB node #%d",
                           node_index);
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

TfLiteIntArray* Delegate::PrepareOpsToDelegate(TfLiteContext* context) {
  // Clear previous data, in case the delegate is reused without re-creation.
  static_unpacked_data_map_.clear();
  static_unpacked_data_.clear();
  static_unpack_nodes_.clear();

  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "Unable to get graph execution plan.");
    return nullptr;
  }

  // Mapping for quasi-static (unpacked from static) tensor index to the node
  // index that produced it.
  std::unordered_map<int, int> quasi_static_tensors_producers;
  // Set of all quasi-static tensors in the execution plan.
  std::unordered_set<int> quasi_static_tensors;
  // Set of quasi-static tensors consumed by the delegated nodes.
  std::unordered_set<int> quasi_static_tensors_to_unpack;

  TfLiteIntArray* nodes_to_delegate =
      TfLiteIntArrayCreate(execution_plan->size);
  nodes_to_delegate->size = 0;
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

    // Prepare to unpack FP16 tensors.
    if (registration->builtin_code == kTfLiteBuiltinDequantize &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];
      if (input_tensor.allocation_type == kTfLiteMmapRo &&
          input_tensor.type == kTfLiteFloat16 &&
          output_tensor.type == kTfLiteFloat32) {
        static_unpack_nodes_.insert(i);
        quasi_static_tensors_producers[node->outputs->data[0]] = i;
        quasi_static_tensors.insert(node->outputs->data[0]);

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    // Prepare to unpack sparse tensors.
    // TODO(b/157729695): In the future, we also need to handle the case where a
    // sparse tensor is fed to a TFLite op directly, and no Densify() op is
    // inserted. For now this is not a problem because the Conv() op in tflite
    // can only consume dense tensors.
    if (registration->builtin_code == kTfLiteBuiltinDensify &&
        node->inputs->size == 1 && node->outputs->size == 1) {
      const TfLiteTensor& input_tensor =
          context->tensors[node->inputs->data[0]];
      const TfLiteTensor& output_tensor =
          context->tensors[node->outputs->data[0]];
      if (input_tensor.allocation_type == kTfLiteMmapRo &&
          input_tensor.sparsity != nullptr &&
          input_tensor.type == kTfLiteFloat32 &&
          output_tensor.type == kTfLiteFloat32) {
        static_unpack_nodes_.insert(i);
        quasi_static_tensors_producers[node->outputs->data[0]] = i;
        quasi_static_tensors.insert(node->outputs->data[0]);

        // Skip this node for now. If output of the node is consumed only by
        // delegated nodes, it will be added to nodes_to_delegate in the end.
        continue;
      }
    }

    if (Subgraph::VisitNode(/*subgraph=*/nullptr, context, registration, node,
                            node_index, quasi_static_tensors,
                            std::vector<uint32_t>()) != kTfLiteOk) {
      // If a non-delegated node consumes output of a node that unpacks static
      // data, that node shouldn't be delegated.
      for (int j = 0; j < node->inputs->size; j++) {
        const auto it =
            quasi_static_tensors_producers.find(node->inputs->data[j]);
        if (it != quasi_static_tensors_producers.end()) {
          static_unpack_nodes_.erase(it->second);
        }
      }

      // Non-delegatable node is not an error.
      continue;
    }

    for (int j = 0; j < node->inputs->size; j++) {
      if (quasi_static_tensors.count(node->inputs->data[j]) != 0) {
        quasi_static_tensors_to_unpack.insert(node->inputs->data[j]);
      }
    }

    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }

  // Unpack static data of all tensors
  for (int t : quasi_static_tensors_to_unpack) {
    const int producer_index = quasi_static_tensors_producers[t];
    // Check if TFLite nodes can be delegated to XNNPACK
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, producer_index, &node,
                                        &registration) != kTfLiteOk) {
      TF_LITE_KERNEL_LOG(context,
                         "Unable to get node and registration for node %d.",
                         producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->inputs->size != 1) {
      TF_LITE_KERNEL_LOG(context, "unexpected number of inputs (%d) in node %d",
                         node->inputs->size, producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    if (node->outputs->size != 1) {
      TF_LITE_KERNEL_LOG(context,
                         "unexpected number of outputs (%d) in node %d",
                         node->outputs->size, producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    const TfLiteTensor& input_tensor = context->tensors[node->inputs->data[0]];
    if (input_tensor.allocation_type != kTfLiteMmapRo) {
      TF_LITE_KERNEL_LOG(context,
                         "unexpected allocation type in tensor %d in node %d",
                         node->inputs->data[0], producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }

    const TfLiteTensor& output_tensor = context->tensors[t];
    if (output_tensor.type != kTfLiteFloat32) {
      TF_LITE_KERNEL_LOG(context,
                         "unexpected datatype (%s) in tensor %d in node %d",
                         TfLiteTypeGetName(output_tensor.type),
                         node->outputs->data[0], producer_index);
      TfLiteIntArrayFree(nodes_to_delegate);
      return nullptr;  // Hard error.
    }
    const size_t tensor_elements = output_tensor.bytes / sizeof(float);

    // Align to XNN_EXTRA_BYTES bytes
    while (static_unpacked_data_.size() % XNN_EXTRA_BYTES != 0) {
      static_unpacked_data_.push_back(0);
    }
    const size_t tensor_offset = static_unpacked_data_.size();
    static_unpacked_data_.resize(tensor_offset + context->tensors[t].bytes);

    float* unpacked_data =
        reinterpret_cast<float*>(static_unpacked_data_.data() + tensor_offset);
    switch (registration->builtin_code) {
      case kTfLiteBuiltinDequantize: {
        if (input_tensor.type != kTfLiteFloat16) {
          TF_LITE_KERNEL_LOG(
              context, "unexpected tensor %d data type (%s) in node %d",
              node->inputs->data[0], TfLiteTypeGetName(input_tensor.type),
              producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        if (input_tensor.sparsity != nullptr) {
          TF_LITE_KERNEL_LOG(context,
                             "unexpected FP16 sparse tensor %d in node %d",
                             node->inputs->data[0], producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        const uint16_t* packed_data =
            static_cast<const uint16_t*>(input_tensor.data.data);
        for (size_t i = 0; i < tensor_elements; i++) {
          unpacked_data[i] = fp16_ieee_to_fp32_value(packed_data[i]);
        }
        break;
      }
      case kTfLiteBuiltinDensify: {
        if (input_tensor.type != kTfLiteFloat32) {
          TF_LITE_KERNEL_LOG(
              context, "unexpected tensor %d data type (%s) in node %d",
              node->inputs->data[0], TfLiteTypeGetName(input_tensor.type),
              producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        if (input_tensor.sparsity == nullptr) {
          TF_LITE_KERNEL_LOG(context, "unexpected dense tensor %d in node %d",
                             node->inputs->data[0], producer_index);
          TfLiteIntArrayFree(nodes_to_delegate);
          return nullptr;  // Hard error.
        }

        const int dims_count = output_tensor.dims->size;
        std::vector<int> vector_shape(dims_count);
        for (int i = 0; i < dims_count; i++) {
          vector_shape[i] = output_tensor.dims->data[i];
        }

        tflite::optimize::sparsity::FormatConverter<float> converter(
            vector_shape, *input_tensor.sparsity);
        converter.SparseToDense(input_tensor.data.f);
        const std::vector<float> out = converter.GetData();
        for (int i = 0; i < out.size(); i++) {
          unpacked_data[i] = out[i];
        }

        break;
      }
      default:
        TF_LITE_KERNEL_LOG(context, "unexpected op registration %d at node %d",
                           registration->builtin_code, producer_index);
        TfLiteIntArrayFree(nodes_to_delegate);
        return nullptr;  // Hard error.
    }

    static_unpacked_data_map_[t] = tensor_offset;
  }

  // Add nodes that unpack static data consumed by delegated nodes.
  // Note: this is done purely to avoid the overhead of running these nodes
  // again in TFLite interpreter which would allocate memory for their outputs.
  // We mark them as delegated, but the delegate would simply ignore these nodes
  // as the static weights are already unpacked.
  for (int node_index : static_unpack_nodes_) {
    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }
  std::sort(&nodes_to_delegate->data[0],
            &nodes_to_delegate->data[nodes_to_delegate->size]);

#ifdef XNNPACK_DELEGATE_TEST_MODE
  // In the test mode build (used by unit tests), XNNPACK delegate claims to
  // support all operators in the execution plan to disable fallback to the
  // default TensorFlow Lite kernels. Thus, if any of the ops in the model are
  // not supported by the delegate, they will cause a failure in
  // ::tflite::Interpreter::ModifyGraphWithDelegate, to be caught in the unit
  // tests.
  nodes_to_delegate->size = execution_plan->size;
  std::copy(&execution_plan->data[0],
            &execution_plan->data[execution_plan->size],
            &nodes_to_delegate->data[0]);
#endif

  return nodes_to_delegate;
}

void* SubgraphInit(TfLiteContext* context, const char* buffer, size_t length) {
  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);

  return static_cast<void*>(Subgraph::Create(
      context, params,
      static_cast<::tflite::xnnpack::Delegate*>(params->delegate->data_)));
}

TfLiteStatus SubgraphPrepare(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Prepare(context);
}

TfLiteStatus SubgraphInvoke(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

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
  TfLiteIntArray* ops_to_replace =
      static_cast<::tflite::xnnpack::Delegate*>(delegate->data_)
          ->PrepareOpsToDelegate(context);
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
