/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "slinky/base/thread_pool.h"  // from @slinky
#include "slinky/base/thread_pool_impl.h"  // from @slinky
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/delegates/ynnpack/attention.h"
#include "tensorflow/lite/delegates/ynnpack/copy.h"
#include "tensorflow/lite/delegates/ynnpack/dot.h"
#include "tensorflow/lite/delegates/ynnpack/elementwise.h"
#include "tensorflow/lite/delegates/ynnpack/moe.h"
#include "tensorflow/lite/delegates/ynnpack/pooling.h"
#include "tensorflow/lite/delegates/ynnpack/reduction.h"
#include "tensorflow/lite/delegates/ynnpack/softmax.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ynnpack {
namespace {

void MarkDeadSubgraphTreeSkippable(
    const std::vector<std::unique_ptr<tflite::Subgraph>>& subgraphs,
    int root_idx, std::vector<bool>& skippable_subgraphs) {
  std::vector<bool> visited(skippable_subgraphs.size(), false);
  std::vector<int> worklist = {root_idx};
  while (!worklist.empty()) {
    const int idx = worklist.back();
    worklist.pop_back();
    if (idx <= 0 || static_cast<size_t>(idx) >= skippable_subgraphs.size() ||
        visited[idx]) {
      continue;
    }
    visited[idx] = true;
    skippable_subgraphs[idx] = true;
    const tflite::Subgraph* dead_subgraph = subgraphs[idx].get();
    if (dead_subgraph == nullptr) {
      continue;
    }
    const auto& plan = dead_subgraph->pre_delegation_execution_plan().empty()
                           ? dead_subgraph->execution_plan()
                           : dead_subgraph->pre_delegation_execution_plan();
    for (int node_index : plan) {
      const auto* node_and_reg =
          dead_subgraph->node_and_registration(node_index);
      if (node_and_reg == nullptr ||
          node_and_reg->first.builtin_data == nullptr) {
        continue;
      }
      const TfLiteNode& node = node_and_reg->first;
      switch (node_and_reg->second.builtin_code) {
        case kTfLiteBuiltinStablehloComposite: {
          const auto* params =
              static_cast<const TfLiteStablehloCompositeParams*>(
                  node.builtin_data);
          worklist.push_back(params->subgraph_index);
          break;
        }
        case kTfLiteBuiltinWhile: {
          const auto* params =
              static_cast<const TfLiteWhileParams*>(node.builtin_data);
          worklist.push_back(params->cond_subgraph_index);
          worklist.push_back(params->body_subgraph_index);
          break;
        }
        case kTfLiteBuiltinIf: {
          const auto* params =
              static_cast<const TfLiteIfParams*>(node.builtin_data);
          worklist.push_back(params->then_subgraph_index);
          worklist.push_back(params->else_subgraph_index);
          break;
        }
        default:
          break;
      }
    }
  }
}

}  // namespace

class YNNPackDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit YNNPackDelegateKernel(const TfLiteYNNPackDelegateOptions& options,
                                 slinky::thread_pool* thread_pool)
      : options_(options),
        thread_pool_(thread_pool),
        subgraph_(nullptr),
        runtime_(nullptr) {}

  ~YNNPackDelegateKernel() override {
    if (runtime_) ynn_delete_runtime(runtime_);
    if (subgraph_) ynn_delete_subgraph(subgraph_);
  }

  TfLiteStatus BuildSubgraphAndRuntime(TfLiteContext* context) {
    // The constants folded by the previous runtime (e.g. the packed weights)
    // are only reused by the new subgraph's constant folding while that
    // runtime is alive, so keep it until the new subgraph is optimized.
    // Rebuilds happen at runtime, e.g. when an MoE node switches between
    // bucketing tokens on the host and gathering weights.
    std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)> old_runtime(
        runtime_, ynn_delete_runtime);
    runtime_ = nullptr;
    if (subgraph_) {
      ynn_delete_subgraph(subgraph_);
      subgraph_ = nullptr;
    }
    tensor_to_value_id_.clear();
    inputs_.clear();
    outputs_.clear();
    dummy_inputs_.clear();
    input_shapes_.clear();
    moe_dispatches_.clear();

    int num_extra_external_inputs = 0;
    for (const auto& node : nodes_info_) {
      if (node.composite_op_type == CompositeOpType::kRuntimeBmm &&
          node.inputs.size() >= 3) {
        num_extra_external_inputs += 2;
      } else if (node.composite_op_type == CompositeOpType::kSdpa) {
        num_extra_external_inputs += 4;
      } else if (node.composite_op_type == CompositeOpType::kMoe &&
                 ShouldDispatchMoeOnHost(context, node, input_tensor_indices_,
                                         options_)) {
        num_extra_external_inputs += kNumMoeDispatchExternalValues;
      }
    }

    int external_value_ids = input_tensor_indices_.size() +
                             output_tensor_indices_.size() +
                             num_extra_external_inputs;
    uint32_t subgraph_flags = 0;
    if (options_.fast_math) {
      subgraph_flags |= YNN_FLAG_FAST_MATH;
    }
    if (options_.consistent_arithmetic) {
      subgraph_flags |= YNN_FLAG_CONSISTENT_ARITHMETIC;
    }
    if (options_.no_excess_precision) {
      subgraph_flags |= YNN_FLAG_NO_EXCESS_PRECISION;
    }
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_create_subgraph(external_value_ids, subgraph_flags, &subgraph_));

    uint32_t next_external_id = 0;

    // Define input tensors of the partition as external inputs.
    input_shapes_.resize(input_tensor_indices_.size());
    for (size_t i = 0; i < input_tensor_indices_.size(); ++i) {
      int tensor_index = input_tensor_indices_[i];
      const TfLiteTensor& tensor = context->tensors[tensor_index];
      uint32_t val_id = next_external_id++;

      ynn_type ynn_type = GetYnnType(tensor.type);
      if (ynn_type == ynn_type_invalid) {
        continue;
      }

      TF_LITE_ENSURE_MSG(context, tensor.dims->size <= YNN_MAX_TENSOR_RANK,
                         "Tensor %d rank %d exceeds max %d", tensor_index,
                         tensor.dims->size, YNN_MAX_TENSOR_RANK);
      if (IsConstant(tensor, options_.static_shape)) {
        size_t dims[YNN_MAX_TENSOR_RANK] = {0};
        std::copy_n(tensor.dims->data, tensor.dims->size, dims);
        TF_LITE_ENSURE_YNN_STATUS(ynn_define_tensor(
            subgraph_, ynn_type, tensor.dims->size, dims, tensor.data.raw,
            /*flags=*/0, &val_id));
      } else {
        size_t dims[YNN_MAX_TENSOR_RANK] = {0};
        const size_t* dims_ptr = nullptr;
        if (options_.static_shape) {
          std::copy_n(tensor.dims->data, tensor.dims->size, dims);
          dims_ptr = dims;
        }
        TF_LITE_ENSURE_YNN_STATUS(
            ynn_define_tensor(subgraph_, ynn_type, tensor.dims->size, dims_ptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &val_id));
        inputs_.push_back({tensor_index, val_id});
      }
      input_shapes_[i].assign(tensor.dims->data,
                              tensor.dims->data + tensor.dims->size);

      tensor_to_value_id_[tensor_index] = val_id;
    }

    // Define output tensors of the partition as external outputs.
    outputs_.reserve(output_tensor_indices_.size());
    for (size_t i = 0; i < output_tensor_indices_.size(); ++i) {
      int tensor_index = output_tensor_indices_[i];
      const TfLiteTensor& tensor = context->tensors[tensor_index];
      uint32_t val_id = next_external_id++;

      ynn_type ynn_type = GetYnnType(tensor.type);
      TF_LITE_ENSURE_MSG(context, ynn_type != ynn_type_invalid,
                         "Unsupported type %d for output tensor %d in Build",
                         tensor.type, tensor_index);

      TF_LITE_ENSURE_MSG(context, tensor.dims->size <= YNN_MAX_TENSOR_RANK,
                         "Tensor %d rank %d exceeds max %d", tensor_index,
                         tensor.dims->size, YNN_MAX_TENSOR_RANK);
      size_t dims[YNN_MAX_TENSOR_RANK] = {0};
      std::copy_n(tensor.dims->data, tensor.dims->size, dims);

      TF_LITE_ENSURE_YNN_STATUS(
          ynn_define_tensor(subgraph_, ynn_type, tensor.dims->size, dims,
                            nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &val_id));
      outputs_.push_back({tensor_index, val_id});

      tensor_to_value_id_[tensor_index] = val_id;
    }

    // Now define internal nodes.
    for (size_t i = 0; i < nodes_info_.size(); ++i) {
      const auto& node = nodes_info_[i];
      if (IsUnaryOp(node.builtin_code)) {
        TF_LITE_ENSURE_STATUS(
            DefineUnaryNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinSoftmax) {
        TF_LITE_ENSURE_STATUS(
            DefineSoftmaxNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinLogSoftmax) {
        TF_LITE_ENSURE_STATUS(DefineLogSoftmaxNode(context, subgraph_,
                                                   tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinMaxPool2d) {
        TF_LITE_ENSURE_STATUS(
            DefineMaxPool2DNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinAveragePool2d) {
        TF_LITE_ENSURE_STATUS(DefineAveragePool2DNode(
            context, subgraph_, tensor_to_value_id_, node));
      } else if (IsBinaryOp(node.builtin_code)) {
        TF_LITE_ENSURE_STATUS(
            DefineBinaryNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinTranspose) {
        TF_LITE_ENSURE_STATUS(
            DefineTransposeNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinGather) {
        TF_LITE_ENSURE_STATUS(
            DefineGatherNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinGatherNd) {
        TF_LITE_ENSURE_STATUS(
            DefineGatherNdNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinSlice ||
                 node.builtin_code == kTfLiteBuiltinStridedSlice) {
        TF_LITE_ENSURE_STATUS(
            DefineSliceNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinConcatenation) {
        TF_LITE_ENSURE_STATUS(DefineConcatenationNode(
            context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinReshape) {
        TF_LITE_ENSURE_STATUS(DefineReshapeNode(
            context, subgraph_, tensor_to_value_id_, node, options_));
      } else if (node.builtin_code == kTfLiteBuiltinExpandDims) {
        TF_LITE_ENSURE_STATUS(DefineExpandDimsNode(context, subgraph_,
                                                   tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinPad ||
                 node.builtin_code == kTfLiteBuiltinPadv2) {
        TF_LITE_ENSURE_STATUS(
            DefinePadNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinSplit) {
        TF_LITE_ENSURE_STATUS(
            DefineSplitNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinSpaceToDepth) {
        TF_LITE_ENSURE_STATUS(DefineSpaceToDepthNode(
            context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinDepthToSpace) {
        TF_LITE_ENSURE_STATUS(DefineDepthToSpaceNode(
            context, subgraph_, tensor_to_value_id_, node));
      } else if (node.composite_op_type == CompositeOpType::kRuntimeBmm) {
        TF_LITE_ENSURE_STATUS(DefineRuntimeBatchedMatMulNode(
            context, subgraph_, tensor_to_value_id_, next_external_id,
            dummy_inputs_, node));
      } else if (node.composite_op_type == CompositeOpType::kSdpa) {
        TF_LITE_ENSURE_STATUS(
            DefineSdpaNode(context, subgraph_, tensor_to_value_id_,
                           next_external_id, dummy_inputs_, node));
      } else if (node.builtin_code == kTfLiteBuiltinBatchMatmul) {
        TF_LITE_ENSURE_STATUS(DefineBatchMatMulNode(context, subgraph_,
                                                    tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinFullyConnected) {
        TF_LITE_ENSURE_STATUS(DefineFullyConnectedNode(
            context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinConv2d) {
        TF_LITE_ENSURE_STATUS(
            DefineConvNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinDepthwiseConv2d) {
        TF_LITE_ENSURE_STATUS(DefineDepthwiseConvNode(
            context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinSum ||
                 node.builtin_code == kTfLiteBuiltinReduceMin ||
                 node.builtin_code == kTfLiteBuiltinReduceMax ||
                 node.builtin_code == kTfLiteBuiltinMean) {
        TF_LITE_ENSURE_STATUS(
            DefineReductionNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinStablehloClamp) {
        TF_LITE_ENSURE_STATUS(DefineStablehloClampNode(
            context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinQuantize) {
        TF_LITE_ENSURE_STATUS(
            DefineQuantizeNode(context, subgraph_, tensor_to_value_id_, node));
      } else if (node.builtin_code == kTfLiteBuiltinDequantize) {
        TF_LITE_ENSURE_STATUS(DefineDequantizeNode(context, subgraph_,
                                                   tensor_to_value_id_, node));
      } else if (node.composite_op_type == CompositeOpType::kMoe) {
        MoeDispatch* dispatch = nullptr;
        if (ShouldDispatchMoeOnHost(context, node, input_tensor_indices_,
                                    options_)) {
          dispatch = &moe_dispatches_.emplace_back();
        }
        TF_LITE_ENSURE_STATUS(DefineMoeNode(context, subgraph_,
                                            tensor_to_value_id_,
                                            next_external_id, dispatch, node));
      } else {
        TF_LITE_ENSURE_MSG(context, false, "Unsupported op: %d",
                           node.builtin_code);
      }
    }

    ynn_threadpool_t ynn_tp = reinterpret_cast<ynn_threadpool_t>(thread_pool_);
    TF_LITE_ENSURE_YNN_STATUS(ynn_optimize_subgraph(subgraph_, ynn_tp, 0));
    old_runtime.reset();
    const ynn_status create_status =
        ynn_create_runtime(subgraph_, ynn_tp, 0, &runtime_);
    ynn_delete_subgraph(subgraph_);
    subgraph_ = nullptr;
    TF_LITE_ENSURE_YNN_STATUS(create_status);

    return kTfLiteOk;
  }

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    input_tensor_indices_.assign(
        params->input_tensors->data,
        params->input_tensors->data + params->input_tensors->size);
    output_tensor_indices_.assign(
        params->output_tensors->data,
        params->output_tensors->data + params->output_tensors->size);

    nodes_info_.clear();
    nodes_info_.reserve(params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      int node_index = params->nodes_to_replace->data[i];
      TfLiteNode* node;
      TfLiteRegistration* reg;
      TF_LITE_ENSURE_STATUS(
          context->GetNodeAndRegistration(context, node_index, &node, &reg));

      NodeInfo node_info;
      node_info.node_index = node_index;
      node_info.builtin_code = reg->builtin_code;
      node_info.inputs.assign(node->inputs->data,
                              node->inputs->data + node->inputs->size);
      node_info.outputs.assign(node->outputs->data,
                               node->outputs->data + node->outputs->size);
      node_info.activation = GetFusedActivation(reg, node);
      node_info.builtin_data = node->builtin_data;
      node_info.custom_name = reg->custom_name;
      node_info.custom_initial_data = node->custom_initial_data;
      node_info.custom_initial_data_size = node->custom_initial_data_size;
      if (IsRuntimeBmm(reg, node)) {
        node_info.composite_op_type = CompositeOpType::kRuntimeBmm;
      } else if (IsSdpa(reg, node)) {
        node_info.composite_op_type = CompositeOpType::kSdpa;
      } else if (IsMoe(reg, node)) {
        node_info.composite_op_type = CompositeOpType::kMoe;
      }
      nodes_info_.push_back(node_info);
    }

    return BuildSubgraphAndRuntime(context);
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    bool rebuild_required = false;
    for (size_t i = 0; i < input_tensor_indices_.size(); ++i) {
      const TfLiteTensor& tensor = context->tensors[input_tensor_indices_[i]];
      if (options_.static_shape) {
        if (tensor.dims->size != input_shapes_[i].size() ||
            !std::equal(tensor.dims->data,
                        tensor.dims->data + tensor.dims->size,
                        input_shapes_[i].begin())) {
          rebuild_required = true;
          break;
        }
      } else {
        if (tensor.dims->size != input_shapes_[i].size()) {
          rebuild_required = true;
          break;
        }
      }
    }

    if (!rebuild_required && !options_.static_shape) {
      // Whether an MoE node buckets its tokens by expert depends on the
      // number of tokens, so a resize can change the lowering a node should
      // use. Not rebuilding would still be correct (both lowerings compute
      // the same result); the rebuild is for performance. Only the number of
      // bucketing nodes is compared, so one node switching on while another
      // switches off is not detected.
      size_t num_moe_dispatches = 0;
      for (const auto& node_info : nodes_info_) {
        if (node_info.composite_op_type == CompositeOpType::kMoe &&
            ShouldDispatchMoeOnHost(context, node_info, input_tensor_indices_,
                                    options_)) {
          ++num_moe_dispatches;
        }
      }
      if (num_moe_dispatches != moe_dispatches_.size()) {
        rebuild_required = true;
      }
    }

    if (rebuild_required) {
      TF_LITE_ENSURE_STATUS(BuildSubgraphAndRuntime(context));
    }

    // Set input shapes in YNNPACK.
    for (const auto& input : inputs_) {
      const TfLiteTensor& tensor = context->tensors[input.tensor_index];
      size_t dims[YNN_MAX_TENSOR_RANK] = {0};
      std::copy_n(tensor.dims->data, tensor.dims->size, dims);
      TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
          runtime_, input.val_id, tensor.dims->size, dims));
    }

    for (const auto& dummy : dummy_inputs_) {
      TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
          runtime_, dummy.dummy_val_id, dummy.rank, dummy.full_dims));
    }

    for (const MoeDispatch& dispatch : moe_dispatches_) {
      TF_LITE_ENSURE_STATUS(SetMoeDispatchShapes(context, runtime_, dispatch));
    }

    TF_LITE_ENSURE_YNN_STATUS(ynn_reshape_runtime(runtime_));

    // Query output shapes from YNNPACK and resize TFLite tensors.
    for (const auto& output : outputs_) {
      TfLiteTensor& tensor = context->tensors[output.tensor_index];
      size_t rank = YNN_MAX_TENSOR_RANK;
      size_t dims[YNN_MAX_TENSOR_RANK] = {0};
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_get_external_value_shape(runtime_, output.val_id, &rank, dims));

      TfLiteIntArray* output_shape = TfLiteIntArrayCreate(rank);
      for (size_t d = 0; d < rank; ++d) {
        constexpr size_t kMaxTfLiteDim = std::numeric_limits<int>::max();
        TF_LITE_ENSURE(context, dims[d] < kMaxTfLiteDim);
        output_shape->data[d] = dims[d];
      }
      TF_LITE_ENSURE_STATUS(
          context->ResizeTensor(context, &tensor, output_shape));
    }

    return kTfLiteOk;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    bool reshape_required = false;
    for (MoeDispatch& dispatch : moe_dispatches_) {
      TF_LITE_ENSURE_STATUS(
          SetupMoeDispatch(context, runtime_, dispatch, &reshape_required));
    }

    if (!dummy_inputs_.empty()) {
      // Set shape for dummy inputs based on param_tensor data at each
      // invocation.
      for (const auto& dummy : dummy_inputs_) {
        const TfLiteTensor& param_tensor =
            context->tensors[dummy.param_tensor_index];
        size_t dims[YNN_MAX_TENSOR_RANK] = {0};
        std::copy_n(dummy.full_dims, dummy.rank, dims);
        size_t num_elements = tflite::NumElements(&param_tensor);
        if (num_elements > 0) {
          int64_t active_tokens = 0;
          size_t index = (num_elements >= 2) ? 1 : 0;
          if (param_tensor.type == kTfLiteInt32 &&
              param_tensor.data.raw != nullptr &&
              param_tensor.bytes >= (index + 1) * sizeof(int32_t)) {
            const int32_t* i32_data =
                reinterpret_cast<const int32_t*>(param_tensor.data.raw);
            active_tokens = i32_data[index];
          } else if (param_tensor.type == kTfLiteInt64 &&
                     param_tensor.data.raw != nullptr &&
                     param_tensor.bytes >= (index + 1) * sizeof(int64_t)) {
            const int64_t* i64_data =
                reinterpret_cast<const int64_t*>(param_tensor.data.raw);
            active_tokens = i64_data[index];
          }
          if (active_tokens > 0) {
            dims[dummy.seq_axis] =
                std::min<size_t>(active_tokens, dims[dummy.seq_axis]);
          }
        }
        TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_shape(
            runtime_, dummy.dummy_val_id, dummy.rank, dims));
      }
      reshape_required = true;
    }
    if (reshape_required) {
      TF_LITE_ENSURE_YNN_STATUS(ynn_reshape_runtime(runtime_));
    }

    for (MoeDispatch& dispatch : moe_dispatches_) {
      TF_LITE_ENSURE_STATUS(SetMoeDispatchData(context, runtime_, dispatch));
    }

    // Set input buffers.
    for (const auto& input : inputs_) {
      TfLiteTensor& tensor = context->tensors[input.tensor_index];
      TF_LITE_ENSURE_YNN_STATUS(
          ynn_set_external_value_data(runtime_, input.val_id, tensor.data.raw));
    }

    // Set output buffers.
    for (const auto& output : outputs_) {
      TfLiteTensor& tensor = context->tensors[output.tensor_index];
      TF_LITE_ENSURE_YNN_STATUS(ynn_set_external_value_data(
          runtime_, output.val_id, tensor.data.raw));
    }

    TF_LITE_ENSURE_YNN_STATUS(ynn_invoke_runtime(runtime_));
    return kTfLiteOk;
  }

 private:
  const TfLiteYNNPackDelegateOptions options_;
  slinky::thread_pool* thread_pool_ = nullptr;
  ynn_subgraph_t subgraph_;
  ynn_runtime_t runtime_;

  struct TensorMap {
    int tensor_index;
    uint32_t val_id;
  };
  std::vector<TensorMap> inputs_;
  std::vector<TensorMap> outputs_;

  std::vector<NodeInfo> nodes_info_;
  std::vector<int> input_tensor_indices_;
  std::vector<int> output_tensor_indices_;
  std::vector<std::vector<size_t>> input_shapes_;
  TensorToValueIdMap tensor_to_value_id_;
  std::vector<DummyInputInfo> dummy_inputs_;
  std::vector<MoeDispatch> moe_dispatches_;
};

class YNNPackDelegate : public SimpleDelegateInterface {
 public:
  explicit YNNPackDelegate(const TfLiteYNNPackDelegateOptions& options)
      : options_(options) {
    if (options_.num_threads > 1) {
      thread_pool_ =
          std::make_unique<slinky::thread_pool_impl>(options_.num_threads - 1);
    }
  }

  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    int builtin_code = registration->builtin_code;
    if (IsUnaryOp(builtin_code)) {
      return IsUnaryOpSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (IsBinaryOp(builtin_code)) {
      return IsBinaryOpSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinTranspose) {
      return IsTransposeSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinSlice ||
               builtin_code == kTfLiteBuiltinStridedSlice) {
      return IsSliceSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinConcatenation) {
      return IsConcatenationSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinReshape) {
      return IsReshapeSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinExpandDims) {
      return IsExpandDimsSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinPad ||
               builtin_code == kTfLiteBuiltinPadv2) {
      return IsPadSupported(registration, node, context, options_) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinSplit) {
      return IsSplitSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinGather) {
      return IsGatherSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinGatherNd) {
      return IsGatherNdSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinSpaceToDepth) {
      return IsSpaceToDepthSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinDepthToSpace) {
      return IsDepthToSpaceSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinBatchMatmul) {
      return IsBatchMatMulSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinFullyConnected) {
      return IsFullyConnectedSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinConv2d) {
      return IsConvSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinDepthwiseConv2d) {
      return IsDepthwiseConvSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinSoftmax ||
               builtin_code == kTfLiteBuiltinLogSoftmax) {
      return IsSoftmaxSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinMaxPool2d ||
               builtin_code == kTfLiteBuiltinAveragePool2d) {
      return IsPoolingSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinSum ||
               builtin_code == kTfLiteBuiltinReduceMin ||
               builtin_code == kTfLiteBuiltinReduceMax ||
               builtin_code == kTfLiteBuiltinMean) {
      return IsReductionSupported(registration, node, context, options_) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinStablehloClamp) {
      return IsStablehloClampSupported(registration, node, context) ==
             kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinQuantize) {
      return IsQuantizeSupported(registration, node, context) == kTfLiteOk;
    } else if (builtin_code == kTfLiteBuiltinDequantize) {
      return IsDequantizeSupported(registration, node, context) == kTfLiteOk;
    } else if (IsRuntimeBmm(registration, node)) {
      return IsRuntimeBatchedMatMulSupported(registration, node, context,
                                             options_) == kTfLiteOk;
    } else if (IsSdpa(registration, node)) {
      return IsSdpaSupported(registration, node, context) == kTfLiteOk;
    } else if (IsMoe(registration, node)) {
      return IsMoeSupported(registration, node, context) == kTfLiteOk;
    }
    return false;
  }

  // There is an issue with composite ops: if we leave composite ops we don't
  // support in the graph, TFlite will undo delegates, inline the composite ops,
  // and then redo-delegation. This is expensive, and also somehow causes
  // a performance issue at invocation time too (not just delegation). To avoid
  // this, we need to inline all the composite ops we don't support first.
  // TODO: b/541012735 - This might break other delegates that would have
  // supported the composite op without inlining.
  TfLiteStatus Initialize(TfLiteContext* context) override {
    auto* subgraph = reinterpret_cast<tflite::Subgraph*>(context->impl_);
    if (subgraph == nullptr) {
      return kTfLiteOk;
    }
    const auto* subgraphs = subgraph->GetSubgraphs();
    const size_t num_subgraphs = subgraphs != nullptr ? subgraphs->size() : 0;
    // Subgraph::InlineCompositeNodes calls UndoAllDelegates(), which resets
    // is_delegation_skippable_ across all subgraphs in the interpreter.
    // Preserve previously marked subgraphs and also mark composite
    // decomposition subgraphs that are delegated or inlined here.
    std::vector<bool> skippable_subgraphs(num_subgraphs, false);
    for (size_t i = 1; i < num_subgraphs; ++i) {
      if ((*subgraphs)[i] != nullptr &&
          (*subgraphs)[i]->IsDelegationSkippable()) {
        skippable_subgraphs[i] = true;
      }
    }
    auto filter = [context, this, subgraphs, &skippable_subgraphs](
                      const TfLiteNode* node,
                      const TfLiteRegistration* reg) -> bool {
      int decomp_idx = -1;
      if (reg->builtin_code == kTfLiteBuiltinStablehloComposite &&
          node->builtin_data != nullptr) {
        decomp_idx = static_cast<const TfLiteStablehloCompositeParams*>(
                         node->builtin_data)
                         ->subgraph_index;
      }
      if (IsNodeSupportedByDelegate(reg, node, context)) {
        // Don't inline this supported composite; its decomposition subgraph
        // (and any nested callee subgraphs inside it) is dead.
        if (subgraphs != nullptr && decomp_idx > 0) {
          MarkDeadSubgraphTreeSkippable(*subgraphs, decomp_idx,
                                        skippable_subgraphs);
        }
        return false;
      }
      if (decomp_idx > 0 &&
          static_cast<size_t>(decomp_idx) < skippable_subgraphs.size()) {
        skippable_subgraphs[decomp_idx] = true;
      }
      return true;
    };
    TF_LITE_ENSURE_STATUS(subgraph->InlineCompositeNodes(filter));
    for (size_t i = 1; i < skippable_subgraphs.size(); ++i) {
      if (skippable_subgraphs[i] && (*subgraphs)[i] != nullptr &&
          !(*subgraphs)[i]->IsDelegationSkippable()) {
        TF_LITE_ENSURE_STATUS(subgraph->MarkSubgraphAsDelegationSkippable(i));
      }
    }
    return kTfLiteOk;
  }

  const char* Name() const override {
    static constexpr char kName[] = "YNNPackDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<YNNPackDelegateKernel>(options_,
                                                   thread_pool_.get());
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    return SimpleDelegateInterface::Options();
  }

 private:
  const TfLiteYNNPackDelegateOptions options_;
  std::unique_ptr<slinky::thread_pool_impl> thread_pool_;
};

}  // namespace ynnpack
}  // namespace tflite

TfLiteYNNPackDelegateOptions TfLiteYNNPackDelegateOptionsDefault() {
  TfLiteYNNPackDelegateOptions options = {0};
  options.num_threads = 1;
  options.static_shape = false;
  options.fast_math = false;
  options.consistent_arithmetic = false;
  options.no_excess_precision = false;
  options.enable_blockwise_quantization = false;
  options.dispatch_moe_on_host = false;
  return options;
}

TfLiteDelegate* TfLiteYNNPackDelegateCreate(
    const TfLiteYNNPackDelegateOptions* options) {
  std::unique_ptr<tflite::ynnpack::YNNPackDelegate> ynnpack(
      new tflite::ynnpack::YNNPackDelegate(
          options ? *options : TfLiteYNNPackDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(
      std::move(ynnpack), kTfLiteDelegateFlagsAllowDynamicTensors);
}

void TfLiteYNNPackDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
