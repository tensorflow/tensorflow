
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
#include "tensorflow/lite/delegates/xnnpack/moe_delegate_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xnnpack.h"  // from @XNNPACK
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "pthreadpool.h"  // from @pthreadpool
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace xnnpack {
namespace {

constexpr char kMoeCustomOp[] = "moe";
constexpr uintptr_t kMoeXnnpackWorkspaceAlignment = 128;

struct MoeExpertsAttributes {
  int num_experts = 0;
  int num_active_experts = 0;
  int model_dim = 0;
  int hidden_dim = 0;
};

struct MoeExpertsAssignment {
  int token = 0;
  int route = 0;
};

}  // namespace

class MoeExpertsDelegateKernel::Impl {
 public:
  using XnnOperatorPtr =
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>;

  static bool IsMoeExpertsNode(const TfLiteRegistration* registration,
                               const TfLiteNode* node) {
    (void)node;
    return registration->builtin_code == kTfLiteBuiltinCustom &&
           registration->custom_name != nullptr &&
           std::strcmp(registration->custom_name, kMoeCustomOp) == 0;
  }

  static TfLiteStatus IsSupported(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  const TfLiteRegistration* registration,
                                  int node_index) {
    if (!IsMoeExpertsNode(registration, node)) {
      return kTfLiteError;
    }
    MoeExpertsAttributes attr;
    if (!ReadAttributes(context, node, registration, node_index, &attr)) {
      return kTfLiteError;
    }
    if (node->inputs == nullptr || node->outputs == nullptr) {
      TF_LITE_KERNEL_LOG(context, "%s node #%d has null inputs or outputs",
                         kMoeCustomOp, node_index);
      return kTfLiteError;
    }
    if (node->inputs->size != 7 || node->outputs->size != 1) {
      TF_LITE_KERNEL_LOG(
          context,
          "%s node #%d expects 7 fp32 inputs and 1 output in the XNNPACK "
          "prototype path",
          kMoeCustomOp, node_index);
      return kTfLiteError;
    }
    const TfLiteTensor* src = &context->tensors[node->inputs->data[0]];
    const TfLiteTensor* top_weights = &context->tensors[node->inputs->data[1]];
    const TfLiteTensor* top_indices = &context->tensors[node->inputs->data[2]];
    const TfLiteTensor* gate_weight = &context->tensors[node->inputs->data[3]];
    const TfLiteTensor* ff1_weight = &context->tensors[node->inputs->data[4]];
    const TfLiteTensor* linear_weight =
        &context->tensors[node->inputs->data[5]];
    const TfLiteTensor* per_expert_scale =
        &context->tensors[node->inputs->data[6]];
    const TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

    if (src->type != kTfLiteFloat32 || top_weights->type != kTfLiteFloat32 ||
        top_indices->type != kTfLiteInt32 ||
        gate_weight->type != kTfLiteFloat32 ||
        ff1_weight->type != kTfLiteFloat32 ||
        linear_weight->type != kTfLiteFloat32 ||
        per_expert_scale->type != kTfLiteFloat32 ||
        output->type != kTfLiteFloat32) {
      TF_LITE_KERNEL_LOG(context,
                         "%s node #%d currently supports fp32 weights, fp32 "
                         "activations, and int32 top_indices only",
                         kMoeCustomOp, node_index);
      return kTfLiteError;
    }
    if (gate_weight->allocation_type != kTfLiteMmapRo ||
        ff1_weight->allocation_type != kTfLiteMmapRo ||
        linear_weight->allocation_type != kTfLiteMmapRo ||
        per_expert_scale->allocation_type != kTfLiteMmapRo) {
      TF_LITE_KERNEL_LOG(context,
                         "%s node #%d expects constant expert weights and "
                         "per_expert_scale",
                         kMoeCustomOp, node_index);
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  static std::unique_ptr<Impl> Create(TfLiteContext* context,
                                      const TfLiteDelegateParams* params,
                                      pthreadpool_t threadpool) {
    if (params->nodes_to_replace == nullptr ||
        params->nodes_to_replace->size != 1) {
      return nullptr;
    }
    const int node_index = params->nodes_to_replace->data[0];
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      return nullptr;
    }
    if (node == nullptr ||
        IsSupported(context, node, registration, node_index) != kTfLiteOk) {
      return nullptr;
    }

    MoeExpertsAttributes attr;
    if (!ReadAttributes(context, node, registration, node_index, &attr)) {
      return nullptr;
    }

    XnnOperatorPtr gate_up_fc(nullptr, &xnn_delete_operator);
    XnnOperatorPtr linear_fc(nullptr, &xnn_delete_operator);
    if (!CreateDynamicFullyConnected(context, node_index, &gate_up_fc) ||
        !CreateDynamicFullyConnected(context, node_index, &linear_fc)) {
      return nullptr;
    }

    return std::unique_ptr<Impl>(new Impl(
        attr, node->inputs->data[0], node->inputs->data[1],
        node->inputs->data[2], node->inputs->data[3], node->inputs->data[4],
        node->inputs->data[5], node->inputs->data[6], node->outputs->data[0],
        std::move(gate_up_fc), std::move(linear_fc), threadpool));
  }

  TfLiteStatus Prepare(TfLiteContext* context) {
    const TfLiteTensor& src = context->tensors[src_id_];
    TfLiteTensor& output = context->tensors[output_id_];
    if (src.dims != nullptr && output.dims != nullptr &&
        !TfLiteIntArrayEqual(src.dims, output.dims)) {
      TfLiteIntArray* new_shape = TfLiteIntArrayCopy(src.dims);
      return context->ResizeTensor(context, &output, new_shape);
    }
    return kTfLiteOk;
  }

  TfLiteStatus Invoke(TfLiteContext* context) {
    const TfLiteTensor& src_tensor = context->tensors[src_id_];
    const TfLiteTensor& top_weights_tensor = context->tensors[top_weights_id_];
    const TfLiteTensor& top_indices_tensor = context->tensors[top_indices_id_];
    const TfLiteTensor& gate_weight_tensor = context->tensors[gate_weight_id_];
    const TfLiteTensor& ff1_weight_tensor = context->tensors[ff1_weight_id_];
    const TfLiteTensor& linear_weight_tensor =
        context->tensors[linear_weight_id_];
    const TfLiteTensor& per_expert_scale_tensor =
        context->tensors[per_expert_scale_id_];
    TfLiteTensor& output_tensor = context->tensors[output_id_];

    const int src_elements = NumElements(&src_tensor);
    if (src_elements % attr_.model_dim != 0) {
      TF_LITE_KERNEL_LOG(context, "%s src element count is not divisible by %d",
                         kMoeCustomOp, attr_.model_dim);
      return kTfLiteError;
    }
    const int tokens = src_elements / attr_.model_dim;
    if (NumElements(&top_weights_tensor) != tokens * attr_.num_active_experts ||
        NumElements(&top_indices_tensor) != tokens * attr_.num_active_experts ||
        NumElements(&output_tensor) != tokens * attr_.model_dim) {
      TF_LITE_KERNEL_LOG(context,
                         "%s runtime tensor sizes do not match parsed attrs",
                         kMoeCustomOp);
      return kTfLiteError;
    }

    const float* src = GetTensorData<float>(&src_tensor);
    const float* top_weights = GetTensorData<float>(&top_weights_tensor);
    const int32_t* top_indices = GetTensorData<int32_t>(&top_indices_tensor);
    const float* gate_weight = GetTensorData<float>(&gate_weight_tensor);
    const float* ff1_weight = GetTensorData<float>(&ff1_weight_tensor);
    const float* linear_weight = GetTensorData<float>(&linear_weight_tensor);
    const float* per_expert_scale =
        GetTensorData<float>(&per_expert_scale_tensor);
    float* output = GetTensorData<float>(&output_tensor);
    if (src == nullptr || top_weights == nullptr || top_indices == nullptr ||
        gate_weight == nullptr || ff1_weight == nullptr ||
        linear_weight == nullptr || per_expert_scale == nullptr ||
        output == nullptr) {
      TF_LITE_KERNEL_LOG(context, "%s received a null tensor data pointer",
                         kMoeCustomOp);
      return kTfLiteError;
    }

    std::fill(output, output + tokens * attr_.model_dim, 0.0f);
    const int dispatches = tokens * attr_.num_active_experts;
    if (!BuildExpertAssignments(context, top_indices, tokens, dispatches)) {
      return kTfLiteError;
    }

    for (int expert = 0; expert < attr_.num_experts; ++expert) {
      const int begin = expert_offsets_[expert];
      const int end = expert_offsets_[expert + 1];
      const int routed_tokens = end - begin;
      if (routed_tokens == 0) {
        continue;
      }
      if (!RunExpert(context, expert, assignments_.data() + begin,
                     routed_tokens, src, top_weights, gate_weight, ff1_weight,
                     linear_weight, per_expert_scale, output)) {
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

 private:
  Impl(MoeExpertsAttributes attr, int src_id, int top_weights_id,
       int top_indices_id, int gate_weight_id, int ff1_weight_id,
       int linear_weight_id, int per_expert_scale_id, int output_id,
       XnnOperatorPtr gate_up_fc, XnnOperatorPtr linear_fc,
       pthreadpool_t threadpool)
      : attr_(attr),
        src_id_(src_id),
        top_weights_id_(top_weights_id),
        top_indices_id_(top_indices_id),
        gate_weight_id_(gate_weight_id),
        ff1_weight_id_(ff1_weight_id),
        linear_weight_id_(linear_weight_id),
        per_expert_scale_id_(per_expert_scale_id),
        output_id_(output_id),
        gate_up_fc_(std::move(gate_up_fc)),
        linear_fc_(std::move(linear_fc)),
        threadpool_(threadpool) {}

  static std::optional<flexbuffers::Map> ReadAttributeMap(
      TfLiteContext* context, const TfLiteNode* node,
      const TfLiteRegistration* registration, int node_index) {
    if (registration->builtin_code == kTfLiteBuiltinCustom) {
      if (node->custom_initial_data == nullptr ||
          node->custom_initial_data_size == 0) {
        TF_LITE_KERNEL_LOG(context, "%s node #%d is missing custom options",
                           kMoeCustomOp, node_index);
        return std::nullopt;
      }
      return flexbuffers::GetRoot(
                 static_cast<const uint8_t*>(node->custom_initial_data),
                 node->custom_initial_data_size)
          .AsMap();
    }
    return std::nullopt;
  }

  static bool ReadAttributes(TfLiteContext* context, const TfLiteNode* node,
                             const TfLiteRegistration* registration,
                             int node_index, MoeExpertsAttributes* attr) {
    std::optional<flexbuffers::Map> map =
        ReadAttributeMap(context, node, registration, node_index);
    if (!map.has_value()) {
      return false;
    }
    for (const char* key : {"num_experts", "num_active_experts", "model_dim",
                            "hidden_dim", "weight_type"}) {
      if ((*map)[key].IsNull()) {
        TF_LITE_KERNEL_LOG(context, "%s node #%d is missing attribute %s",
                           kMoeCustomOp, node_index, key);
        return false;
      }
    }
    const std::string weight_type = (*map)["weight_type"].AsString().str();
    if (weight_type != "fp32") {
      TF_LITE_KERNEL_LOG(context,
                         "%s node #%d has unsupported weight_type '%s' for the "
                         "XNNPACK prototype path",
                         kMoeCustomOp, node_index, weight_type.c_str());
      return false;
    }
    if (!(*map)["activation"].IsNull() &&
        (*map)["activation"].AsString().str() != "gelu") {
      TF_LITE_KERNEL_LOG(context, "%s node #%d only supports activation='gelu'",
                         kMoeCustomOp, node_index);
      return false;
    }
    attr->num_experts = (*map)["num_experts"].AsInt32();
    attr->num_active_experts = (*map)["num_active_experts"].AsInt32();
    attr->model_dim = (*map)["model_dim"].AsInt32();
    attr->hidden_dim = (*map)["hidden_dim"].AsInt32();
    if (attr->num_experts <= 0 || attr->num_active_experts <= 0 ||
        attr->num_active_experts > attr->num_experts || attr->model_dim <= 0 ||
        attr->hidden_dim <= 0) {
      TF_LITE_KERNEL_LOG(context, "%s node #%d has invalid dimensions",
                         kMoeCustomOp, node_index);
      return false;
    }
    return true;
  }

  static bool CreateDynamicFullyConnected(TfLiteContext* context,
                                          int node_index, XnnOperatorPtr* op) {
    xnn_operator_t raw_op = nullptr;
    const xnn_status status = xnn_create_dynamic_fully_connected_nc_f32(
        /*output_min=*/-std::numeric_limits<float>::infinity(),
        /*output_max=*/+std::numeric_limits<float>::infinity(),
        /*flags=*/0, &raw_op);
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context,
                         "failed to create XNNPACK dynamic FC for %s node #%d",
                         kMoeCustomOp, node_index);
      return false;
    }
    *op = XnnOperatorPtr(raw_op, &xnn_delete_operator);
    return true;
  }

  static float Gelu(float x) {
    // TODO: lower this to xnn unary gelu once the expert body is expressed
    // as a subgraph instead of host-stitched dynamic FC calls.
    return 0.5f * x * std::erfc(x * -0.70710678118654752440f);
  }

  static void* AlignWorkspace(void* ptr) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned = (address + kMoeXnnpackWorkspaceAlignment - 1) &
                              ~(kMoeXnnpackWorkspaceAlignment - 1);
    return reinterpret_cast<void*>(aligned);
  }

  template <typename T>
  static void EnsureSize(std::vector<T>* buffer, size_t size) {
    if (buffer->size() < size) {
      buffer->resize(size);
    }
  }

  bool BuildExpertAssignments(TfLiteContext* context,
                              const int32_t* top_indices, int tokens,
                              int dispatches) {
    EnsureSize(&expert_counts_, attr_.num_experts);
    std::fill(expert_counts_.begin(),
              expert_counts_.begin() + attr_.num_experts, 0);
    EnsureSize(&normalized_experts_, dispatches);

    for (int token = 0; token < tokens; ++token) {
      for (int route = 0; route < attr_.num_active_experts; ++route) {
        const int dispatch = token * attr_.num_active_experts + route;
        int expert = top_indices[dispatch];
        if (expert < 0) {
          expert += attr_.num_experts;
        }
        if (expert < 0 || expert >= attr_.num_experts) {
          TF_LITE_KERNEL_LOG(context, "%s expert index %d is out of range",
                             kMoeCustomOp, expert);
          return false;
        }
        normalized_experts_[dispatch] = expert;
        ++expert_counts_[expert];
      }
    }

    EnsureSize(&expert_offsets_, attr_.num_experts + 1);
    expert_offsets_[0] = 0;
    for (int expert = 0; expert < attr_.num_experts; ++expert) {
      expert_offsets_[expert + 1] =
          expert_offsets_[expert] + expert_counts_[expert];
    }

    EnsureSize(&write_offsets_, attr_.num_experts);
    std::copy_n(expert_offsets_.begin(), attr_.num_experts,
                write_offsets_.begin());
    EnsureSize(&assignments_, dispatches);
    for (int token = 0; token < tokens; ++token) {
      for (int route = 0; route < attr_.num_active_experts; ++route) {
        const int dispatch = token * attr_.num_active_experts + route;
        const int expert = normalized_experts_[dispatch];
        assignments_[write_offsets_[expert]++] = {token, route};
      }
    }
    return true;
  }

  static void CopyExpertWeightRows(const float* weight, int num_experts,
                                   int expert, int output_channels,
                                   int input_channels, float* dst) {
    for (int out = 0; out < output_channels; ++out) {
      const float* src = weight + (out * num_experts + expert) * input_channels;
      std::memcpy(dst + out * input_channels, src,
                  input_channels * sizeof(float));
    }
  }

  void CopyGateUpExpertWeight(const float* gate_weight, const float* ff1_weight,
                              int expert) {
    const int rows = 2 * attr_.hidden_dim;
    EnsureSize(&kernel_buffer_, rows * attr_.model_dim);
    float* dst = kernel_buffer_.data();
    // TODO: replace this host-side row gather when the delegate can either
    // consume expert-major weights directly or lower this as a reusable gather.
    CopyExpertWeightRows(gate_weight, attr_.num_experts, expert,
                         attr_.hidden_dim, attr_.model_dim, dst);
    CopyExpertWeightRows(ff1_weight, attr_.num_experts, expert,
                         attr_.hidden_dim, attr_.model_dim,
                         dst + attr_.hidden_dim * attr_.model_dim);
  }

  void CopyExpertWeight(const float* weight, int expert, int output_channels,
                        int input_channels) {
    EnsureSize(&kernel_buffer_, output_channels * input_channels);
    // TODO: replace this host-side row gather when the xnn can either
    // consume expert-major weights directly or lower this as a reusable gather.
    CopyExpertWeightRows(weight, attr_.num_experts, expert, output_channels,
                         input_channels, kernel_buffer_.data());
  }

  bool RunDynamicFullyConnected(TfLiteContext* context, xnn_operator_t op,
                                int batch_size, int input_channels,
                                int output_channels, const float* input,
                                const float* kernel, float* output) {
    size_t workspace_size = 0;
    xnn_status status = xnn_reshape_dynamic_fully_connected_nc_f32(
        op, batch_size, input_channels, output_channels, input_channels,
        output_channels, &workspace_size, threadpool_);
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "%s failed to reshape dynamic FC",
                         kMoeCustomOp);
      return false;
    }
    if (workspace_size == 0) {
      TF_LITE_KERNEL_LOG(context, "%s dynamic FC returned empty workspace",
                         kMoeCustomOp);
      return false;
    }
    EnsureSize(&workspace_, workspace_size + kMoeXnnpackWorkspaceAlignment - 1);
    char* workspace = static_cast<char*>(AlignWorkspace(workspace_.data()));
    status = xnn_setup_dynamic_fully_connected_nc_f32(
        op, workspace, input, kernel, /*bias=*/nullptr, output);
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "%s failed to setup dynamic FC",
                         kMoeCustomOp);
      return false;
    }
    status = xnn_run_operator(op, threadpool_);
    if (status != xnn_status_success) {
      TF_LITE_KERNEL_LOG(context, "%s failed to run dynamic FC", kMoeCustomOp);
      return false;
    }
    return true;
  }

  bool RunExpert(TfLiteContext* context, int expert,
                 const MoeExpertsAssignment* expert_assignments,
                 int routed_tokens, const float* src, const float* top_weights,
                 const float* gate_weight, const float* ff1_weight,
                 const float* linear_weight, const float* per_expert_scale,
                 float* output) {
    EnsureSize(&routed_src_, routed_tokens * attr_.model_dim);
    EnsureSize(&gate_up_, routed_tokens * 2 * attr_.hidden_dim);
    EnsureSize(&hidden_, routed_tokens * attr_.hidden_dim);
    EnsureSize(&down_, routed_tokens * attr_.model_dim);

    // TODO: lower this token dispatch as a gather-style delegate op. Keeping it
    // here preserves correctness while XNNPACK lacks a ragged/grouped gather.
    for (int i = 0; i < routed_tokens; ++i) {
      const int token = expert_assignments[i].token;
      const float* token_src = src + token * attr_.model_dim;
      std::memcpy(routed_src_.data() + i * attr_.model_dim, token_src,
                  attr_.model_dim * sizeof(float));
    }

    CopyGateUpExpertWeight(gate_weight, ff1_weight, expert);
    if (!RunDynamicFullyConnected(context, gate_up_fc_.get(), routed_tokens,
                                  attr_.model_dim, 2 * attr_.hidden_dim,
                                  routed_src_.data(), kernel_buffer_.data(),
                                  gate_up_.data())) {
      return false;
    }

    for (int token = 0; token < routed_tokens; ++token) {
      const float* gate = gate_up_.data() + token * 2 * attr_.hidden_dim;
      const float* ff1 = gate + attr_.hidden_dim;
      float* hidden = hidden_.data() + token * attr_.hidden_dim;
      for (int dim = 0; dim < attr_.hidden_dim; ++dim) {
        hidden[dim] = Gelu(gate[dim]) * ff1[dim];
      }
    }

    CopyExpertWeight(linear_weight, expert, attr_.model_dim, attr_.hidden_dim);
    if (!RunDynamicFullyConnected(context, linear_fc_.get(), routed_tokens,
                                  attr_.hidden_dim, attr_.model_dim,
                                  hidden_.data(), kernel_buffer_.data(),
                                  down_.data())) {
      return false;
    }

    const float expert_scale = per_expert_scale[expert];
    // TODO: lower this route-weighted scatter-add as a delegate op once the
    // XNNPACK path has a reusable primitive for ragged MoE combine.
    for (int i = 0; i < routed_tokens; ++i) {
      const int token = expert_assignments[i].token;
      const int route = expert_assignments[i].route;
      const float route_scale =
          expert_scale * top_weights[token * attr_.num_active_experts + route];
      float* token_output = output + token * attr_.model_dim;
      const float* token_down = down_.data() + i * attr_.model_dim;
      for (int dim = 0; dim < attr_.model_dim; ++dim) {
        token_output[dim] += token_down[dim] * route_scale;
      }
    }
    return true;
  }

  MoeExpertsAttributes attr_;
  int src_id_ = -1;
  int top_weights_id_ = -1;
  int top_indices_id_ = -1;
  int gate_weight_id_ = -1;
  int ff1_weight_id_ = -1;
  int linear_weight_id_ = -1;
  int per_expert_scale_id_ = -1;
  int output_id_ = -1;
  XnnOperatorPtr gate_up_fc_{nullptr, &xnn_delete_operator};
  XnnOperatorPtr linear_fc_{nullptr, &xnn_delete_operator};
  pthreadpool_t threadpool_ = nullptr;
  std::vector<int> expert_counts_;
  std::vector<int> expert_offsets_;
  std::vector<int> write_offsets_;
  std::vector<int> normalized_experts_;
  std::vector<MoeExpertsAssignment> assignments_;
  std::vector<float> routed_src_;
  std::vector<float> gate_up_;
  std::vector<float> hidden_;
  std::vector<float> down_;
  std::vector<float> kernel_buffer_;
  std::vector<char> workspace_;
};

MoeExpertsDelegateKernel::MoeExpertsDelegateKernel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MoeExpertsDelegateKernel::~MoeExpertsDelegateKernel() = default;

bool MoeExpertsDelegateKernel::IsMoeExpertsNode(
    const TfLiteRegistration* registration, const TfLiteNode* node) {
  return Impl::IsMoeExpertsNode(registration, node);
}

TfLiteStatus MoeExpertsDelegateKernel::IsSupported(
    TfLiteContext* context, const TfLiteNode* node,
    const TfLiteRegistration* registration, int node_index) {
  return Impl::IsSupported(context, node, registration, node_index);
}

std::unique_ptr<MoeExpertsDelegateKernel> MoeExpertsDelegateKernel::Create(
    TfLiteContext* context, const TfLiteDelegateParams* params,
    pthreadpool_t threadpool) {
  std::unique_ptr<Impl> impl = Impl::Create(context, params, threadpool);
  if (impl == nullptr) {
    return nullptr;
  }
  return std::unique_ptr<MoeExpertsDelegateKernel>(
      new MoeExpertsDelegateKernel(std::move(impl)));
}

TfLiteStatus MoeExpertsDelegateKernel::Prepare(TfLiteContext* context) {
  return impl_->Prepare(context);
}

TfLiteStatus MoeExpertsDelegateKernel::Invoke(TfLiteContext* context) {
  return impl_->Invoke(context);
}

}  // namespace xnnpack
}  // namespace tflite
