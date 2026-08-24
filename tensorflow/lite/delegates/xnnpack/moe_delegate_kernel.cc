
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
constexpr int kInvalidTensorId = -1;
constexpr uintptr_t kMoeXnnpackWorkspaceAlignment = 128;

struct MoeExpertsAttributes {
  enum class WeightType {
    kFp32,
    kInt8,
    kInt4,
  };
  enum class Activation {
    kGelu,
    kGeluTanh,
  };

  int num_experts = 0;
  int num_active_experts = 0;
  int model_dim = 0;
  int hidden_dim = 0;
  WeightType weight_type = WeightType::kFp32;
  Activation activation = Activation::kGelu;
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
    const bool is_quantized =
        attr.weight_type == MoeExpertsAttributes::WeightType::kInt8 ||
        attr.weight_type == MoeExpertsAttributes::WeightType::kInt4;
    const int expected_inputs = is_quantized ? 10 : 7;
    if (node->inputs->size != expected_inputs || node->outputs->size != 1) {
      TF_LITE_KERNEL_LOG(context, "%s node #%d expects %d inputs and 1 output",
                         kMoeCustomOp, node_index, expected_inputs);
      return kTfLiteError;
    }
    const TfLiteTensor* src = &context->tensors[node->inputs->data[0]];
    const TfLiteTensor* top_weights = &context->tensors[node->inputs->data[1]];
    const TfLiteTensor* top_indices = &context->tensors[node->inputs->data[2]];
    const TfLiteTensor* output = &context->tensors[node->outputs->data[0]];

    if (src->type != kTfLiteFloat32 || top_weights->type != kTfLiteFloat32 ||
        top_indices->type != kTfLiteInt32 || output->type != kTfLiteFloat32) {
      TF_LITE_KERNEL_LOG(
          context,
          "%s node #%d requires fp32 activations and int32 top_indices",
          kMoeCustomOp, node_index);
      return kTfLiteError;
    }

    auto check_weight_tensor = [&](int idx, size_t expected_elements,
                                   const char* name) -> bool {
      const TfLiteTensor* t = &context->tensors[node->inputs->data[idx]];
      if (attr.weight_type == MoeExpertsAttributes::WeightType::kFp32) {
        if (t->type != kTfLiteFloat32) {
          TF_LITE_KERNEL_LOG(context, "%s node #%d %s requires fp32 type",
                             kMoeCustomOp, node_index, name);
          return false;
        }
      } else if (attr.weight_type == MoeExpertsAttributes::WeightType::kInt8) {
        if (t->type != kTfLiteInt8) {
          TF_LITE_KERNEL_LOG(context, "%s node #%d %s requires int8 type",
                             kMoeCustomOp, node_index, name);
          return false;
        }
      } else if (attr.weight_type == MoeExpertsAttributes::WeightType::kInt4) {
        if (t->type != kTfLiteInt4 && t->type != kTfLiteInt8) {
          TF_LITE_KERNEL_LOG(context,
                             "%s node #%d %s requires int4 or int8 type",
                             kMoeCustomOp, node_index, name);
          return false;
        }
      }
      if (t->allocation_type != kTfLiteMmapRo) {
        TF_LITE_KERNEL_LOG(
            context, "%s node #%d %s (input #%d) must be a constant tensor",
            kMoeCustomOp, node_index, name, idx);
        return false;
      }
      if (t->dims == nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "%s node #%d %s (input #%d) has null dimensions",
                           kMoeCustomOp, node_index, name, idx);
        return false;
      }
      const int count = NumElements(t);
      const int expected_count =
          (t->type == kTfLiteInt8 &&
           attr.weight_type == MoeExpertsAttributes::WeightType::kInt4)
              ? static_cast<int>((expected_elements + 1) / 2)
              : static_cast<int>(expected_elements);
      if (count != expected_count) {
        TF_LITE_KERNEL_LOG(
            context,
            "%s node #%d %s element count %d does not match expected %d",
            kMoeCustomOp, node_index, name, count, expected_count);
        return false;
      }
      return true;
    };

    auto check_scale_tensor = [&](int idx, size_t num_rows,
                                  const char* name) -> bool {
      const TfLiteTensor* t = &context->tensors[node->inputs->data[idx]];
      if (t->type != kTfLiteFloat32 || t->allocation_type != kTfLiteMmapRo ||
          t->dims == nullptr) {
        TF_LITE_KERNEL_LOG(context,
                           "%s node #%d %s must be a constant fp32 tensor",
                           kMoeCustomOp, node_index, name);
        return false;
      }
      const int count = NumElements(t);
      if (attr.weight_type == MoeExpertsAttributes::WeightType::kInt8) {
        if (count != static_cast<int>(num_rows)) {
          TF_LITE_KERNEL_LOG(
              context,
              "%s node #%d %s element count %d does not match rows %zu",
              kMoeCustomOp, node_index, name, count, num_rows);
          return false;
        }
      } else if (attr.weight_type == MoeExpertsAttributes::WeightType::kInt4) {
        if (count <= 0 || (static_cast<size_t>(count) % num_rows != 0)) {
          TF_LITE_KERNEL_LOG(
              context,
              "%s node #%d %s element count %d is not a multiple of rows %zu",
              kMoeCustomOp, node_index, name, count, num_rows);
          return false;
        }
      }
      return true;
    };

    auto check_per_expert_scale = [&](int idx, size_t expected_elements,
                                      const char* name) -> bool {
      const TfLiteTensor* t = &context->tensors[node->inputs->data[idx]];
      if (t->type != kTfLiteFloat32 || t->allocation_type != kTfLiteMmapRo ||
          t->dims == nullptr ||
          NumElements(t) != static_cast<int>(expected_elements)) {
        TF_LITE_KERNEL_LOG(context,
                           "%s node #%d %s invalid per_expert_scale tensor",
                           kMoeCustomOp, node_index, name);
        return false;
      }
      return true;
    };

    const size_t hidden_dim = static_cast<size_t>(attr.hidden_dim);
    const size_t model_dim = static_cast<size_t>(attr.model_dim);
    const size_t num_experts = static_cast<size_t>(attr.num_experts);

    const size_t gate_ff1_elements = hidden_dim * num_experts * model_dim;
    const size_t linear_elements = model_dim * num_experts * hidden_dim;
    const size_t gate_ff1_rows = hidden_dim * num_experts;
    const size_t linear_rows = model_dim * num_experts;
    const size_t per_expert_scales = num_experts;

    int idx = 3;
    if (!check_weight_tensor(idx++, gate_ff1_elements, "gate_weight")) {
      return kTfLiteError;
    }
    if (is_quantized &&
        !check_scale_tensor(idx++, gate_ff1_rows, "gate_scale")) {
      return kTfLiteError;
    }
    if (!check_weight_tensor(idx++, gate_ff1_elements, "ff1_weight")) {
      return kTfLiteError;
    }
    if (is_quantized &&
        !check_scale_tensor(idx++, gate_ff1_rows, "ff1_scale")) {
      return kTfLiteError;
    }
    if (!check_weight_tensor(idx++, linear_elements, "linear_weight")) {
      return kTfLiteError;
    }
    if (is_quantized &&
        !check_scale_tensor(idx++, linear_rows, "linear_scale")) {
      return kTfLiteError;
    }
    if (!check_per_expert_scale(idx++, per_expert_scales, "per_expert_scale")) {
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

    int gate_weight_id = kInvalidTensorId;
    int gate_scale_id = kInvalidTensorId;
    int ff1_weight_id = kInvalidTensorId;
    int ff1_scale_id = kInvalidTensorId;
    int linear_weight_id = kInvalidTensorId;
    int linear_scale_id = kInvalidTensorId;
    int per_expert_scale_id = kInvalidTensorId;
    if (attr.weight_type == MoeExpertsAttributes::WeightType::kInt8 ||
        attr.weight_type == MoeExpertsAttributes::WeightType::kInt4) {
      gate_weight_id = node->inputs->data[3];
      gate_scale_id = node->inputs->data[4];
      ff1_weight_id = node->inputs->data[5];
      ff1_scale_id = node->inputs->data[6];
      linear_weight_id = node->inputs->data[7];
      linear_scale_id = node->inputs->data[8];
      per_expert_scale_id = node->inputs->data[9];
    } else {
      gate_weight_id = node->inputs->data[3];
      ff1_weight_id = node->inputs->data[4];
      linear_weight_id = node->inputs->data[5];
      per_expert_scale_id = node->inputs->data[6];
    }

    return std::unique_ptr<Impl>(
        new Impl(attr, node->inputs->data[0], node->inputs->data[1],
                 node->inputs->data[2], gate_weight_id, gate_scale_id,
                 ff1_weight_id, ff1_scale_id, linear_weight_id, linear_scale_id,
                 per_expert_scale_id, node->outputs->data[0],
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
    const void* gate_weight = gate_weight_tensor.data.raw;
    const float* gate_scale =
        gate_scale_id_ != kInvalidTensorId
            ? GetTensorData<float>(&context->tensors[gate_scale_id_])
            : nullptr;
    const size_t gate_scale_elements =
        gate_scale_id_ != kInvalidTensorId
            ? static_cast<size_t>(
                  NumElements(&context->tensors[gate_scale_id_]))
            : 0;
    const void* ff1_weight = ff1_weight_tensor.data.raw;
    const float* ff1_scale =
        ff1_scale_id_ != kInvalidTensorId
            ? GetTensorData<float>(&context->tensors[ff1_scale_id_])
            : nullptr;
    const size_t ff1_scale_elements =
        ff1_scale_id_ != kInvalidTensorId
            ? static_cast<size_t>(NumElements(&context->tensors[ff1_scale_id_]))
            : 0;
    const void* linear_weight = linear_weight_tensor.data.raw;
    const float* linear_scale =
        linear_scale_id_ != kInvalidTensorId
            ? GetTensorData<float>(&context->tensors[linear_scale_id_])
            : nullptr;
    const size_t linear_scale_elements =
        linear_scale_id_ != kInvalidTensorId
            ? static_cast<size_t>(
                  NumElements(&context->tensors[linear_scale_id_]))
            : 0;
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
    if ((attr_.weight_type == MoeExpertsAttributes::WeightType::kInt8 ||
         attr_.weight_type == MoeExpertsAttributes::WeightType::kInt4) &&
        (gate_scale == nullptr || ff1_scale == nullptr ||
         linear_scale == nullptr)) {
      TF_LITE_KERNEL_LOG(context,
                         "%s quantized mode received null scale pointers",
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
                     routed_tokens, src, top_weights, gate_weight, gate_scale,
                     gate_scale_elements, ff1_weight, ff1_scale,
                     ff1_scale_elements, linear_weight, linear_scale,
                     linear_scale_elements, per_expert_scale, output)) {
        return kTfLiteError;
      }
    }
    return kTfLiteOk;
  }

 private:
  Impl(MoeExpertsAttributes attr, int src_id, int top_weights_id,
       int top_indices_id, int gate_weight_id, int gate_scale_id,
       int ff1_weight_id, int ff1_scale_id, int linear_weight_id,
       int linear_scale_id, int per_expert_scale_id, int output_id,
       XnnOperatorPtr gate_up_fc, XnnOperatorPtr linear_fc,
       pthreadpool_t threadpool)
      : attr_(attr),
        src_id_(src_id),
        top_weights_id_(top_weights_id),
        top_indices_id_(top_indices_id),
        gate_weight_id_(gate_weight_id),
        gate_scale_id_(gate_scale_id),
        ff1_weight_id_(ff1_weight_id),
        ff1_scale_id_(ff1_scale_id),
        linear_weight_id_(linear_weight_id),
        linear_scale_id_(linear_scale_id),
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
    if (weight_type == "fp32") {
      attr->weight_type = MoeExpertsAttributes::WeightType::kFp32;
    } else if (weight_type == "int8") {
      attr->weight_type = MoeExpertsAttributes::WeightType::kInt8;
    } else if (weight_type == "int4") {
      attr->weight_type = MoeExpertsAttributes::WeightType::kInt4;
    } else {
      TF_LITE_KERNEL_LOG(context,
                         "%s node #%d has unsupported weight_type '%s'",
                         kMoeCustomOp, node_index, weight_type.c_str());
      return false;
    }
    if (!(*map)["activation"].IsNull()) {
      const std::string act = (*map)["activation"].AsString().str();
      if (act == "gelu") {
        attr->activation = MoeExpertsAttributes::Activation::kGelu;
      } else if (act == "gelu_tanh") {
        attr->activation = MoeExpertsAttributes::Activation::kGeluTanh;
      } else {
        TF_LITE_KERNEL_LOG(context, "%s node #%d unsupported activation='%s'",
                           kMoeCustomOp, node_index, act.c_str());
        return false;
      }
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
    const uint64_t max_int = std::numeric_limits<int>::max();
    const uint64_t num_experts_64 = static_cast<uint64_t>(attr->num_experts);
    const uint64_t model_dim_64 = static_cast<uint64_t>(attr->model_dim);
    const uint64_t hidden_dim_64 = static_cast<uint64_t>(attr->hidden_dim);
    if (num_experts_64 * hidden_dim_64 > max_int ||
        num_experts_64 * model_dim_64 > max_int ||
        num_experts_64 * hidden_dim_64 * model_dim_64 > max_int) {
      TF_LITE_KERNEL_LOG(context,
                         "%s node #%d dimensions exceed maximum bounds",
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

  static float GeluTanh(float x) {
    const float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
    const float kBeta = 0.044715f;
    const float inner = kAlpha * x * (1.0f + kBeta * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
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
                              const int32_t* top_indices, size_t tokens,
                              size_t dispatches) {
    const size_t num_experts = static_cast<size_t>(attr_.num_experts);
    const size_t num_active_experts =
        static_cast<size_t>(attr_.num_active_experts);
    EnsureSize(&expert_counts_, num_experts);
    std::fill(expert_counts_.begin(), expert_counts_.begin() + num_experts, 0);
    EnsureSize(&normalized_experts_, dispatches);

    for (size_t token = 0; token < tokens; ++token) {
      for (size_t route = 0; route < num_active_experts; ++route) {
        const size_t dispatch = token * num_active_experts + route;
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

    EnsureSize(&expert_offsets_, num_experts + 1);
    expert_offsets_[0] = 0;
    for (size_t expert = 0; expert < num_experts; ++expert) {
      expert_offsets_[expert + 1] =
          expert_offsets_[expert] + expert_counts_[expert];
    }

    EnsureSize(&write_offsets_, num_experts);
    std::copy_n(expert_offsets_.begin(), num_experts, write_offsets_.begin());
    EnsureSize(&assignments_, dispatches);
    for (size_t token = 0; token < tokens; ++token) {
      for (size_t route = 0; route < num_active_experts; ++route) {
        const size_t dispatch = token * num_active_experts + route;
        const int expert = normalized_experts_[dispatch];
        assignments_[write_offsets_[expert]++] = {static_cast<int>(token),
                                                  static_cast<int>(route)};
      }
    }
    return true;
  }

  static void CopyExpertWeightRows(const float* weight, size_t num_experts,
                                   size_t expert, size_t output_channels,
                                   size_t input_channels, float* dst) {
    for (size_t out = 0; out < output_channels; ++out) {
      const size_t row_idx = out * num_experts + expert;
      const float* src = weight + row_idx * input_channels;
      std::memcpy(dst + out * input_channels, src,
                  input_channels * sizeof(float));
    }
  }

  static void CopyAndDequantizeExpertWeightRowsInt8(
      const int8_t* weight_i8, const float* scale, size_t num_experts,
      size_t expert, size_t output_channels, size_t input_channels,
      float* dst) {
    for (size_t out = 0; out < output_channels; ++out) {
      const size_t row_idx = out * num_experts + expert;
      const float row_scale = scale[row_idx];
      const int8_t* src_row = weight_i8 + row_idx * input_channels;
      float* dst_row = dst + out * input_channels;
      for (size_t in = 0; in < input_channels; ++in) {
        dst_row[in] = static_cast<float>(src_row[in]) * row_scale;
      }
    }
  }

  static void CopyAndDequantizeExpertWeightRowsInt4(
      const int8_t* weight_i4_packed, const float* scale, size_t scale_elements,
      size_t num_experts, size_t expert, size_t output_channels,
      size_t input_channels, float* dst) {
    const size_t num_rows = output_channels * num_experts;
    const size_t groups_per_row = scale_elements / num_rows;
    const size_t group_size =
        (groups_per_row > 0 && groups_per_row <= input_channels)
            ? (input_channels / groups_per_row)
            : 1;

    for (size_t out = 0; out < output_channels; ++out) {
      const size_t row_idx = out * num_experts + expert;
      const int8_t* src_row_packed =
          weight_i4_packed + (row_idx * input_channels) / 2;
      float* dst_row = dst + out * input_channels;
      const float* row_scales = scale + row_idx * groups_per_row;

      for (size_t in = 0; in < input_channels; ++in) {
        const size_t byte_idx = in / 2;
        const int8_t byte_val = src_row_packed[byte_idx];
        const int8_t nibble = (in % 2 == 0)
                                  ? static_cast<int8_t>(byte_val << 4) >> 4
                                  : static_cast<int8_t>(byte_val >> 4);
        const size_t group_idx =
            (groups_per_row > 1 && group_size > 0)
                ? std::min(in / group_size, groups_per_row - 1)
                : 0;
        const float scale_val = row_scales[group_idx];
        dst_row[in] = static_cast<float>(nibble) * scale_val;
      }
    }
  }

  void CopyGateUpExpertWeight(const void* gate_weight, const float* gate_scale,
                              size_t gate_scale_elements,
                              const void* ff1_weight, const float* ff1_scale,
                              size_t ff1_scale_elements, size_t expert) {
    const size_t hidden_dim = static_cast<size_t>(attr_.hidden_dim);
    const size_t model_dim = static_cast<size_t>(attr_.model_dim);
    const size_t num_experts = static_cast<size_t>(attr_.num_experts);
    const size_t rows = 2 * hidden_dim;
    EnsureSize(&kernel_buffer_, rows * model_dim);
    float* dst = kernel_buffer_.data();
    if (attr_.weight_type == MoeExpertsAttributes::WeightType::kInt4) {
      CopyAndDequantizeExpertWeightRowsInt4(
          static_cast<const int8_t*>(gate_weight), gate_scale,
          gate_scale_elements, num_experts, expert, hidden_dim, model_dim, dst);
      CopyAndDequantizeExpertWeightRowsInt4(
          static_cast<const int8_t*>(ff1_weight), ff1_scale, ff1_scale_elements,
          num_experts, expert, hidden_dim, model_dim,
          dst + hidden_dim * model_dim);
    } else if (attr_.weight_type == MoeExpertsAttributes::WeightType::kInt8) {
      CopyAndDequantizeExpertWeightRowsInt8(
          static_cast<const int8_t*>(gate_weight), gate_scale, num_experts,
          expert, hidden_dim, model_dim, dst);
      CopyAndDequantizeExpertWeightRowsInt8(
          static_cast<const int8_t*>(ff1_weight), ff1_scale, num_experts,
          expert, hidden_dim, model_dim, dst + hidden_dim * model_dim);
    } else {
      CopyExpertWeightRows(static_cast<const float*>(gate_weight), num_experts,
                           expert, hidden_dim, model_dim, dst);
      CopyExpertWeightRows(static_cast<const float*>(ff1_weight), num_experts,
                           expert, hidden_dim, model_dim,
                           dst + hidden_dim * model_dim);
    }
  }

  void CopyExpertWeight(const void* weight, const float* scale,
                        size_t scale_elements, size_t expert,
                        size_t output_channels, size_t input_channels) {
    const size_t num_experts = static_cast<size_t>(attr_.num_experts);
    EnsureSize(&kernel_buffer_, output_channels * input_channels);
    if (attr_.weight_type == MoeExpertsAttributes::WeightType::kInt4) {
      CopyAndDequantizeExpertWeightRowsInt4(
          static_cast<const int8_t*>(weight), scale, scale_elements,
          num_experts, expert, output_channels, input_channels,
          kernel_buffer_.data());
    } else if (attr_.weight_type == MoeExpertsAttributes::WeightType::kInt8) {
      CopyAndDequantizeExpertWeightRowsInt8(
          static_cast<const int8_t*>(weight), scale, num_experts, expert,
          output_channels, input_channels, kernel_buffer_.data());
    } else {
      CopyExpertWeightRows(static_cast<const float*>(weight), num_experts,
                           expert, output_channels, input_channels,
                           kernel_buffer_.data());
    }
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
                 const void* gate_weight, const float* gate_scale,
                 size_t gate_scale_elements, const void* ff1_weight,
                 const float* ff1_scale, size_t ff1_scale_elements,
                 const void* linear_weight, const float* linear_scale,
                 size_t linear_scale_elements, const float* per_expert_scale,
                 float* output) {
    const size_t model_dim = static_cast<size_t>(attr_.model_dim);
    const size_t hidden_dim = static_cast<size_t>(attr_.hidden_dim);
    const size_t num_active_experts =
        static_cast<size_t>(attr_.num_active_experts);
    const size_t tokens = static_cast<size_t>(routed_tokens);
    EnsureSize(&routed_src_, tokens * model_dim);
    EnsureSize(&gate_up_, tokens * 2 * hidden_dim);
    EnsureSize(&hidden_, tokens * hidden_dim);
    EnsureSize(&down_, tokens * model_dim);

    // TODO: lower this token dispatch as a gather-style delegate op. Keeping it
    // here preserves correctness while XNNPACK lacks a ragged/grouped gather.
    for (size_t i = 0; i < tokens; ++i) {
      const size_t token = static_cast<size_t>(expert_assignments[i].token);
      const float* token_src = src + token * model_dim;
      std::memcpy(routed_src_.data() + i * model_dim, token_src,
                  model_dim * sizeof(float));
    }

    CopyGateUpExpertWeight(gate_weight, gate_scale, gate_scale_elements,
                           ff1_weight, ff1_scale, ff1_scale_elements, expert);
    if (!RunDynamicFullyConnected(context, gate_up_fc_.get(), routed_tokens,
                                  attr_.model_dim, 2 * attr_.hidden_dim,
                                  routed_src_.data(), kernel_buffer_.data(),
                                  gate_up_.data())) {
      return false;
    }

    for (size_t token = 0; token < tokens; ++token) {
      const float* gate = gate_up_.data() + token * 2 * hidden_dim;
      const float* ff1 = gate + hidden_dim;
      float* hidden = hidden_.data() + token * hidden_dim;
      for (size_t dim = 0; dim < hidden_dim; ++dim) {
        float act_val =
            (attr_.activation == MoeExpertsAttributes::Activation::kGeluTanh)
                ? GeluTanh(gate[dim])
                : Gelu(gate[dim]);
        hidden[dim] = act_val * ff1[dim];
      }
    }

    CopyExpertWeight(linear_weight, linear_scale, linear_scale_elements, expert,
                     model_dim, hidden_dim);
    if (!RunDynamicFullyConnected(context, linear_fc_.get(), routed_tokens,
                                  attr_.hidden_dim, attr_.model_dim,
                                  hidden_.data(), kernel_buffer_.data(),
                                  down_.data())) {
      return false;
    }

    const float expert_scale = per_expert_scale[expert];
    // TODO: lower this route-weighted scatter-add as a delegate op once the
    // XNNPACK path has a reusable primitive for ragged MoE combine.
    for (size_t i = 0; i < tokens; ++i) {
      const size_t token = static_cast<size_t>(expert_assignments[i].token);
      const size_t route = static_cast<size_t>(expert_assignments[i].route);
      const float route_scale =
          expert_scale * top_weights[token * num_active_experts + route];
      float* token_output = output + token * model_dim;
      const float* token_down = down_.data() + i * model_dim;
      for (size_t dim = 0; dim < model_dim; ++dim) {
        token_output[dim] += token_down[dim] * route_scale;
      }
    }
    return true;
  }

  MoeExpertsAttributes attr_;
  int src_id_ = kInvalidTensorId;
  int top_weights_id_ = kInvalidTensorId;
  int top_indices_id_ = kInvalidTensorId;
  int gate_weight_id_ = kInvalidTensorId;
  int gate_scale_id_ = kInvalidTensorId;
  int ff1_weight_id_ = kInvalidTensorId;
  int ff1_scale_id_ = kInvalidTensorId;
  int linear_weight_id_ = kInvalidTensorId;
  int linear_scale_id_ = kInvalidTensorId;
  int per_expert_scale_id_ = kInvalidTensorId;
  int output_id_ = kInvalidTensorId;
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
