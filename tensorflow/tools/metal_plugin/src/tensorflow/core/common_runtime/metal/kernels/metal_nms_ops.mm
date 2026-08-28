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

#include "tensorflow/core/common_runtime/metal/kernels/metal_kernels.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/metal/kernels/metal_kernel_util.h"
#include "tensorflow/core/common_runtime/metal/metal_platform.h"
#include "tensorflow/core/common_runtime/metal/metal_stream.h"

namespace tensorflow {
namespace metal {
namespace {

// Non-maximum suppression, V2 through V4.
//
// The algorithm is a sequential greedy scan: take the highest-scoring box that
// remains, emit it, discard everything that overlaps it too much, repeat. Each
// decision depends on every decision before it, so there is no parallel form
// of the loop itself, and the output length is not known until the loop ends.
//
// It therefore runs on the host, after waiting for the stream. On Apple
// Silicon that wait is the whole cost: the boxes are already in memory the CPU
// can read, so there is no transfer to add to it. Suppression is a handful of
// arithmetic per pair on a few thousand boxes, which the CPU does in less time
// than dispatching a kernel would take.

struct NmsOp {
  bool pad_to_max = false;
};

void* NmsOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new NmsOp();
  TF_Bool flag = 0;
  TF_OpKernelConstruction_GetAttrBool(ctx, "pad_to_max_output_size", &flag,
                                      status);
  if (TF_GetCode(status) == TF_OK) op->pad_to_max = flag != 0;
  TF_SetStatus(status, TF_OK, "");
  TF_DeleteStatus(status);
  return op;
}

void NmsOp_Delete(void* kernel) { delete static_cast<NmsOp*>(kernel); }

void WaitForStream(SP_Stream stream) {
  uint64_t target = 0;
  {
    absl::MutexLock lock(&stream->mu);
    target = stream->last_enqueued;
  }
  if (target > 0) {
    [stream->order_event waitUntilSignaledValue:target timeoutMS:UINT64_MAX];
  }
}

// TensorFlow's intersection over union, including its treatment of a box with
// no area, which suppresses nothing rather than dividing by zero.
float IntersectionOverUnion(const float* boxes, int i, int j) {
  const float ymin_i = std::min(boxes[4 * i], boxes[4 * i + 2]);
  const float xmin_i = std::min(boxes[4 * i + 1], boxes[4 * i + 3]);
  const float ymax_i = std::max(boxes[4 * i], boxes[4 * i + 2]);
  const float xmax_i = std::max(boxes[4 * i + 1], boxes[4 * i + 3]);
  const float ymin_j = std::min(boxes[4 * j], boxes[4 * j + 2]);
  const float xmin_j = std::min(boxes[4 * j + 1], boxes[4 * j + 3]);
  const float ymax_j = std::max(boxes[4 * j], boxes[4 * j + 2]);
  const float xmax_j = std::max(boxes[4 * j + 1], boxes[4 * j + 3]);
  const float area_i = (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const float area_j = (ymax_j - ymin_j) * (xmax_j - xmin_j);
  if (area_i <= 0.0f || area_j <= 0.0f) return 0.0f;
  const float ymin = std::max(ymin_i, ymin_j);
  const float xmin = std::max(xmin_i, xmin_j);
  const float ymax = std::min(ymax_i, ymax_j);
  const float xmax = std::min(xmax_i, xmax_j);
  const float intersection =
      std::max(ymax - ymin, 0.0f) * std::max(xmax - xmin, 0.0f);
  return intersection / (area_i + area_j - intersection);
}

// Reads a scalar that arrives in host memory.
bool ReadScalarFloat(TF_OpKernelContext* ctx, int index, float* out,
                     TF_Status* status) {
  ScopedTensor t;
  TF_GetInput(ctx, index, t.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  const void* data = TF_TensorData(t.get());
  if (data == nullptr || TF_TensorElementCount(t.get()) < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a suppression threshold has no data.");
    return false;
  }
  *out = *static_cast<const float*>(data);
  return true;
}

bool ReadScalarInt(TF_OpKernelContext* ctx, int index, int64_t* out,
                   TF_Status* status) {
  ScopedTensor t;
  TF_GetInput(ctx, index, t.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  const void* data = TF_TensorData(t.get());
  if (data == nullptr || TF_TensorElementCount(t.get()) < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: max_output_size has no data.");
    return false;
  }
  *out = TF_TensorType(t.get()) == TF_INT64
             ? *static_cast<const int64_t*>(data)
             : *static_cast<const int32_t*>(data);
  return true;
}

// `score_threshold_index` is -1 for V2, which has no score threshold, and
// `emit_valid` adds V4's count of the boxes actually selected.
void Nms_ComputeImpl(NmsOp* op, TF_OpKernelContext* ctx,
                     int score_threshold_index, bool emit_valid,
                     TF_Status* status) {
  ScopedTensor boxes_t, scores_t;
  TF_GetInput(ctx, 0, boxes_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, scores_t.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> box_shape = ShapeOf(boxes_t.get());
  const std::vector<int64_t> score_shape = ShapeOf(scores_t.get());
  if (box_shape.size() != 2 || box_shape[1] != 4 || score_shape.size() != 1 ||
      score_shape[0] != box_shape[0]) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: non-maximum suppression expects boxes of shape "
                 "[n, 4] and one score per box.");
    return;
  }
  const int num_boxes = static_cast<int>(box_shape[0]);

  int64_t max_output = 0;
  if (!ReadScalarInt(ctx, 2, &max_output, status)) return;
  float iou_threshold = 0.0f;
  if (!ReadScalarFloat(ctx, 3, &iou_threshold, status)) return;
  float score_threshold = -std::numeric_limits<float>::infinity();
  if (score_threshold_index >= 0 &&
      !ReadScalarFloat(ctx, score_threshold_index, &score_threshold, status)) {
    return;
  }
  if (max_output < 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: max_output_size must not be negative.");
    return;
  }

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  // The boxes and scores have to exist before they can be ranked.
  WaitForStream(stream);

  const float* boxes = static_cast<const float*>(TF_TensorData(boxes_t.get()));
  const float* scores =
      static_cast<const float*>(TF_TensorData(scores_t.get()));
  if ((boxes == nullptr || scores == nullptr) && num_boxes > 0) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: the suppression inputs have no storage.");
    return;
  }

  // Highest score first, and on a tie the lower index first, which is the
  // order TensorFlow's priority queue produces.
  std::vector<int> candidates;
  candidates.reserve(static_cast<size_t>(num_boxes));
  for (int i = 0; i < num_boxes; ++i) {
    if (scores[i] > score_threshold) candidates.push_back(i);
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [scores](int a, int b) {
                     if (scores[a] != scores[b]) return scores[a] > scores[b];
                     return a < b;
                   });

  std::vector<int32_t> selected;
  for (int index : candidates) {
    if (static_cast<int64_t>(selected.size()) >= max_output) break;
    bool keep = true;
    // Against the most recent selection first: a box that survives at all
    // usually survives against everything, and a box that does not is most
    // often killed by a neighbour selected just before it.
    for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
      if (IntersectionOverUnion(boxes, index, *it) > iou_threshold) {
        keep = false;
        break;
      }
    }
    if (keep) selected.push_back(static_cast<int32_t>(index));
  }

  const int64_t out_length =
      op->pad_to_max ? max_output : static_cast<int64_t>(selected.size());
  const std::vector<int64_t> out_shape = {out_length};
  ScopedTensor output;
  output.reset(TF_AllocateOutput(
      ctx, 0, TF_INT32, out_shape.data(), 1,
      static_cast<size_t>(out_length) * sizeof(int32_t), status));
  if (TF_GetCode(status) != TF_OK) return;
  int32_t* out = static_cast<int32_t*>(TF_TensorData(output.get()));
  if (out != nullptr) {
    for (int64_t i = 0; i < out_length; ++i) {
      out[i] = i < static_cast<int64_t>(selected.size())
                   ? selected[static_cast<size_t>(i)]
                   : 0;
    }
  }

  if (emit_valid) {
    ScopedTensor valid;
    valid.reset(
        TF_AllocateOutput(ctx, 1, TF_INT32, nullptr, 0, sizeof(int32_t),
                          status));
    if (TF_GetCode(status) != TF_OK) return;
    int32_t* count = static_cast<int32_t*>(TF_TensorData(valid.get()));
    if (count != nullptr) *count = static_cast<int32_t>(selected.size());
  }
}

#define METAL_NMS_COMPUTE(NAME, SCORE_INDEX, EMIT_VALID)                    \
  void NAME(void* kernel, TF_OpKernelContext* ctx) {                        \
    ScopedAutoreleasePool pool;                                             \
    TF_Status* status = TF_NewStatus();                                     \
    auto* op = static_cast<NmsOp*>(kernel);                                 \
    if (op == nullptr) {                                                    \
      TF_SetStatus(status, TF_INTERNAL,                                     \
                   "Metal: a suppression kernel has no state.");            \
    } else {                                                                \
      Nms_ComputeImpl(op, ctx, SCORE_INDEX, EMIT_VALID, status);            \
    }                                                                       \
    if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status); \
    TF_DeleteStatus(status);                                                \
  }

METAL_NMS_COMPUTE(NmsV2_Compute, -1, false)
METAL_NMS_COMPUTE(NmsV3_Compute, 4, false)
METAL_NMS_COMPUTE(NmsV4_Compute, 4, true)

#undef METAL_NMS_COMPUTE

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name,
              const std::vector<const char*>& host_inputs) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder = TF_NewKernelBuilder(
      op_name, kMetalDeviceType, &NmsOp_Create, compute, &NmsOp_Delete);
  TF_KernelBuilder_TypeConstraint(builder, "T", TF_FLOAT, status);
  for (const char* input : host_inputs) {
    TF_KernelBuilder_HostMemory(builder, input);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_RegisterKernelBuilder(name.c_str(), builder, status);
  } else {
    TF_DeleteKernelBuilder(builder);
  }
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalNmsKernels() {
  // The thresholds and the output bound are scalars the loop needs on the
  // host, so they are placed there rather than read back.
  Register("NonMaxSuppressionV2", &NmsV2_Compute, "MetalNonMaxSuppressionV2",
           {"max_output_size", "iou_threshold"});
  Register("NonMaxSuppressionV3", &NmsV3_Compute, "MetalNonMaxSuppressionV3",
           {"max_output_size", "iou_threshold", "score_threshold"});
  Register("NonMaxSuppressionV4", &NmsV4_Compute, "MetalNonMaxSuppressionV4",
           {"max_output_size", "iou_threshold", "score_threshold"});
}

}  // namespace metal
}  // namespace tensorflow
