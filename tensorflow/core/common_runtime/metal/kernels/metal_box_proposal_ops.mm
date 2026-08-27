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
#include <cmath>
#include <cstdint>
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

// GenerateBoundingBoxProposals.
//
// A pipeline of selections: rank every anchor by its score, keep the best few,
// decode them into boxes, drop the ones that came out too small, suppress the
// ones that overlap a better box, and keep what is left. Every stage but the
// decode is a sequential selection whose output length depends on the values,
// which is the same shape of problem as non-maximum suppression, and it runs
// the same way: on the host, after waiting for the stream, over memory the CPU
// can already read.
//
// The decode is the one part that would parallelise, and it is a handful of
// operations on at most pre_nms_topn boxes per image, which is not worth a
// dispatch and a round trip of its own.

struct ProposalOp {
  int32_t post_nms_topn = 300;
};

void* ProposalOp_Create(TF_OpKernelConstruction* ctx) {
  TF_Status* status = TF_NewStatus();
  auto* op = new ProposalOp();
  int32_t value = 300;
  TF_OpKernelConstruction_GetAttrInt32(ctx, "post_nms_topn", &value, status);
  if (TF_GetCode(status) == TF_OK) op->post_nms_topn = value;
  TF_SetStatus(status, TF_OK, "");
  if (op->post_nms_topn <= 0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: post_nms_topn must be positive.");
    TF_OpKernelConstruction_Failure(ctx, status);
    TF_DeleteStatus(status);
    delete op;
    return nullptr;
  }
  TF_DeleteStatus(status);
  return op;
}

void ProposalOp_Delete(void* kernel) {
  delete static_cast<ProposalOp*>(kernel);
}

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

float Iou(const float* box, int i, int j) {
  const float* a = box + 4 * i;
  const float* b = box + 4 * j;
  const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
  const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
  if (area_a <= 0.0f || area_b <= 0.0f) return 0.0f;
  const float x1 = std::max(a[0], b[0]);
  const float y1 = std::max(a[1], b[1]);
  const float x2 = std::min(a[2], b[2]);
  const float y2 = std::min(a[3], b[3]);
  const float intersection =
      std::max(x2 - x1, 0.0f) * std::max(y2 - y1, 0.0f);
  return intersection / (area_a + area_b - intersection);
}

bool ReadHostScalar(TF_OpKernelContext* ctx, int index, double* out,
                    TF_Status* status) {
  ScopedTensor t;
  TF_GetInput(ctx, index, t.address(), status);
  if (TF_GetCode(status) != TF_OK) return false;
  const void* data = TF_TensorData(t.get());
  if (data == nullptr || TF_TensorElementCount(t.get()) < 1) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: a proposal parameter has no data.");
    return false;
  }
  switch (TF_TensorType(t.get())) {
    case TF_FLOAT:
      *out = *static_cast<const float*>(data);
      return true;
    case TF_INT32:
      *out = *static_cast<const int32_t*>(data);
      return true;
    case TF_INT64:
      *out = static_cast<double>(*static_cast<const int64_t*>(data));
      return true;
    default:
      TF_SetStatus(status, TF_INVALID_ARGUMENT,
                   "Metal: a proposal parameter has an unexpected type.");
      return false;
  }
}

void Proposals_ComputeImpl(ProposalOp* op, TF_OpKernelContext* ctx,
                           TF_Status* status) {
  ScopedTensor scores, deltas, image_info, anchors;
  TF_GetInput(ctx, 0, scores.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 1, deltas.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 2, image_info.address(), status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_GetInput(ctx, 3, anchors.address(), status);
  if (TF_GetCode(status) != TF_OK) return;

  const std::vector<int64_t> score_shape = ShapeOf(scores.get());
  if (score_shape.size() != 4) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: scores must have shape [batch, height, width, "
                 "anchors].");
    return;
  }
  const int64_t batch = score_shape[0];
  const int64_t height = score_shape[1];
  const int64_t width = score_shape[2];
  const int64_t num_anchors = score_shape[3];
  const int64_t per_image = height * width * num_anchors;

  double nms_threshold = 0.0, pre_nms_topn = 0.0, min_size = 0.0;
  if (!ReadHostScalar(ctx, 4, &nms_threshold, status)) return;
  if (!ReadHostScalar(ctx, 5, &pre_nms_topn, status)) return;
  if (!ReadHostScalar(ctx, 6, &min_size, status)) return;
  if (nms_threshold < 0.0 || nms_threshold > 1.0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: nms_threshold must lie between 0 and 1.");
    return;
  }
  if (pre_nms_topn <= 0.0) {
    TF_SetStatus(status, TF_INVALID_ARGUMENT,
                 "Metal: pre_nms_topn must be positive.");
    return;
  }

  const int64_t keep = op->post_nms_topn;
  const std::vector<int64_t> roi_shape = {batch, keep, 4};
  const std::vector<int64_t> prob_shape = {batch, keep};
  ScopedTensor rois, probs;
  rois.reset(TF_AllocateOutput(
      ctx, 0, TF_FLOAT, roi_shape.data(), 3,
      static_cast<size_t>(batch * keep * 4) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  probs.reset(TF_AllocateOutput(
      ctx, 1, TF_FLOAT, prob_shape.data(), 2,
      static_cast<size_t>(batch * keep) * sizeof(float), status));
  if (TF_GetCode(status) != TF_OK) return;
  if (batch == 0 || keep == 0) return;

  SP_Stream stream = StreamForContext(ctx, status);
  if (TF_GetCode(status) != TF_OK) return;
  // The scores decide which boxes exist at all, so they have to be readable.
  WaitForStream(stream);

  const float* score_data =
      static_cast<const float*>(TF_TensorData(scores.get()));
  const float* delta_data =
      static_cast<const float*>(TF_TensorData(deltas.get()));
  const float* info_data =
      static_cast<const float*>(TF_TensorData(image_info.get()));
  const float* anchor_data =
      static_cast<const float*>(TF_TensorData(anchors.get()));
  float* roi_out = static_cast<float*>(TF_TensorData(rois.get()));
  float* prob_out = static_cast<float*>(TF_TensorData(probs.get()));
  if (score_data == nullptr || delta_data == nullptr ||
      info_data == nullptr || anchor_data == nullptr || roi_out == nullptr ||
      prob_out == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: the proposal inputs have no storage.");
    return;
  }

  // The same bound on the width and height deltas the CUDA kernel uses, which
  // keeps a large delta from turning into an infinite box.
  const float delta_clip = std::log(1000.0f / 16.0f);
  const int64_t take =
      std::min<int64_t>(per_image, static_cast<int64_t>(pre_nms_topn));

  std::vector<int64_t> order;
  std::vector<float> boxes;
  std::vector<float> kept_scores;
  std::vector<int32_t> selected;
  for (int64_t image = 0; image < batch; ++image) {
    const float* image_scores = score_data + image * per_image;
    order.resize(static_cast<size_t>(per_image));
    for (int64_t i = 0; i < per_image; ++i) order[static_cast<size_t>(i)] = i;
    // Highest score first, with ties left in their original order, which is
    // what a radix sort by score gives on the CUDA path.
    std::stable_sort(order.begin(), order.end(),
                     [image_scores](int64_t a, int64_t b) {
                       return image_scores[a] > image_scores[b];
                     });

    const float img_height = info_data[5 * image + 0];
    const float img_width = info_data[5 * image + 1];
    const float min_size_scaled =
        static_cast<float>(min_size) * info_data[5 * image + 2];

    boxes.clear();
    kept_scores.clear();
    for (int64_t rank = 0; rank < take; ++rank) {
      const int64_t index = order[static_cast<size_t>(rank)];
      // Anchors and deltas are both stored as (y1, x1, y2, x2) quadruples.
      const float* anchor = anchor_data + 4 * index;
      const float* delta = delta_data + image * per_image * 4 + 4 * index;
      float y1 = anchor[0], x1 = anchor[1], y2 = anchor[2], x2 = anchor[3];
      const float dy = delta[0], dx = delta[1];
      const float dh = std::min(delta[2], delta_clip);
      const float dw = std::min(delta[3], delta_clip);

      const float box_width = x2 - x1;
      const float centre_x = x1 + 0.5f * box_width;
      const float new_centre_x = centre_x + box_width * dx;
      const float new_width = box_width * std::exp(dw);
      x1 = new_centre_x - 0.5f * new_width;
      x2 = new_centre_x + 0.5f * new_width;

      const float box_height = y2 - y1;
      const float centre_y = y1 + 0.5f * box_height;
      const float new_centre_y = centre_y + box_height * dy;
      const float new_height = box_height * std::exp(dh);
      y1 = new_centre_y - 0.5f * new_height;
      y2 = new_centre_y + 0.5f * new_height;

      x1 = std::max(std::min(x1, img_width), 0.0f);
      y1 = std::max(std::min(y1, img_height), 0.0f);
      x2 = std::max(std::min(x2, img_width), 0.0f);
      y2 = std::max(std::min(y2, img_height), 0.0f);

      // A box that clipping has shrunk below the minimum is dropped here, not
      // before: the minimum applies to what is left inside the image.
      if (std::min(x2 - x1, y2 - y1) < min_size_scaled) continue;
      boxes.push_back(x1);
      boxes.push_back(y1);
      boxes.push_back(x2);
      boxes.push_back(y2);
      kept_scores.push_back(image_scores[index]);
    }

    selected.clear();
    const int candidate_count = static_cast<int>(kept_scores.size());
    for (int i = 0; i < candidate_count; ++i) {
      if (static_cast<int64_t>(selected.size()) >= keep) break;
      bool survives = true;
      for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
        if (Iou(boxes.data(), i, *it) > static_cast<float>(nms_threshold)) {
          survives = false;
          break;
        }
      }
      if (survives) selected.push_back(i);
    }

    for (int64_t i = 0; i < keep; ++i) {
      float* roi = roi_out + (image * keep + i) * 4;
      if (i < static_cast<int64_t>(selected.size())) {
        const float* box = boxes.data() + 4 * selected[static_cast<size_t>(i)];
        // Back to TensorFlow's box order on the way out.
        roi[0] = box[1];
        roi[1] = box[0];
        roi[2] = box[3];
        roi[3] = box[2];
        prob_out[image * keep + i] =
            kept_scores[static_cast<size_t>(selected[static_cast<size_t>(i)])];
      } else {
        roi[0] = roi[1] = roi[2] = roi[3] = 0.0f;
        prob_out[image * keep + i] = 0.0f;
      }
    }
  }
}

void Proposals_Compute(void* kernel, TF_OpKernelContext* ctx) {
  ScopedAutoreleasePool pool;
  TF_Status* status = TF_NewStatus();
  auto* op = static_cast<ProposalOp*>(kernel);
  if (op == nullptr) {
    TF_SetStatus(status, TF_INTERNAL,
                 "Metal: GenerateBoundingBoxProposals has no state.");
  } else {
    Proposals_ComputeImpl(op, ctx, status);
  }
  if (TF_GetCode(status) != TF_OK) TF_OpKernelContext_Failure(ctx, status);
  TF_DeleteStatus(status);
}

void Register(const char* op_name,
              void (*compute)(void*, TF_OpKernelContext*),
              const std::string& name) {
  TF_Status* status = TF_NewStatus();
  TF_KernelBuilder* builder =
      TF_NewKernelBuilder(op_name, kMetalDeviceType, &ProposalOp_Create,
                          compute, &ProposalOp_Delete);
  // The three parameters bound the work rather than take part in it.
  TF_KernelBuilder_HostMemory(builder, "nms_threshold");
  TF_KernelBuilder_HostMemory(builder, "pre_nms_topn");
  TF_KernelBuilder_HostMemory(builder, "min_size");
  TF_RegisterKernelBuilder(name.c_str(), builder, status);
  if (TF_GetCode(status) != TF_OK) {
    LOG(ERROR) << "Metal: could not register kernel " << name << ": "
               << TF_Message(status);
  }
  TF_DeleteStatus(status);
}

}  // namespace

void RegisterMetalBoxProposalKernels() {
  Register("GenerateBoundingBoxProposals", &Proposals_Compute,
           "MetalGenerateBoundingBoxProposals");
}

}  // namespace metal
}  // namespace tensorflow
