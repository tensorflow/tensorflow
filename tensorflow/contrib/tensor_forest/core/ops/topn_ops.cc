// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
// The three Ops used to implement a TopN structure:  Insert, Remove, and
// RefreshShortlist.

#include <algorithm>
#include <numeric>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("TopNInsert")
    .Input("ids: int64")
    .Input("scores: float32")
    .Input("new_ids: int64")
    .Input("new_scores: float32")
    .Output("shortlist_ids: int64")
    .Output("update_ids: int64")
    .Output("update_scores: float32")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    })
    .Doc(R"doc(
  Outputs update Tensors for adding new_ids and new_scores to the shortlist.

  ids:= A 1-D int64 tensor containing the ids on the shortlist (except for
    ids[0], which is the current size of the shortlist.
  scores:= A 1-D float32 tensor containing the scores on the shortlist.
  new_ids:= A 1-D int64 tensor containing the new ids to add to the shortlist.
  shortlist_ids:= A 1-D int64 tensor containing the ids of the shortlist entries
    to update.  Intended to be used with
    tf.scatter_update(shortlist_scores, shortlist_ids, new_scores).
  update_ids:= A 1-D int64 tensor containing ...
  update_scores:= A 1-D float32 tensor containing ...
)doc");

class TopNInsert : public OpKernel {
 public:
  explicit TopNInsert(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& ids = context->input(0);
    const Tensor& scores = context->input(1);
    const Tensor& new_ids = context->input(2);
    const Tensor& new_scores = context->input(3);

    OP_REQUIRES(context, ids.shape().dims() == 1,
                errors::InvalidArgument("ids should be one-dimensional"));
    OP_REQUIRES(context, scores.shape().dims() == 1,
                errors::InvalidArgument("scores should be one-dimensional"));
    OP_REQUIRES(context, new_ids.shape().dims() == 1,
                errors::InvalidArgument("new_ids should be one-dimensional"));
    OP_REQUIRES(
        context, new_scores.shape().dims() == 1,
        errors::InvalidArgument("new_scores should be one-dimensional"));

    OP_REQUIRES(
        context, ids.shape().dim_size(0) == scores.shape().dim_size(0),
        errors::InvalidArgument("ids and scores should be the same length"));
    OP_REQUIRES(context,
                new_ids.shape().dim_size(0) == new_scores.shape().dim_size(0),
                errors::InvalidArgument(
                    "new_ids and new_scores should be the same length"));

    const auto flat_ids = ids.unaligned_flat<int64>();
    const auto flat_scores = scores.unaligned_flat<float>();
    const auto flat_new_ids = new_ids.unaligned_flat<int64>();
    const auto flat_new_scores = new_scores.unaligned_flat<float>();

    const int num_updates = new_ids.shape().dim_size(0);
    const int shortlist_max_size = ids.shape().dim_size(0) - 1;
    int shortlist_size = std::max(0, static_cast<int>(flat_ids(0)));
    int overflow = shortlist_size + num_updates - shortlist_max_size;

    std::vector<std::tuple<int64, int64, float>> updates;
    float score_cutoff = flat_scores(0);

    if (overflow > 0) {
      // Sort the *highest* overflow updates
      std::vector<int> update_indices(num_updates);
      for (int i = 0; i < num_updates; i++) {
        update_indices[i] = i;
      }
      auto cmp = [&flat_new_scores](int a, int b) {
        return flat_new_scores(a) > flat_new_scores(b);
      };
      std::sort(update_indices.begin(), update_indices.end(), cmp);

      // Sort the *lowest* overflow shortlist entries
      std::vector<int> shortlist_indices(shortlist_max_size + 1);
      std::iota(shortlist_indices.begin() + 1, shortlist_indices.end(), 1);
      auto cmp2 = [&flat_scores](int a, int b) {
        return flat_scores(a) < flat_scores(b);
      };
      std::sort(shortlist_indices.begin() + 1, shortlist_indices.end(), cmp2);

      int i = 0;  // Points into update_indices
      int j = 1;  // Points into shortlist_indices
      while (i < num_updates && j <= shortlist_max_size) {
        VLOG(2) << "i = " << i;
        VLOG(2) << "j = " << j;
        VLOG(2) << "update_indices[i] = " << update_indices[i];
        VLOG(2) << "shortlist_indices[j] = " << shortlist_indices[j];
        VLOG(2) << "flat_new_scores(update_indices[i]) = "
                << flat_new_scores(update_indices[i]);
        VLOG(2) << "flat_scores(shortlist_indices[j])) = "
                << flat_scores(shortlist_indices[j]);
        if (flat_new_scores(update_indices[i]) >
            flat_scores(shortlist_indices[j])) {
          // Whenever we erase something from the shortlist, we need to
          // update score_cutoff.
          score_cutoff =
              std::max(score_cutoff, flat_scores(shortlist_indices[j]));
          updates.push_back(std::make_tuple(
              shortlist_indices[j], flat_new_ids(update_indices[i]),
              flat_new_scores(update_indices[i])));
          if (flat_ids(shortlist_indices[j]) == -1) {
            shortlist_size++;
          }
          j++;
        } else {
          // Whenever we fail to insert something into the shortlist, we need to
          // update score_cutoff.
          score_cutoff =
              std::max(score_cutoff, flat_new_scores(update_indices[i]));
        }
        i++;
      }
    } else {
      // Everything fits, no need to sort.
      int j = 1;
      for (int i = 0; i < num_updates; i++) {
        if (flat_new_scores(i) < score_cutoff) {
          continue;
        }
        while (j <= shortlist_max_size && flat_ids(j) != -1) {
          j++;
        }
        if (j > shortlist_max_size) {
          LOG(FATAL) << "Bug";
        }
        updates.push_back(
            std::make_tuple(j, flat_new_ids(i), flat_new_scores(i)));
        j++;
        shortlist_size++;
      }
    }

    updates.push_back(std::make_tuple(0, shortlist_size, score_cutoff));

    Tensor* output_shortlist_ids = nullptr;
    TensorShape shortlist_ids_shape;
    shortlist_ids_shape.AddDim(updates.size());
    OP_REQUIRES_OK(context, context->allocate_output(0, shortlist_ids_shape,
                                                     &output_shortlist_ids));
    auto shortlist_ids_flat = output_shortlist_ids->tensor<int64, 1>();

    Tensor* output_ids = nullptr;
    TensorShape ids_shape;
    ids_shape.AddDim(updates.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, ids_shape, &output_ids));
    auto output_ids_flat = output_ids->tensor<int64, 1>();

    Tensor* output_scores = nullptr;
    TensorShape scores_shape;
    scores_shape.AddDim(updates.size());
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, scores_shape, &output_scores));
    auto output_scores_flat = output_scores->tensor<float, 1>();

    int i = 0;
    for (const auto& update : updates) {
      shortlist_ids_flat(i) = std::get<0>(update);
      output_ids_flat(i) = std::get<1>(update);
      output_scores_flat(i) = std::get<2>(update);
      i++;
    }
  }
};

REGISTER_OP("TopNRemove")
    .Input("ids: int64")
    .Input("remove_ids: int64")
    .Output("shortlist_ids: int64")
    .Output("new_length: int64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      return Status::OK();
    })
    .Doc(R"doc(
  Remove ids from a shortlist.

  ids:= A 1-D int64 tensor containing the ids on the shortlist (except for
    ids[0], which is the current size of the shortlist.
  remove_ids:= A 1-D int64 tensor containing the ids to remove.
  shortlist_ids:= A 1-D int64 tensor containing the shortlist entries that
    need to be removed.
  new_length:= A length 1 1-D int64 tensor containing the new length of the
    shortlist.
)doc");

class TopNRemove : public OpKernel {
 public:
  explicit TopNRemove(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& ids = context->input(0);
    const Tensor& remove_ids = context->input(1);

    OP_REQUIRES(context, ids.shape().dims() == 1,
                errors::InvalidArgument("ids should be one-dimensional"));
    OP_REQUIRES(
        context, remove_ids.shape().dims() == 1,
        errors::InvalidArgument("remove_ids should be one-dimensional"));

    const auto flat_ids = ids.unaligned_flat<int64>();
    const auto flat_remove_ids = remove_ids.unaligned_flat<int64>();

    const int num_to_remove = remove_ids.shape().dim_size(0);
    const int shortlist_max_size = ids.shape().dim_size(0);

    // First, turn remove_ids into a set for easy membership checking.
    std::unordered_set<int> ids_to_remove(
        flat_remove_ids.data(), flat_remove_ids.data() + num_to_remove);

    std::vector<int64> updates;
    int shortlist_size = std::max(0, static_cast<int>(flat_ids(0)));
    for (int j = 1; j < shortlist_max_size; j++) {
      if (ids_to_remove.find(flat_ids(j)) != ids_to_remove.end()) {
        shortlist_size--;
        updates.push_back(j);
      }
    }

    Tensor* output_shortlist_ids = nullptr;
    TensorShape shortlist_ids_shape;
    shortlist_ids_shape.AddDim(updates.size());
    OP_REQUIRES_OK(context, context->allocate_output(0, shortlist_ids_shape,
                                                     &output_shortlist_ids));
    auto shortlist_ids_flat = output_shortlist_ids->tensor<int64, 1>();

    std::copy(updates.begin(), updates.end(), shortlist_ids_flat.data());

    Tensor* new_length = nullptr;
    TensorShape new_length_shape;
    new_length_shape.AddDim(1);
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, new_length_shape, &new_length));
    new_length->tensor<int64, 1>()(0) = shortlist_size;
  }
};

REGISTER_KERNEL_BUILDER(Name("TopNInsert").Device(DEVICE_CPU), TopNInsert);
REGISTER_KERNEL_BUILDER(Name("TopNRemove").Device(DEVICE_CPU), TopNRemove);
}  // namespace tensorflow
