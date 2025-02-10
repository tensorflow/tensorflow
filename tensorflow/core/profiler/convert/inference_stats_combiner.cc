/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/convert/inference_stats_combiner.h"

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/gtl/map_util.h"

namespace tensorflow::profiler {
namespace {
// Combines two ModelIdDatabases. Returns true if this combination requires
// updating the model_id_index in the SessionRunTimes of dst. This will be
// the case if: (1) Src has a model name that doesn't already exist in dst;
// or (2) Src has a model name that does exist in dst but has a different
// index.
bool CombineModelIdDatabases(const ModelIdDatabase& src, ModelIdDatabase* dst) {
  if (dst->ids_size() == 0) {
    // dst is empty. Simply copy src to dst. This avoids rebuilding
    // dst from src from scratch, which may change the name-to-index mapping.
    *dst = src;
    return false;
  }
  // TODO(tianrun): For now, assume a model is always served with the same
  // parameter on different hosts. In the future, we might consider the case
  // when the same model are served with different batching parameters on
  // different hosts.
  for (const auto& id_and_param : src.id_to_batching_params()) {
    dst->mutable_id_to_batching_params()->insert(id_and_param);
  }
  bool need_update = false;
  for (const auto& [src_id, index] : src.id_to_index()) {
    auto [iter, was_inserted] =
        dst->mutable_id_to_index()->insert({src_id, dst->ids_size()});
    if (was_inserted) {
      *dst->add_ids() = src_id;
      need_update = true;
      continue;
    }
    if (iter->second != index) {
      // src_id is already in dst but has a different index.
      need_update = true;
    }
  }
  return need_update;
}

// Combines two TensorPatternDatabase. Returns true if this combination requires
// updating the tensor_pattern_index. This will be the case if: (1) Src has a
// tensor pattern that doesn't exist in dst; or (2) Src has a tensor pattern
// that does exist in dst but has a different index.
bool CombineTensorPatternDatabase(
    const TensorPatternDatabase& src, TensorPatternDatabase* dst,
    absl::flat_hash_map<absl::string_view, int>* dst_pattern_to_index) {
  if (dst->tensor_pattern().empty()) {
    *dst = src;
    return false;
  }

  bool need_update = false;
  for (int i = 0; i < static_cast<int>(src.tensor_pattern_size()); i++) {
    auto [iter, inserted] = dst_pattern_to_index->insert(
        {src.tensor_pattern(i), dst_pattern_to_index->size()});
    if (inserted) {
      // Src has a tensor pattern that doesn't exist in dst.
      dst->add_tensor_pattern(src.tensor_pattern(i));
      need_update = true;
    } else if (iter->second != i) {
      // Src has a tensor pattern with different index than dst.
      need_update = true;
    }
  }
  return need_update;
}

void UpdateTensorPatternIndex(
    const TensorPatternDatabase& src,
    const absl::flat_hash_map<absl::string_view, int>& dst_pattern_to_index,
    TensorEventDetail* detail) {
  absl::string_view tensor_pattern =
      src.tensor_pattern(detail->tensor_pattern_index());
  if (const int* new_index =
          tsl::gtl::FindOrNull(dst_pattern_to_index, tensor_pattern)) {
    detail->set_tensor_pattern_index(*new_index);
  } else {
    LOG(WARNING) << "Tensor pattern " << tensor_pattern
                 << " is not found in dst->tensor_pattern_db()";
  }
}
}  // namespace

void CombineInferenceStatsResult(int src_host_id, const InferenceStats& src,
                                 InferenceStats* dst) {
  // There should be one key-value pair inside src.inference_stats_per_host(),
  // because the src comes from one XprofResponse (i.e., one host).
  DCHECK_LE(src.inference_stats_per_host_size(), 1);
  bool need_update_model_id =
      CombineModelIdDatabases(src.model_id_db(), dst->mutable_model_id_db());
  absl::flat_hash_map<absl::string_view, int> dst_pattern_to_index;
  for (int i = 0;
       i < static_cast<int>(dst->tensor_pattern_db().tensor_pattern_size());
       i++) {
    dst_pattern_to_index[dst->tensor_pattern_db().tensor_pattern(i)] = i;
  }
  bool need_update_tensor_pattern = CombineTensorPatternDatabase(
      src.tensor_pattern_db(), dst->mutable_tensor_pattern_db(),
      &dst_pattern_to_index);
  for (const auto& [host_id, inf_stats] : src.inference_stats_per_host()) {
    auto [iter, was_inserted] = dst->mutable_inference_stats_per_host()->insert(
        {src_host_id, inf_stats});
    if (!was_inserted) {
      LOG(INFO) << "Duplicate host_id: " << iter->first;
    }
    if (need_update_model_id || need_update_tensor_pattern) {
      // Needs to update the model_id_index in the dst.
      PerHostInferenceStats* dst_inference_stats =
          &(*dst->mutable_inference_stats_per_host())[src_host_id];
      for (RequestDetail& request_detail :
           *dst_inference_stats->mutable_request_details()) {
        if (need_update_model_id && request_detail.model_id_index() != -1) {
          // "model_id_index = -1" means there is no model_id associated with
          // the group id in this event if client doesn't specify "model_id" in
          // TraceMeEncode. so we don't need to update model_id if it doesn't
          // have a model.
          const std::string& model_id =
              src.model_id_db().ids(request_detail.model_id_index());
          auto iter = dst->model_id_db().id_to_index().find(model_id);
          if (iter == dst->model_id_db().id_to_index().end()) {
            LOG(WARNING) << "Model ID " << model_id
                         << " is not found in dst->model_id_db()";
            continue;
          }
          request_detail.set_model_id_index(iter->second);
        }
        if (need_update_tensor_pattern) {
          for (auto& tensor_event_details :
               *request_detail.mutable_tensor_event_details()) {
            UpdateTensorPatternIndex(src.tensor_pattern_db(),
                                     dst_pattern_to_index,
                                     &tensor_event_details);
          }
        }
      }
    }
    if (need_update_tensor_pattern) {
      PerHostInferenceStats* dst_inference_stats =
          &(*dst->mutable_inference_stats_per_host())[src_host_id];
      for (BatchDetail& batch_detail :
           *dst_inference_stats->mutable_batch_details()) {
        UpdateTensorPatternIndex(src.tensor_pattern_db(), dst_pattern_to_index,
                                 batch_detail.mutable_tensor_event_detail());
      }
    }
  }
}
}  // namespace tensorflow::profiler
