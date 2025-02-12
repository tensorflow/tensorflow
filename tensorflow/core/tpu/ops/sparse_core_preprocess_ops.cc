/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

REGISTER_OP("ConvertToCooTensor")
    .Input("indices_or_row_splits: int32")
    .Input("values: int32")
    .Input("weights: float32")
    .Output("row_ids: int32")
    .Output("col_ids: int32")
    .Output("gains: float32")
    .Attr("sample_count: int >= 1")
    .Attr("combiner: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle rank;
      for (int i = 0; i < c->num_outputs(); ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->output(i), 1, &rank));
        c->set_output(i, c->UnknownShapeOfRank(1));
      }
      return absl::OkStatus();
    });

REGISTER_OP("GetMinibatchesInCsrWithPhysicalReplica")
    .Input("program_key: string")
    .Input("row_ids: int32")
    .Input("col_ids: int32")
    .Input("gains: float32")
    .Input("splits: int64")
    .Input("id_counts: int32")
    .Output("row_pointers: int32")
    .Output("sorted_sample_ids: int32")
    .Output("sorted_token_ids: int32")
    .Output("sorted_gains: float32")
    .Output("row_pointers_unpadded_size: int32")
    .Output("ids_unpadded_size: int32")
    .Output("num_minibatches_per_physical_sparse_core: int32")
    .Attr("sample_count : int >= 1")
    .Attr("num_replica: int >= 1")
    .Attr("max_minibatches_per_sc: int >= 1")
    .Attr("max_ids_per_chip_per_sample: int >= 1")
    .Attr("table_vocab_size: int >= 1")
    .Attr("feature_width: int >= 1")
    .Attr("num_sc_per_chip: int >= 1")
    .Attr("table_name: string")
    .Attr("mini_batch_in_csr: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle rank;
      for (int i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 1, &rank));
      }
      int32 max_minibatches_per_sc;
      TF_RETURN_IF_ERROR(
          c->GetAttr("max_minibatches_per_sc", &max_minibatches_per_sc));
      int32 num_replica;
      TF_RETURN_IF_ERROR(c->GetAttr("num_replica", &num_replica));
      int32 sample_count;
      TF_RETURN_IF_ERROR(c->GetAttr("sample_count", &sample_count));
      int32 max_ids_per_chip_per_sample;
      TF_RETURN_IF_ERROR(c->GetAttr("max_ids_per_chip_per_sample",
                                    &max_ids_per_chip_per_sample));

      // We can't get this number programmatically since the shape inference
      // will be run as part of the graph generation which might not have the
      // tpu system available.
      const int xla_pad_size = 8;
      int32 num_sc_per_chip;
      TF_RETURN_IF_ERROR(c->GetAttr("num_sc_per_chip", &num_sc_per_chip));

      const int num_physical_replica = num_replica * num_sc_per_chip;
      const int max_total_minibatches =
          num_sc_per_chip * max_minibatches_per_sc;
      const int max_ids_per_chip = max_ids_per_chip_per_sample * sample_count;

      const int padded_row_pointers_size_per_sc =
          xla::RoundUpTo(num_physical_replica, xla_pad_size);

      c->set_output(0, c->MakeShape({max_total_minibatches *
                                     padded_row_pointers_size_per_sc}));
      for (int i = 1; i < 4; ++i) {
        c->set_output(i, c->MakeShape({max_ids_per_chip}));
      }
      c->set_output(4, c->Scalar());
      c->set_output(5, c->Scalar());
      c->set_output(6, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("GetMinibatchSplitsWithPhysicalReplica")
    .Input("program_key: string")
    .Input("row_ids: int32")
    .Input("col_ids: int32")
    .Input("gains: float32")
    .Output("sorted_row_ids: int32")
    .Output("sorted_col_ids: int32")
    .Output("sorted_gains: float32")
    .Output("splits: int64")
    .Output("id_counts: int32")
    .Output("max_ids: int32")
    .Output("max_uniques: int32")
    .Attr("sample_count : int >= 1")
    .Attr("num_replica: int >= 1")
    .Attr("table_vocab_size: int >= 1")
    .Attr("feature_width: int >= 1")
    .Attr("num_sc_per_chip: int >= 1")
    .Attr("table_name: string")
    .Attr("mini_batch_splits: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->UnknownShapeOfRank(1));
      c->set_output(1, c->UnknownShapeOfRank(1));
      c->set_output(2, c->UnknownShapeOfRank(1));
      c->set_output(3, c->Scalar());
      // Depends on max division level, which is currently passed by flag.
      c->set_output(4, c->UnknownShapeOfRank(1));
      c->set_output(5, c->Scalar());
      c->set_output(6, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("StoreMinibatchStatisticsInFdo")
    .Input("program_key: string")
    .Input("max_ids: int32")
    .Input("max_uniques: int32")
    .Attr("sample_count : int >= 1")
    .Attr("num_replica: int >= 1")
    .Attr("feature_width: int >= 1")
    .Attr("num_sc_per_chip: int >= 1")
    .Attr("table_name: string")
    .Attr("mini_batch_splits: string")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      return absl::OkStatus();
    });

REGISTER_OP("ConvertToListOfSparseCoreCooTensors")
    .Input("indices_or_row_splits: int32")
    .Input("values: int32")
    .Input("weights: float32")
    .Output("row_ids_list: num_sc_per_chip * int32")
    .Output("col_ids_list: num_sc_per_chip * int32")
    .Output("gains_list: num_sc_per_chip * float32")
    .Attr("sample_count: int >= 1")
    .Attr("num_sc_per_chip: int >= 1")
    .Attr("row_offset: int >= 0")
    .Attr("col_offset: int >= 0")
    .Attr("col_shift: int >= 0")
    .Attr("num_sc_shards: int >= 1")
    .Attr("stacked_table_sample_count: int >= 1")
    .Attr("combiner: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32_t num_sc_per_chip;
      TF_RETURN_IF_ERROR(c->GetAttr("num_sc_per_chip", &num_sc_per_chip));
      std::vector<shape_inference::ShapeHandle> output_id_shape(
          num_sc_per_chip, c->UnknownShapeOfRank(1));
      TF_RETURN_IF_ERROR(c->set_output("row_ids_list", output_id_shape));
      TF_RETURN_IF_ERROR(c->set_output("col_ids_list", output_id_shape));
      TF_RETURN_IF_ERROR(c->set_output("gains_list", output_id_shape));
      return absl::OkStatus();
    });

REGISTER_OP("SortListOfSparseCoreCooTensors")
    .Input("row_ids_list: N * int32")
    .Input("col_ids_list: N * int32")
    .Input("gains_list:  N * float32")
    .Output("sorted_row_ids: int32")
    .Output("sorted_col_ids: int32")
    .Output("sorted_gains: float32")
    .Output("id_counts: int32")
    .Attr("sample_count_list : list(int)")
    .Attr("col_offset_list : list(int)")
    .Attr("num_replica: int >= 1")
    .Attr("table_vocab_size: int >= 1")
    .Attr("feature_width: int >= 1")
    .Attr("num_sc_per_chip: int >= 1")
    .Attr("max_ids_per_sparse_core: int >= 1")
    .Attr("max_unique_ids_per_sparse_core: int >= 1")
    .Attr("table_name: string")
    .Attr("N: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<int32_t> sample_count_list;
      TF_RETURN_IF_ERROR(c->GetAttr("sample_count_list", &sample_count_list));
      std::vector<int32_t> col_offset_list;
      TF_RETURN_IF_ERROR(c->GetAttr("col_offset_list", &col_offset_list));
      int32_t num_features;
      TF_RETURN_IF_ERROR(c->GetAttr("N", &num_features));

      if (sample_count_list.size() != num_features) {
        return absl::InvalidArgumentError(
            absl::StrCat("sample_count_list must have the same size as number "
                         "of feature inputs, but got ",
                         sample_count_list.size(), " vs ", num_features));
      }

      if (col_offset_list.size() != num_features) {
        return absl::InvalidArgumentError(
            absl::StrCat("col_offset_list must have the same size as number of "
                         "feature inputs, but got ",
                         col_offset_list.size(), " vs ", num_features));
      }

      c->set_output(0, c->UnknownShapeOfRank(1));
      c->set_output(1, c->UnknownShapeOfRank(1));
      c->set_output(2, c->UnknownShapeOfRank(1));
      c->set_output(3, c->UnknownShapeOfRank(1));
      return absl::OkStatus();
    });

REGISTER_OP("ConvertToSparseCoreCsrWrappedCooTensor")
    .Input("sorted_row_ids_list: num_sc_per_chip * int32")
    .Input("sorted_col_ids_list: num_sc_per_chip * int32")
    .Input("sorted_gains_list: num_sc_per_chip * float32")
    .Input("id_counts_list: num_sc_per_chip * int32")
    .Input("splits: int64")
    .Output("row_pointers: int32")
    .Output("sorted_sample_ids: int32")
    .Output("sorted_token_ids: int32")
    .Output("sorted_gains: float32")
    .Output("row_pointers_unpadded_size: int32")
    .Output("ids_unpadded_size: int32")
    .Output("num_minibatches_per_sc: int32")
    .Attr("sample_count_per_sc : int >= 1")
    .Attr("num_replica: int >= 1")
    .Attr("max_minibatches_per_sc: int >= 1")
    .Attr("max_ids_per_chip_per_sample: int >= 1")
    .Attr("table_vocab_size: int >= 1")
    .Attr("feature_width: int >= 1")
    .Attr("num_sc_per_chip: int >= 1")
    .Attr("table_name: string")
    .Attr("allow_id_dropping: bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int32 max_minibatches_per_sc;
      TF_RETURN_IF_ERROR(
          c->GetAttr("max_minibatches_per_sc", &max_minibatches_per_sc));
      int32 num_replica;
      TF_RETURN_IF_ERROR(c->GetAttr("num_replica", &num_replica));
      int32 sample_count_per_sc;
      TF_RETURN_IF_ERROR(
          c->GetAttr("sample_count_per_sc", &sample_count_per_sc));
      int32 max_ids_per_chip_per_sample;
      TF_RETURN_IF_ERROR(c->GetAttr("max_ids_per_chip_per_sample",
                                    &max_ids_per_chip_per_sample));
      // We can't get this number programmatically since the shape inference
      // will be run as part of the graph generation which might not have the
      // tpu system available.
      const int xla_pad_size = 8;
      int32 num_sc_per_chip;
      TF_RETURN_IF_ERROR(c->GetAttr("num_sc_per_chip", &num_sc_per_chip));

      const int num_physical_replica = num_replica * num_sc_per_chip;
      const int max_total_minibatches =
          num_sc_per_chip * max_minibatches_per_sc;
      const int max_ids_per_chip =
          max_ids_per_chip_per_sample * sample_count_per_sc * num_sc_per_chip;

      const int padded_row_pointers_size_per_sc =
          xla::RoundUpTo(num_physical_replica, xla_pad_size);

      c->set_output(0, c->MakeShape({max_total_minibatches *
                                     padded_row_pointers_size_per_sc}));
      for (int i = 1; i < 4; ++i) {
        c->set_output(i, c->MakeShape({max_ids_per_chip}));
      }
      c->set_output(4, c->Scalar());
      c->set_output(5, c->Scalar());
      c->set_output(6, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("GetStatsFromListOfSparseCoreCooTensors")
    .Input("row_ids_list: N * int32")
    .Input("col_ids_list: N * int32")
    .Input("gains_list:  N * float32")
    .Output("max_ids_per_sparse_core: int32")
    .Output("max_unique_ids_per_sparse_core: int32")
    .Attr("sample_count_list : list(int)")
    .Attr("col_offset_list : list(int)")
    .Attr("num_replica: int >= 1")
    .Attr("table_vocab_size: int >= 1")
    .Attr("feature_width: int >= 1")
    .Attr("num_sc_per_chip: int >= 1")
    .Attr("table_name: string")
    .Attr("N: int >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return absl::OkStatus();
    });

}  // namespace tensorflow
