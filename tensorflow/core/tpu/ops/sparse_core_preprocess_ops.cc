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
#include "xla/util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tsl/platform/errors.h"

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
}  // namespace tensorflow
