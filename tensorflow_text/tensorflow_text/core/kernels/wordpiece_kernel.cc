// Copyright 2025 TF.Text Authors.
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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/dataset_stateful_op_allowlist.h"
#include "tensorflow/core/framework/lookup_interface.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow_text/core/kernels/wordpiece_tokenizer.h"

namespace tensorflow {
namespace text {

namespace {
string GetWordSplitChar(OpKernelConstruction* ctx) {
  string suffix_indicator;
  ([=](string* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("suffix_indicator", c));
  })(&suffix_indicator);
  return suffix_indicator;
}

int32 GetMaxCharsPerWord(OpKernelConstruction* ctx) {
  int32 max_chars_per_word;
  ([=](int32* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_bytes_per_word", c));
  })(&max_chars_per_word);
  return max_chars_per_word;
}

int32 GetMaxCharsPerToken(OpKernelConstruction* ctx) {
  int32 max_chars_per_token;
  ([=](int32* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_chars_per_token", c));
  })(&max_chars_per_token);
  return max_chars_per_token;
}

bool GetShouldUseUnknownToken(OpKernelConstruction* ctx) {
  bool use_unknown_token;
  ([=](bool* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_unknown_token", c));
  })(&use_unknown_token);
  return use_unknown_token;
}

string GetUnknownToken(OpKernelConstruction* ctx) {
  string unknown_token;
  ([=](string* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unknown_token", c));
  })(&unknown_token);
  return unknown_token;
}

bool GetSplitUnknownCharacters(OpKernelConstruction* ctx) {
  bool split_unknown_characters;
  ([=](bool* c) -> void {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("split_unknown_characters", c));
  })(&split_unknown_characters);
  return split_unknown_characters;
}

Status GetTableHandle(const string& input_name, OpKernelContext* ctx,
                      string* container, string* table_handle) {
  {
    mutex* mu;
    TF_RETURN_IF_ERROR(ctx->input_ref_mutex(input_name, &mu));
    mutex_lock l(*mu);
    Tensor tensor;
    TF_RETURN_IF_ERROR(ctx->mutable_input(input_name, &tensor, true));
    if (tensor.NumElements() != 2) {
      return errors::InvalidArgument(
          "Lookup table handle must be scalar, but had shape: ",
          tensor.shape().DebugString());
    }
    auto h = tensor.flat<tstring>();
    *container = h(0);
    *table_handle = h(1);
  }
  return absl::OkStatus();
}

// Gets the LookupTable stored in the ctx->resource_manager() with key
// passed by attribute with name input_name, returns null if the table
// doesn't exist.
Status GetLookupTable(const string& input_name, OpKernelContext* ctx,
                      lookup::LookupInterface** table) {
  string container;
  string table_handle;
  DataType handle_dtype;
  TF_RETURN_IF_ERROR(ctx->input_dtype(input_name, &handle_dtype));
  if (handle_dtype == DT_RESOURCE) {
    ResourceHandle handle;
    TF_RETURN_IF_ERROR(HandleFromInput(ctx, input_name, &handle));
    return LookupResource(ctx, handle, table);
  } else {
    TF_RETURN_IF_ERROR(
        GetTableHandle(input_name, ctx, &container, &table_handle));
    return ctx->resource_manager()->Lookup(container, table_handle, table);
  }
}

class LookupTableVocab : public WordpieceVocab {
 public:
  LookupTableVocab(lookup::LookupInterface* table, OpKernelContext* ctx);

  virtual LookupStatus Contains(const absl::string_view key, bool* value) const;

 private:
  // not owned
  mutable lookup::LookupInterface* table_;
  OpKernelContext* ctx_;
  Tensor default_value_;
};

Status ToStatus(const LookupStatus& status) {
  if (status.success) {
    return absl::OkStatus();
  }

  return errors::InvalidArgument(status.error_msg);
}

constexpr int64 kOutOfVocabValue = -1;

LookupTableVocab::LookupTableVocab(lookup::LookupInterface* table,
                                   OpKernelContext* ctx)
    : table_(table), ctx_(ctx), default_value_(DT_INT64, TensorShape({1})) {
  default_value_.flat<int64>()(0) = kOutOfVocabValue;
}

LookupStatus LookupTableVocab::Contains(const absl::string_view key,
                                        bool* value) const {
  if (value == nullptr) {
    return LookupStatus("Bad 'value' param.");
  }
  Tensor keys(DT_STRING, TensorShape({1}));
  keys.flat<tstring>()(0) = tstring(key.data(), key.size());
  Tensor values(DT_INT64, TensorShape({1}));
  auto status = table_->Find(ctx_, keys, &values, default_value_);
  if (!status.ok()) {
// On April 2023, there is not yet an official release of Tensorflow which
// includes `message().` One will need to wait for the release following 2.12.0.
// The code can be updated to just be the else branch after such release exists.
#if TF_GRAPH_DEF_VERSION < 1467
    return LookupStatus(std::string(status.error_message()));
#else
    return LookupStatus(std::string(status.message()));
#endif
  }

  if (static_cast<int64>(values.flat<int64>()(0)) != kOutOfVocabValue) {
    *value = true;
    return LookupStatus::OK();
  }
  *value = false;
  return LookupStatus::OK();
}

}  // namespace

class WordpieceTokenizeWithOffsetsOp : public OpKernel {
 public:
  explicit WordpieceTokenizeWithOffsetsOp(OpKernelConstruction* ctx)
      : OpKernel(ctx),
        suffix_indicator_(GetWordSplitChar(ctx)),
        max_bytes_per_word_(GetMaxCharsPerWord(ctx)),
        max_chars_per_token_(GetMaxCharsPerToken(ctx)),
        use_unknown_token_(GetShouldUseUnknownToken(ctx)),
        unknown_token_(GetUnknownToken(ctx)),
        split_unknown_characters_(GetSplitUnknownCharacters(ctx)) {
    string output_row_partition_type;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_row_partition_type",
                                     &output_row_partition_type));
    if (output_row_partition_type == "row_lengths") {
      row_partition_type_ = ROW_LENGTHS;
    } else if (output_row_partition_type == "row_splits") {
      row_partition_type_ = ROW_SPLITS;
    } else {
      OP_REQUIRES(
          ctx, false,
          errors::Internal("Unexpected value for output_row_partition_type"));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_values;
    OP_REQUIRES_OK(ctx, ctx->input("input_values", &input_values));
    const auto& values_vec = input_values->flat<tstring>();

    lookup::LookupInterface* lookup_table;
    OP_REQUIRES_OK(ctx,
                   GetLookupTable("vocab_lookup_table", ctx, &lookup_table));
    core::ScopedUnref unref_me(lookup_table);
    LookupTableVocab vocab_map(lookup_table, ctx);

    std::vector<string> subwords;
    std::vector<int> begin_offset;
    std::vector<int> end_offset;
    std::vector<int> row_partition;

    if (row_partition_type_ == ROW_SPLITS) {
      row_partition.push_back(0);
    }

    // Iterate through all the values and wordpiece tokenize them.
    for (int i = 0; i < values_vec.size(); ++i) {
      // Tokenize into subwords and record the offset locations.
      int num_wordpieces = 0;
      OP_REQUIRES_OK(
          ctx, ToStatus(WordpieceTokenize(
                   values_vec(i), max_bytes_per_word_, max_chars_per_token_,
                   suffix_indicator_, use_unknown_token_, unknown_token_,
                   split_unknown_characters_, &vocab_map, &subwords,
                   &begin_offset, &end_offset, &num_wordpieces)));

      // Record the row splits.
      switch (row_partition_type_) {
        case ROW_LENGTHS:
          row_partition.push_back(num_wordpieces);
          break;
        case ROW_SPLITS:
          row_partition.push_back(num_wordpieces + row_partition.back());
          break;
      }
    }

    std::vector<int64> output_subwords_shape;
    output_subwords_shape.push_back(subwords.size());

    std::vector<int64> output_row_partition_shape;
    output_row_partition_shape.push_back(row_partition.size());

    Tensor* output_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output_values",
                                             TensorShape(output_subwords_shape),
                                             &output_values));
    auto output_values_vec = output_values->vec<tstring>();

    Tensor* output_row_partition;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("output_row_lengths",
                                        TensorShape(output_row_partition_shape),
                                        &output_row_partition));
    auto output_row_partition_vec = output_row_partition->vec<int64>();

    Tensor* start_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("start_values",
                                             TensorShape(output_subwords_shape),
                                             &start_values));
    auto start_values_vec = start_values->vec<int64>();

    Tensor* limit_values;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("limit_values",
                                             TensorShape(output_subwords_shape),
                                             &limit_values));
    auto limit_values_vec = limit_values->vec<int64>();

    for (int i = 0; i < subwords.size(); ++i) {
      output_values_vec(i) = subwords[i];
    }

    for (int i = 0; i < row_partition.size(); ++i) {
      output_row_partition_vec(i) = row_partition[i];
    }

    for (int i = 0; i < begin_offset.size(); ++i) {
      start_values_vec(i) = begin_offset[i];
    }

    for (int i = 0; i < end_offset.size(); ++i) {
      limit_values_vec(i) = end_offset[i];
    }
  }

 private:
  enum RowPartitionType { ROW_LENGTHS, ROW_SPLITS };

  const string suffix_indicator_;
  const int max_bytes_per_word_;
  const int max_chars_per_token_;
  const bool use_unknown_token_;
  const string unknown_token_;
  const bool split_unknown_characters_;
  RowPartitionType row_partition_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(WordpieceTokenizeWithOffsetsOp);
};

REGISTER_KERNEL_BUILDER(Name("WordpieceTokenizeWithOffsets").Device(DEVICE_CPU),
                        WordpieceTokenizeWithOffsetsOp);
ALLOW_STATEFUL_OP_FOR_DATASET_FUNCTIONS("WordpieceTokenizeWithOffsets");

}  // namespace text
}  // namespace tensorflow
