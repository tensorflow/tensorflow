/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/lookup_table_init_op.h"
#include "tensorflow/core/kernels/lookup_table_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace {
// lookup::InitializeTableFromTextFile requires a delimiter even though we use
// the entire line for vocabularies.
constexpr char kUnusedLookupDelim = '\t';
}  // namespace

// This Op generates a vocab remapping Tensor from an old and new vocabulary
// file that maps new ID's to old ID's.
class GenerateVocabRemappingOp : public OpKernel {
 public:
  explicit GenerateVocabRemappingOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("new_vocab_offset", &new_vocab_offset_));
    OP_REQUIRES_OK(context, context->GetAttr("num_new_vocab", &num_new_vocab_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("old_vocab_size", &old_vocab_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* new_vocab_file_tensor;
    OP_REQUIRES_OK(context,
                   context->input("new_vocab_file", &new_vocab_file_tensor));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(new_vocab_file_tensor->shape()),
                errors::InvalidArgument(
                    "new_vocab_file should be a single string, but got ",
                    new_vocab_file_tensor->shape().DebugString()));

    // Build a new ID->token lookup table.
    const string& new_vocab_filename =
        new_vocab_file_tensor->scalar<string>()();
    OP_REQUIRES(context, !new_vocab_filename.empty(),
                errors::InvalidArgument("new vocab filename cannot be empty."));
    lookup::HashTable<int64, string>* new_vocab_table =
        new lookup::HashTable<int64, string>(context, this);
    core::ScopedUnref unref_new(new_vocab_table);
    // Note: we pass -1 (unknown) for vocab_size, which is supposed to be the
    // total elements in file.  This is different from num_new_vocab_, which
    // accounts for partitioning.
    OP_REQUIRES_OK(context, lookup::InitializeTableFromTextFile(
                                new_vocab_filename,
                                -1,  // vocab_size
                                kUnusedLookupDelim,
                                -1,  // key_index, use the line number.
                                -2,  // value_index, use the whole line/token.
                                context->env(), new_vocab_table));
    OP_REQUIRES(context,
                new_vocab_offset_ + num_new_vocab_ <= new_vocab_table->size(),
                errors::InvalidArgument("lookup table size must be larger than "
                                        "last new vocab entry's line"));

    const Tensor* old_vocab_file_tensor;
    OP_REQUIRES_OK(context,
                   context->input("old_vocab_file", &old_vocab_file_tensor));
    OP_REQUIRES(context,
                TensorShapeUtils::IsScalar(old_vocab_file_tensor->shape()),
                errors::InvalidArgument(
                    "old_vocab_file should be a single string, but got ",
                    old_vocab_file_tensor->shape().DebugString()));
    // Build a token->old ID lookup table.
    const string& old_vocab_filename =
        old_vocab_file_tensor->scalar<string>()();
    OP_REQUIRES(context, !old_vocab_filename.empty(),
                errors::InvalidArgument("new vocab filename cannot be empty."));
    lookup::HashTable<string, int64>* old_vocab_table =
        new lookup::HashTable<string, int64>(context, this);
    core::ScopedUnref unref_old(old_vocab_table);
    // Note: If old_vocab_size_ is -1 (unknown), we retrieve all elements in
    // file (see TextFileLineIterator).
    OP_REQUIRES_OK(context,
                   lookup::InitializeTableFromTextFile(
                       old_vocab_filename, old_vocab_size_, kUnusedLookupDelim,
                       -2,  // key_index, use the whole line/token.
                       -1,  // value_index, use the line number.
                       context->env(), old_vocab_table));

    // Fill out new_ids = [new_vocab_offset, new_vocab_offset + 1, ...,
    //                     new_vocab_offset + num_new_vocab_]
    // The double look-up requires a few temporary Tensors.
    Tensor new_ids;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_INT64, TensorShape({num_new_vocab_}),
                                        &new_ids));
    auto new_ids_vec = new_ids.vec<int64>();
    // Note that we should always be able to find tokens for all new ID's, given
    // that the lookup table is constructed with the vocabulary file itself
    // (see the check on offset and table size post-initialization).
    Tensor default_token;
    OP_REQUIRES_OK(
        context, context->allocate_temp(
                     DT_STRING, TensorShape({num_new_vocab_}), &default_token));
    auto default_token_vec = default_token.vec<string>();
    default_token_vec.setConstant("" /* NOT_FOUND_TOKEN */);

    Tensor default_id;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_INT64, TensorShape({num_new_vocab_}),
                                        &default_id));
    auto default_id_vec = default_id.vec<int64>();
    default_id_vec.setConstant(-1 /* NOT_FOUND_ID */);

    for (int i = 0; i < num_new_vocab_; ++i) {
      new_ids_vec(i) = static_cast<int64>(i + new_vocab_offset_);
    }
    Tensor tokens;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(
                       DT_STRING, TensorShape({num_new_vocab_}), &tokens));
    Tensor* remapping;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       "remapping", TensorShape({num_new_vocab_}), &remapping));
    // In the corner case where num_new_vocab_ is 0 (we are dealing with an
    // OOV-only partition), we should not do this lookup.
    if (num_new_vocab_ != 0) {
      OP_REQUIRES_OK(context, new_vocab_table->Find(context, new_ids, &tokens,
                                                    default_token));
      OP_REQUIRES_OK(context, old_vocab_table->Find(context, tokens, remapping,
                                                    default_id));
    }
    // Iterate through remapping to calculate num_present.
    const auto remapping_vec = remapping->vec<int64>();
    int num_present = 0;
    for (int i = 0; i < num_new_vocab_; ++i) {
      if (remapping_vec(i) != -1 /* NOT_FOUND_ID */) {
        ++num_present;
      }
    }
    Tensor* num_present_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output("num_present", TensorShape({}),
                                            &num_present_t));
    num_present_t->scalar<int>()() = num_present;
  }

 private:
  int new_vocab_offset_;
  int num_new_vocab_;
  int old_vocab_size_;
};

REGISTER_KERNEL_BUILDER(Name("GenerateVocabRemapping").Device(DEVICE_CPU),
                        GenerateVocabRemappingOp);

}  // namespace tensorflow
