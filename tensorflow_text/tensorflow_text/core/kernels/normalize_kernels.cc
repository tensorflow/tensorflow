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

#include <cstdint>
#include <locale>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "icu4c/source/common/unicode/edits.h"
#include "icu4c/source/common/unicode/errorcode.h"
#include "icu4c/source/common/unicode/normalizer2.h"
#include "icu4c/source/common/unicode/utypes.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow_text/core/kernels/edit_changes.pb.h"

namespace tensorflow {
namespace text {

class CaseFoldUTF8Op : public tensorflow::OpKernel {
 public:
  explicit CaseFoldUTF8Op(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_vec = input_tensor->flat<tstring>();

    // TODO(gregbillock): support forwarding
    tensorflow::Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                     &output_tensor));
    auto output_vec = output_tensor->flat<tstring>();

    icu::ErrorCode icu_error;
    const icu::Normalizer2* nfkc_cf =
        icu::Normalizer2::getNFKCCasefoldInstance(icu_error);
    OP_REQUIRES(context, icu_error.isSuccess(),
                errors::Internal(absl::StrCat(
                    icu_error.errorName(),
                    ": Could not retrieve ICU NFKC_CaseFold normalizer")));

    for (int64 i = 0; i < input_vec.size(); ++i) {
      string output_text;
      icu::StringByteSink<string> byte_sink(&output_text);
      const auto& input = input_vec(i);
      nfkc_cf->normalizeUTF8(0, icu::StringPiece(input.data(), input.size()),
                             byte_sink, nullptr, icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal("Could not normalize input string: " +
                                   input_vec(i)));
      output_vec(i) = output_text;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CaseFoldUTF8").Device(tensorflow::DEVICE_CPU),
                        CaseFoldUTF8Op);

namespace {

string GetNormalizationForm(OpKernelConstruction* context) {
  string normalization_form;
  ([=](string* c) -> void {
    OP_REQUIRES_OK(context, context->GetAttr("normalization_form", c));
  })(&normalization_form);
  return absl::AsciiStrToUpper(normalization_form);
}

}  // namespace

class NormalizeUTF8Op : public tensorflow::OpKernel {
 public:
  explicit NormalizeUTF8Op(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context),
        normalization_form_(GetNormalizationForm(context)) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_vec = input_tensor->flat<tstring>();

    tensorflow::Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                     &output_tensor));
    auto output_vec = output_tensor->flat<tstring>();

    icu::ErrorCode icu_error;
    const icu::Normalizer2* normalizer = nullptr;
    if (normalization_form_ == "NFKC") {
      normalizer = icu::Normalizer2::getNFKCInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(absl::StrCat(
                      icu_error.errorName(),
                      ": Could not retrieve ICU NFKC normalizer")));
    } else if (normalization_form_ == "NFC") {
      normalizer = icu::Normalizer2::getNFCInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(
                      absl::StrCat(icu_error.errorName(),
                                   ": Could not retrieve ICU NFC normalizer")));
    } else if (normalization_form_ == "NFD") {
      normalizer = icu::Normalizer2::getNFDInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(
                      absl::StrCat(icu_error.errorName(),
                                   ": Could not retrieve ICU NFD normalizer")));
    } else if (normalization_form_ == "NFKD") {
      normalizer = icu::Normalizer2::getNFKDInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(absl::StrCat(
                      icu_error.errorName(),
                      ": Could not retrieve ICU NFKd normalizer")));
    } else {
      OP_REQUIRES(
          context, false,
          errors::InvalidArgument(absl::StrCat(
              "Unknown normalization form requrested: ", normalization_form_)));
    }

    for (int64 i = 0; i < input_vec.size(); ++i) {
      string output_text;
      icu::StringByteSink<string> byte_sink(&output_text);
      const auto& input = input_vec(i);
      normalizer->normalizeUTF8(0, icu::StringPiece(input.data(), input.size()),
                                byte_sink, nullptr, icu_error);
      OP_REQUIRES(
          context, icu_error.isSuccess(),
          errors::Internal(absl::StrCat(icu_error.errorName(),
                                        ": Could not normalize input string: ",
                                        absl::string_view(input_vec(i)))));
      output_vec(i) = output_text;
    }
  }

 private:
  string normalization_form_;
};

REGISTER_KERNEL_BUILDER(Name("NormalizeUTF8").Device(tensorflow::DEVICE_CPU),
                        NormalizeUTF8Op);

namespace {

// OffsetMapVariant is a tf.Variant object that stores a single icu::Edits
// object and providing encode/decode methods.
// The encode method is called to serialize the stored icu::Edits object when
// the variant is assigned to graph output. The decode method is called to
// reconstruct the icu::Edits object from the serialized `changes` string when
// the variant is at the graph input.
struct OffsetMapVariant {
  string changes;
  icu::Edits edits_;

  std::string TypeName() const { return "(anonymous)::OffsetMapVariant"; }
  void Encode(tensorflow::VariantTensorData* data) const;
  bool Decode(const tensorflow::VariantTensorData& data);
};

void OffsetMapVariant::Encode(tensorflow::VariantTensorData* data) const {
  EditChanges changes;
  icu::Edits::Iterator it = edits_.getFineIterator();
  icu::ErrorCode icu_error;
  while (it.next(icu_error)) {
    auto* change = changes.add_change();
    change->set_old_length(it.oldLength());
    change->set_new_length(it.newLength());
  }
  string changes_str = changes.SerializeAsString();
  data->set_metadata(changes_str);
}

bool OffsetMapVariant::Decode(const tensorflow::VariantTensorData& data) {
  string serialized;
  data.get_metadata(&serialized);
  EditChanges changes;
  changes.ParseFromString(serialized);
  icu::Edits edit;
  icu::ErrorCode icu_error;
  for (int64 j = 0; j < changes.change_size(); ++j) {
    auto* change = changes.mutable_change(j);
    int old_length = change->old_length();
    int new_length = change->new_length();
    if (old_length == new_length) {
      edit.addUnchanged(static_cast<int32_t>(old_length));
    } else {
      edit.addReplace(static_cast<int32_t>(old_length),
                      static_cast<int32_t>(new_length));
    }
  }
  edits_ = edit;
  return true;
}
}  // namespace

class NormalizeUTF8WithOffsetsMapOp : public tensorflow::OpKernel {
 public:
  explicit NormalizeUTF8WithOffsetsMapOp(
      tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context),
        normalization_form_(GetNormalizationForm(context)) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_vec = input_tensor->flat<tstring>();

    tensorflow::Tensor* output_tensor;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor->shape(),
                                                     &output_tensor));
    tensorflow::Tensor* output_offsets_map_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, input_tensor->shape(),
                                            &output_offsets_map_tensor));

    auto output_vec = output_tensor->flat<tstring>();
    auto output_offsets_map_vec = output_offsets_map_tensor->flat<Variant>();

    icu::ErrorCode icu_error;
    const icu::Normalizer2* normalizer = nullptr;
    if (normalization_form_ == "NFKC") {
      normalizer = icu::Normalizer2::getNFKCInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(absl::StrCat(
                      icu_error.errorName(),
                      ": Could not retrieve ICU NFKC normalizer")));
    } else if (normalization_form_ == "NFC") {
      normalizer = icu::Normalizer2::getNFCInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(
                      absl::StrCat(icu_error.errorName(),
                                   ": Could not retrieve ICU NFC normalizer")));
    } else if (normalization_form_ == "NFD") {
      normalizer = icu::Normalizer2::getNFDInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(
                      absl::StrCat(icu_error.errorName(),
                                   ": Could not retrieve ICU NFD normalizer")));
    } else if (normalization_form_ == "NFKD") {
      normalizer = icu::Normalizer2::getNFKDInstance(icu_error);
      OP_REQUIRES(context, icu_error.isSuccess(),
                  errors::Internal(absl::StrCat(
                      icu_error.errorName(),
                      ": Could not retrieve ICU NFKD normalizer")));
    } else {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument(absl::StrCat(
                      "Offset not supported for this normalization form: ",
                      normalization_form_)));
    }

    for (int64 i = 0; i < input_vec.size(); ++i) {
      OffsetMapVariant variant;
      string output_text;
      icu::Edits edits;
      icu::StringByteSink<string> byte_sink(&output_text);
      const auto& input = input_vec(i);
      normalizer->normalizeUTF8(0, icu::StringPiece(input.data(), input.size()),
                                byte_sink, &edits, icu_error);
      OP_REQUIRES(
          context, icu_error.isSuccess(),
          errors::Internal(absl::StrCat(icu_error.errorName(),
                                        ": Could not normalize input string: ",
                                        absl::string_view(input_vec(i)))));

      output_vec(i) = output_text;
      variant.edits_ = std::move(edits);
      output_offsets_map_vec(i) = variant;
    }
  }

 private:
  string normalization_form_;
};

REGISTER_KERNEL_BUILDER(
    Name("NormalizeUTF8WithOffsetsMap").Device(tensorflow::DEVICE_CPU),
    NormalizeUTF8WithOffsetsMapOp);

template <typename SPLITS_TYPE>
class FindSourceOffsetsOp : public tensorflow::OpKernel {
 public:
  explicit FindSourceOffsetsOp(tensorflow::OpKernelConstruction* context)
      : tensorflow::OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext* context) override {
    const tensorflow::Tensor& edits_values = context->input(0);
    const tensorflow::Tensor& input_offsets_values = context->input(1);
    const tensorflow::Tensor& input_offsets_splits = context->input(2);

    const auto& input_offsets_values_vec = input_offsets_values.flat<int64>();
    const auto& input_offsets_splits_vec =
        input_offsets_splits.flat<SPLITS_TYPE>();
    const auto& edits_vec = edits_values.flat<Variant>();

    icu::ErrorCode icu_error;
    int64 cur_split_index_begin = 0;
    int64 cur_split_index_end = 0;
    std::vector<int64> output_offsets_values(input_offsets_values_vec.size());
    int64 idx_edits = 0;
    int64 idx_output = 0;
    for (int64 i = 0; i < input_offsets_splits_vec.size() - 1; ++i) {
      cur_split_index_begin = input_offsets_splits_vec(i);
      cur_split_index_end = input_offsets_splits_vec(i + 1);
      if (cur_split_index_begin == cur_split_index_end) {
        continue;
      }
      OP_REQUIRES(context, idx_edits < edits_vec.size(),
                  tensorflow::errors::InvalidArgument(
                      "Input offset tensor dimension did not match the offset "
                      "map dimension."));
      auto iter = edits_vec(idx_edits++)
                      .get<OffsetMapVariant>()
                      ->edits_.getFineChangesIterator();
      for (int64 j = cur_split_index_begin; j < cur_split_index_end; ++j) {
        output_offsets_values[idx_output++] =
            iter.sourceIndexFromDestinationIndex(input_offsets_values_vec(j),
                                                 icu_error);
      }
    }
    OP_REQUIRES(context, idx_edits == edits_vec.size(),
                tensorflow::errors::InvalidArgument(
                    "Input offset tensor dimension did not match the offset "
                    "map dimension."));

    int64 output_offsets_values_size = output_offsets_values.size();
    Tensor* output_offsets_values_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "output_offsets_values",
                                TensorShape({output_offsets_values_size}),
                                &output_offsets_values_tensor));
    auto output_offsets_values_data =
        output_offsets_values_tensor->flat<int64>().data();
    memcpy(output_offsets_values_data, output_offsets_values.data(),
           output_offsets_values_size * sizeof(int64));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FindSourceOffsetsOp);
};

REGISTER_KERNEL_BUILDER(Name("FindSourceOffsets")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int64>("Tsplits"),
                        FindSourceOffsetsOp<int64>);
REGISTER_KERNEL_BUILDER(Name("FindSourceOffsets")
                            .Device(tensorflow::DEVICE_CPU)
                            .TypeConstraint<int32>("Tsplits"),
                        FindSourceOffsetsOp<int32>);
}  // namespace text
}  // namespace tensorflow
