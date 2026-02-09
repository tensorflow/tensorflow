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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_BERT_NORMALIZER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_BERT_NORMALIZER_KERNEL_TEMPLATE_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/fast_bert_normalizer.h"

namespace tensorflow {
namespace text {

// See `kDoc` data member for the documentation on this op kernel.
//
// This template class can be instantiated into a kernel for either TF or
// TFLite. See go/tfshim for more info on how this works.
template <tflite::shim::Runtime Rt>
class FastBertNormalizeOp
    : public tflite::shim::OpKernelShim<FastBertNormalizeOp, Rt> {
 private:
  enum Inputs { kInputValues = 0, kFastBertNormalizerModel };
  enum Outputs {
    kOutputValues = 0,
    kOutputOffsets,
    kOutputRowSplitsOfOffsets,
  };

  using Shape = tflite::shim::Shape;
  using
      typename tflite::shim::OpKernelShim<FastBertNormalizeOp, Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<FastBertNormalizeOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<FastBertNormalizeOp,
                                            Rt>::ShapeInferenceContext;

  static const char kGetOffsetsAttr[];

  // The real work of the invoke operation.
  template <bool kGetOffsets>
  absl::Status InvokeRealWork(InvokeContext* context);

  bool get_offsets_;

 public:
  FastBertNormalizeOp() = default;
  static constexpr char kOpName[] = "FastBertNormalize";
  static constexpr char kDoc[] = R"doc(
    Normalizes texts.

    It returns the normalized texts and the relative offsets from the normalized
    text to the original text.

    Args:
      * input_values: 1D Tensor of strings to normalize.
      * fast_bert_normalizer_model: Buffer tensor for the FastBertNormalizerModel
        flatbuffer.

    Returns:
      * output_values: 1D tensor containing the normalized text for all input
        strings. The shape is the same as the input strings.
      * output_offsets: 1D tensor containing the offset mapping from the
        normalized text to the original text. A 2D RaggedTensor can be constructed
        from this and output_row_splits. For example, if the input is
        `input_values[i1...iN]` with `N` strings, the constructed 2D RaggedTensor
        `offsets[i1...iN, k]` is the byte offset in `input_values[i1...iN]` for
        the `kth` byte in `output_values[i1...iN]` after normalization. Note that
        `offsets[i1...iN, ...]` also covers the position following the last byte
        in the normalized `output_values[i1...iN]`, so that we know the byte
        offset position in `input_values[i1...iN]` that corresponds to the end of
        `output_values[i1...iN]`.
        
        
      * output_row_splits: 1D int tensor with the row splits that allow us to
        build RaggedTensors from output_offsets.
  )doc";

  static const char* OpName() { return kOpName; }
  static const char* Doc() { return kDoc; }

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs();

  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();

  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* context);

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);
};

////////////////////////// Implementation

template <tflite::shim::Runtime Rt>
const char FastBertNormalizeOp<Rt>::kGetOffsetsAttr[] =
    "get_offsets";

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastBertNormalizeOp<Rt>::Attrs() {
  return {
      absl::StrCat(kGetOffsetsAttr, ": bool = false"),
  };
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastBertNormalizeOp<Rt>::Inputs() {
  return {"input_values: string", "fast_bert_normalizer_model: uint8"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> FastBertNormalizeOp<Rt>::Outputs() {
  return {"output_values: string", "output_offsets: int64",
          "output_row_splits: int64"};
}

template <tflite::shim::Runtime Rt>
absl::Status FastBertNormalizeOp<Rt>::Init(InitContext* context) {
  SH_RETURN_IF_ERROR(
      context->GetAttr(kGetOffsetsAttr, &get_offsets_));
  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status FastBertNormalizeOp<Rt>::Invoke(InvokeContext* context) {
  if (get_offsets_) {
    return InvokeRealWork</*kGetOffsets=*/true>(context);
  } else {
    return InvokeRealWork</*kGetOffsets=*/false>(context);
  }
}

template <tflite::shim::Runtime Rt>
template <bool kGetOffsets>
absl::Status FastBertNormalizeOp<Rt>::InvokeRealWork(InvokeContext* context) {
  SH_ASSIGN_OR_RETURN(const auto input_values, context->GetInput(kInputValues));
  const auto& values_vec = input_values->template As<tstring, 1>();

  SH_ASSIGN_OR_RETURN(const auto fast_bert_normalizer_model,
                      context->GetInput(kFastBertNormalizerModel));
  // OK to create on every call because FastBertNormalizer is a lightweight,
  // memory-mapped wrapper on `fast_bert_normalizer_model` tensor, and thus
  // Create() is very cheap.
  auto text_normalizer = FastBertNormalizer::Create(
      fast_bert_normalizer_model->template Data<uint8>().data());
  SH_RETURN_IF_ERROR(text_normalizer.status());

  SH_ASSIGN_OR_RETURN(
      auto output_values,
      context->GetOutput(kOutputValues, Shape(input_values->Shape())));
  auto output_values_vec = output_values->template As<tensorflow::tstring, 1>();
  std::vector<int> offsets;
  std::vector<int> row_splits;

  if constexpr (kGetOffsets) {
    row_splits.push_back(0);
  }

  // Iterate through all the values and normalize them.
  for (int i = 0; i < values_vec.Dim(0); ++i) {
    // Normalize and record the offset locations.
    std::string normalized_string;
    bool is_normalized_string_identical;
    const int original_size = offsets.size();

    text_normalizer->template NormalizeText</*kGetOffsets=*/kGetOffsets>(
        values_vec(i), &is_normalized_string_identical, &normalized_string,
        &offsets);
    if (is_normalized_string_identical) {
      // When the input string is not changed after normalization,
      // `normalized_string` is empty and `offsets` is not changed by
      // the above function. So here we construct the corresponding result and
      // append to the final output.
      output_values_vec(i) = values_vec(i);  // The normalized text.
      if constexpr (kGetOffsets) {
        // The offset mapping will be the identy mapping.
        for (int j = 0; j < values_vec(i).size(); ++j) {
          offsets.push_back(j);
        }
        // The mapping from the end of the output to the end of the input.
        offsets.push_back(values_vec(i).size());
      }
    } else {
      output_values_vec(i) = normalized_string;
    }

    if constexpr (kGetOffsets) {
      // Record the row splits.
      const int delta_size = offsets.size() - original_size;
      row_splits.push_back(delta_size + row_splits.back());
    }
  }

  if constexpr (kGetOffsets) {
    SH_RETURN_IF_ERROR(this->template FillOutputTensor<int, int64>(
        offsets, kOutputOffsets, context));
    SH_RETURN_IF_ERROR(this->template FillOutputTensor<int, int64>(
        row_splits, kOutputRowSplitsOfOffsets, context));
  } else {
    SH_RETURN_IF_ERROR(this->template FillOutputTensor<int, int64>(
        offsets, kOutputOffsets, context));
    row_splits.resize(1+values_vec.Dim(0));
    SH_RETURN_IF_ERROR(this->template FillOutputTensor<int, int64>(
        row_splits, kOutputRowSplitsOfOffsets, context));
  }
  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status FastBertNormalizeOp<Rt>::ShapeInference(ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  SH_ASSIGN_OR_RETURN(const Shape input_values_shape,
                      c->GetInputShape(kInputValues));
  SH_ASSIGN_OR_RETURN(const auto fast_bert_normalizer_model_shape,
                      c->GetInputShape(kFastBertNormalizerModel));
  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Input values shape must be rank 1: ", input_values_shape.ToString()));
  }
  if (!fast_bert_normalizer_model_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Fast BERT normalizer model shape must be rank 1: ",
                     fast_bert_normalizer_model_shape.ToString()));
  }
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputValues, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputOffsets, rank_1_shape));
  // row splits size
  const int num_splits = Shape::AddDims(1, input_values_shape.Dim(0));
  SH_RETURN_IF_ERROR(
      c->SetOutputShape(kOutputRowSplitsOfOffsets, Shape({num_splits})));

  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow
#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_BERT_NORMALIZER_KERNEL_TEMPLATE_H_
