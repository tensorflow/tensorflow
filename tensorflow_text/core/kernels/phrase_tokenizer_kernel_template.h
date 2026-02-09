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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_KERNEL_TEMPLATE_H_

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow_text/core/kernels/phrase_tokenizer.h"

namespace tensorflow {
namespace text {

// See `kDoc` data member for the documentation on this op kernel.
//
// This template class can be instantiated into a kernel for either TF or
// TFLite. See
// https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels/shim
// for more info on how this works.
template <tflite::shim::Runtime Rt>
class PhraseTokenizeOp
    : public tflite::shim::OpKernelShim<PhraseTokenizeOp, Rt> {
 private:
  enum Inputs { kInputValues = 0, kPhraseModel };
  enum Outputs {
    kOutputSubwords = 0,
    kOutputIds,
    kOutputRowSplits,
  };

  using Shape = tflite::shim::Shape;
  using typename tflite::shim::OpKernelShim<PhraseTokenizeOp, Rt>::InitContext;
  using
      typename tflite::shim::OpKernelShim<PhraseTokenizeOp, Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<PhraseTokenizeOp,
                                            Rt>::ShapeInferenceContext;

 public:
  PhraseTokenizeOp() = default;
  static constexpr char kOpName[] = "PhraseTokenize";
  static constexpr char kDoc[] = R"doc(
    Tokenizes tokens into phrases based off of a vocabulary.

    ### Example:

    ```python
    >>> tokens = ['I have a dream', 'I like coffee']
    >>> phrase, ids, row_splits = (
    ...       phrase_tokenize(tokens, model_buffer))
    >>> RaggedTensor.from_row_splits(phrase, row_splits)
    [['I', 'have', 'a dream'], ['I like', 'coffee']]
    >>> RaggedTensor.from_row_splits(ids, row_splits)
    [[0, 1, 2], [3, 4]]  # Dummy ids.
    ```

    Args:
      input_values: 1D Tensor of strings to tokenize with.
      phrase_model: Buffer tensor for the PhraseTokenizerConfig flatbuffer.

    Returns:
      * output_values: 1D tensor containing the phrases for all input strings.
        A 2D RaggedTensor can be constructed from this and output_row_splits.
      * output_ids: 1D tensor containing the phrase ids for all input strings.
        A 2D RaggedTensor can be constructed from this and output_row_splits.
      * output_row_splits: 1D int tensor with the row splits that allow us to
        build RaggedTensors from output_values, output_ids.
  )doc";

  static const char* OpName() { return kOpName; }
  static const char* Doc() { return kDoc; }

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs() { return {}; }

  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();

  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* context) { return absl::OkStatus(); }

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);
};

////////////////////////// Implementation

template <tflite::shim::Runtime Rt>
std::vector<std::string> PhraseTokenizeOp<Rt>::Inputs() {
  return {"input_values: string", "phrase_model: uint8"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> PhraseTokenizeOp<Rt>::Outputs() {
  return {"output_subwords: string", "output_ids: int64",
          "output_row_splits: int64"};
}

template <tflite::shim::Runtime Rt>
absl::Status PhraseTokenizeOp<Rt>::Invoke(InvokeContext* context) {
  SH_ASSIGN_OR_RETURN(const auto input_values, context->GetInput(kInputValues));
  const auto& values_vec = input_values->template As<tstring, 1>();

  SH_ASSIGN_OR_RETURN(const auto phrase_model, context->GetInput(kPhraseModel));
  // OK to create on every call because PhraseTokenizer is a
  // lightweight, memory-mapped wrapper on `phrase_model` tensor, and thus
  // Create() is very cheap.
  auto phrase_tokenizer = ::tensorflow::text::PhraseTokenizer::Create(
      phrase_model->template Data<uint8>().data());
  SH_RETURN_IF_ERROR(phrase_tokenizer.status());

  std::vector<std::string> subwords;
  std::vector<int> subword_ids;
  std::vector<int> row_splits;

  row_splits.push_back(0);

  // Iterate through all the values and wordpiece tokenize them.
  for (int i = 0; i < values_vec.Dim(0); ++i) {
    // Tokenize into subwords and record the offset locations.
    const int original_num_wordpieces = subwords.size();
    phrase_tokenizer->Tokenize(values_vec(i), &subwords, &subword_ids);
    const int delta_num_wordpieces = subwords.size() - original_num_wordpieces;

    // Record the row splits.
    row_splits.push_back(delta_num_wordpieces + row_splits.back());
  }

  const int subwords_size = subwords.size();
  SH_ASSIGN_OR_RETURN(
      auto output_subwords,
      context->GetOutput(kOutputSubwords, Shape({subwords_size})));
  auto output_subwords_vec =
      output_subwords->template As<tensorflow::tstring, 1>();

  SH_ASSIGN_OR_RETURN(
      auto output_ids,
      context->GetOutput(
          kOutputIds,
          Shape({static_cast<int>(
              subword_ids.size())}))); /* same shape as `output_subwords` */
  auto output_ids_vec = output_ids->template As<int64, 1>();

  SH_ASSIGN_OR_RETURN(
      auto output_row_splits,
      context->GetOutput(kOutputRowSplits,
                         Shape({static_cast<int>(row_splits.size())})));
  auto output_row_splits_vec = output_row_splits->template As<int64, 1>();

  for (int i = 0; i < subwords.size(); ++i) {
    output_subwords_vec(i) = subwords[i];
  }

  for (int i = 0; i < subword_ids.size(); ++i) {
    output_ids_vec(i) = subword_ids[i];
  }

  for (int i = 0; i < row_splits.size(); ++i) {
    output_row_splits_vec(i) = row_splits[i];
  }

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status PhraseTokenizeOp<Rt>::ShapeInference(ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  SH_ASSIGN_OR_RETURN(const Shape input_values_shape,
                      c->GetInputShape(kInputValues));
  SH_ASSIGN_OR_RETURN(const auto phrase_model_shape,
                      c->GetInputShape(kPhraseModel));
  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", input_values_shape.ToString()));
  }
  if (!phrase_model_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", phrase_model_shape.ToString()));
  }
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputSubwords, rank_1_shape));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputIds, rank_1_shape));
  // row splits size
  const int num_splits = Shape::AddDims(1, input_values_shape.Dim(0));
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputRowSplits, Shape({num_splits})));

  return absl::OkStatus();
}

// See `kDoc` data member for the documentation on this op kernel.
//
// This template class can be instantiated into a kernel for either TF or
// TFLite. See
// https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/kernels/shim
// for more info on how this works.
template <tflite::shim::Runtime Rt>
class PhraseDetokenizeOp
    : public tflite::shim::OpKernelShim<PhraseDetokenizeOp, Rt> {
 private:
  enum Inputs { kInputValues = 0, kInputRowSplits, kPhraseModel };
  enum Outputs { kOutputWords = 0 };

  using Shape = tflite::shim::Shape;
  using
      typename tflite::shim::OpKernelShim<PhraseDetokenizeOp, Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<PhraseDetokenizeOp,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<PhraseDetokenizeOp,
                                            Rt>::ShapeInferenceContext;

 public:
  PhraseDetokenizeOp() = default;
  static constexpr char kOpName[] = "TFText>PhraseDetokenize";
  static constexpr char kDoc[] = R"doc(
    Detokenizes phrase ids into sentences.

    ### Example:

    ```python
    >>> # Vocab of the model_buffer: ['I', 'have', 'a dream'].
    >>> wordpiece_ids = [2, 3, 4]
    >>> row_splits = [0, 2, 3]
    >>> tokens = phrase_tokenizer_detokenize(tokens, row_splits, model_buffer)
    >>> tokens
    ['I have', 'a dream']
    ```

    Args:
      input_values: 1D Tensor of phrase ids.
      input_row_splits: 1D Tensor of row splits that denotes the boundary of each
        sentence in the `input_values`.
      phrase_model: Buffer tensor for the PhraseTokenizerConfig flatbuffer.

    Returns:
      * output_values: 1D tensor containing all the sentences.
  )doc";

  static const char* OpName() { return kOpName; }
  static const char* Doc() { return kDoc; }

  // Attributes declaration (syntax: https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Attrs() { return {}; }

  // Input tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Inputs();

  // Output tensors declaration (syntax:
  // https://www.tensorflow.org/guide/create_op)
  static std::vector<std::string> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* context) { return absl::OkStatus(); }

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);
};

////////////////////////// Implementation

template <tflite::shim::Runtime Rt>
std::vector<std::string> PhraseDetokenizeOp<Rt>::Inputs() {
  return {"input_values: int32", "input_row_splits: int64",
          "phrase_model: uint8"};
}

template <tflite::shim::Runtime Rt>
std::vector<std::string> PhraseDetokenizeOp<Rt>::Outputs() {
  return {"output_words: string"};
}

template <tflite::shim::Runtime Rt>
absl::Status PhraseDetokenizeOp<Rt>::Invoke(InvokeContext* context) {
  SH_ASSIGN_OR_RETURN(const auto input_values, context->GetInput(kInputValues));
  const auto& values_vec = input_values->template As<int, 1>();

  SH_ASSIGN_OR_RETURN(const auto input_row_splits,
                      context->GetInput(kInputRowSplits));
  const auto& row_splits_vec = input_row_splits->template As<int64, 1>();

  SH_ASSIGN_OR_RETURN(const auto phrase_model, context->GetInput(kPhraseModel));
  // OK to create on every call because PhraseTokenizer is a
  // lightweight, memory-mapped wrapper on `phrase_model` tensor, and thus
  // Create() is very cheap.
  auto phrase_tokenizer = ::tensorflow::text::PhraseTokenizer::Create(
      phrase_model->template Data<uint8>().data());
  SH_RETURN_IF_ERROR(phrase_tokenizer.status());

  std::vector<std::string> sentences;

  // Iterate through row_splits to split input_values.
  for (int i = 0; i < row_splits_vec.Dim(0) - 1; ++i) {
    auto single_input =
        absl::Span<const int>(values_vec.Ptr() + row_splits_vec(i),
                              row_splits_vec(i + 1) - row_splits_vec(i));
    SH_ASSIGN_OR_RETURN(auto sentence,
                        phrase_tokenizer->Detokenize(single_input));
    sentences.push_back(sentence);
  }

  SH_RETURN_IF_ERROR(this->template FillOutputTensor<std::string,
                                                     tensorflow::tstring>(
      sentences, kOutputWords, context));

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status PhraseDetokenizeOp<Rt>::ShapeInference(ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  SH_ASSIGN_OR_RETURN(const Shape input_values_shape,
                      c->GetInputShape(kInputValues));
  SH_ASSIGN_OR_RETURN(const Shape input_row_splits_shape,
                      c->GetInputShape(kInputRowSplits));
  SH_ASSIGN_OR_RETURN(const auto phrase_model_shape,
                      c->GetInputShape(kPhraseModel));
  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  if (!input_values_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", input_values_shape.ToString()));
  }
  if (!input_row_splits_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Shape must be rank 1: ", input_row_splits_shape.ToString()));
  }
  if (!phrase_model_shape.Compatible(rank_1_shape)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Shape must be rank 1: ", phrase_model_shape.ToString()));
  }
  SH_RETURN_IF_ERROR(c->SetOutputShape(kOutputWords, rank_1_shape));
  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_phrase_TOKENIZER_KERNEL_TEMPLATE_H_
