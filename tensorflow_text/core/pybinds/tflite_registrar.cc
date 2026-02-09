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

#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "tensorflow_text/core/kernels/byte_splitter_tflite.h"
#include "tensorflow_text/core/kernels/fast_bert_normalizer_tflite.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_tflite.h"
#include "tensorflow_text/core/kernels/ngrams_tflite.h"
#include "tensorflow_text/core/kernels/ragged_tensor_to_tensor_tflite.h"
#include "tensorflow_text/core/kernels/round_robin_trimmer_tflite.h"
#include "tensorflow_text/core/kernels/sentence_fragmenter_v2_tflite.h"
#include "tensorflow_text/core/kernels/sentencepiece/py_tflite_registerer.h"
#include "tensorflow_text/core/kernels/utf8_binarize_tflite.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer_tflite.h"

PYBIND11_MODULE(tflite_registrar, m) {
  m.doc() = R"pbdoc(
    tflite_registrar
    A module with a Python wrapper for TFLite TFText ops.
  )pbdoc";
  m.attr("_allowed_symbols") = pybind11::make_tuple(
      "AddByteSplit", "AddByteSplitByOffsets", "AddFastBertNormalize",
      "AddFastSentencepieceDetokenize", "AddFastSentencepieceTokenize",
      "AddFastWordpieceTokenize", "AddFastWordpieceDetokenize",
      "AddNgramsStringJoin", "AddRaggedTensorToTensor",
      "AddRoundRobinGenerateMasks", "AddRoundRobinTrim",
      "AddSentenceFragmenterV2", "AddUtf8Binarize", "AddWhitespaceTokenize",
      "SELECT_TFTEXT_OPS");
  m.def(
      "AddByteSplit",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddByteSplit(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds AddByteSplit to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddByteSplitByOffsets",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddByteSplitByOffsets(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds AddByteSplitByOffsets to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddFastBertNormalize",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddFastBertNormalize(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds FastBertNormalize to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddFastSentencepieceDetokenize",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddFastSentencepieceDetokenize(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      Adds AddFastSentencepieceDetokenize to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddFastSentencepieceTokenize",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddFastSentencepieceTokenize(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      Adds AddFastSentencepieceTokenize to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddFastWordpieceTokenize",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddFastWordpieceTokenize(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds FastWordpieceTokenize to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddFastWordpieceDetokenize",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddFastWordpieceDetokenize(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
    The function that adds FastWordpieceDetokenize to the TFLite interpreter.
    )pbdoc");
  m.def(
      "AddNgramsStringJoin",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddNgramsStringJoin(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
    The function that adds AddNgramsStringJoin to the TFLite interpreter.
    )pbdoc");
  m.def(
      "AddRaggedTensorToTensor",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddRaggedTensorToTensor(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds AddRaggedTensorToTensor to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddRoundRobinGenerateMasks",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddRoundRobinGenerateMasks(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds AddRoundRobinGenerateMasks to the TFLite
      interpreter.
      )pbdoc");
  m.def(
      "AddRoundRobinTrim",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddRoundRobinTrim(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds AddRoundRobinTrim to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddSentenceFragmenterV2",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddSentenceFragmenterV2(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds AddSentenceFragmenterV2 to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddUtf8Binarize",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddUtf8Binarize(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds AddUtf8Binarize to the TFLite interpreter.
      )pbdoc");
  m.def(
      "AddWhitespaceTokenize",
      [](uintptr_t resolver) {
        tflite::ops::custom::text::AddWhitespaceTokenize(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
      The function that adds AddWhitespaceTokenize to the TFLite interpreter.
      )pbdoc");
}
