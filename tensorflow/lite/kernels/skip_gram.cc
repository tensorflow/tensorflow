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

// Generate a list of skip grams from an input.
//
// Options:
//   ngram_size: num of words for each output item.
//   max_skip_size: max num of words to skip.
//                  The op generates ngrams when it is 0.
//   include_all_ngrams: include all ngrams with size up to ngram_size.
//
// Input:
//   A string tensor to generate n-grams.
//   Dim = {1}
//
// Output:
//   A list of strings, each of which contains ngram_size words.
//   Dim = {num_ngram}

#include <ctype.h>

#include <vector>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {

namespace {

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input_tensor;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input_tensor));
  TF_LITE_ENSURE_TYPES_EQ(context, input_tensor->type, kTfLiteString);
  TfLiteTensor* output_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output_tensor));
  TF_LITE_ENSURE_TYPES_EQ(context, output_tensor->type, kTfLiteString);
  return kTfLiteOk;
}

bool ShouldIncludeCurrentNgram(const TfLiteSkipGramParams* params, int size) {
  if (size <= 0) {
    return false;
  }
  if (params->include_all_ngrams) {
    return size <= params->ngram_size;
  } else {
    return size == params->ngram_size;
  }
}

bool ShouldStepInRecursion(const TfLiteSkipGramParams* params,
                           const std::vector<int>& stack, int stack_idx,
                           int num_words) {
  // If current stack size and next word enumeration are within valid range.
  if (stack_idx < params->ngram_size && stack[stack_idx] + 1 < num_words) {
    // If this stack is empty, step in for first word enumeration.
    if (stack_idx == 0) {
      return true;
    }
    // If next word enumeration are within the range of max_skip_size.
    // NOTE: equivalent to
    //   next_word_idx = stack[stack_idx] + 1
    //   next_word_idx - stack[stack_idx-1] <= max_skip_size + 1
    if (stack[stack_idx] - stack[stack_idx - 1] <= params->max_skip_size) {
      return true;
    }
  }
  return false;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSkipGramParams*>(node->builtin_data);

  // Split sentence to words.
  std::vector<StringRef> words;
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  tflite::StringRef strref = tflite::GetString(input, 0);
  int prev_idx = 0;
  for (int i = 1; i < strref.len; i++) {
    if (isspace(*(strref.str + i))) {
      if (i > prev_idx && !isspace(*(strref.str + prev_idx))) {
        words.push_back({strref.str + prev_idx, i - prev_idx});
      }
      prev_idx = i + 1;
    }
  }
  if (strref.len > prev_idx) {
    words.push_back({strref.str + prev_idx, strref.len - prev_idx});
  }

  // Generate n-grams recursively.
  tflite::DynamicBuffer buf;
  if (words.size() < params->ngram_size) {
    buf.WriteToTensorAsVector(GetOutput(context, node, 0));
    return kTfLiteOk;
  }

  // Stack stores the index of word used to generate ngram.
  // The size of stack is the size of ngram.
  std::vector<int> stack(params->ngram_size, 0);
  // Stack index that indicates which depth the recursion is operating at.
  int stack_idx = 1;
  int num_words = words.size();

  while (stack_idx >= 0) {
    if (ShouldStepInRecursion(params, stack, stack_idx, num_words)) {
      // When current depth can fill with a new word
      // and the new word is within the max range to skip,
      // fill this word to stack, recurse into next depth.
      stack[stack_idx]++;
      stack_idx++;
      if (stack_idx < params->ngram_size) {
        stack[stack_idx] = stack[stack_idx - 1];
      }
    } else {
      if (ShouldIncludeCurrentNgram(params, stack_idx)) {
        // Add n-gram to tensor buffer when the stack has filled with enough
        // words to generate the ngram.
        std::vector<StringRef> gram(stack_idx);
        for (int i = 0; i < stack_idx; i++) {
          gram[i] = words[stack[i]];
        }
        buf.AddJoinedString(gram, ' ');
      }
      // When current depth cannot fill with a valid new word,
      // and not in last depth to generate ngram,
      // step back to previous depth to iterate to next possible word.
      stack_idx--;
    }
  }

  buf.WriteToTensorAsVector(GetOutput(context, node, 0));
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration* Register_SKIP_GRAM() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare, Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
