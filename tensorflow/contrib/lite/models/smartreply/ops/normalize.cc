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

// Normalize the string input.
//
// Input:
//     Input[0]: One sentence. string[1]
//
// Output:
//     Output[0]: Normalized sentence. string[1]
//

#include <algorithm>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "re2/re2.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/string_util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace normalize {

// Predictor transforms.
const char kPunctuationsRegex[] = "[.*()\"]";

const std::map<string, string>* kRegexTransforms =
    new std::map<string, string>({
        {"([^\\s]+)n't", "\\1 not"},
        {"([^\\s]+)'nt", "\\1 not"},
        {"([^\\s]+)'ll", "\\1 will"},
        {"([^\\s]+)'re", "\\1 are"},
        {"([^\\s]+)'ve", "\\1 have"},
        {"i'm", "i am"},
    });

static const char kStartToken[] = "<S>";
static const char kEndToken[] = "<E>";
static const int32_t kMaxInputChars = 300;

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  tflite::StringRef input = tflite::GetString(GetInput(context, node, 0), 0);

  string result(absl::AsciiStrToLower(absl::string_view(input.str, input.len)));
  absl::StripAsciiWhitespace(&result);
  // Do not remove commas, semi-colons or colons from the sentences as they can
  // indicate the beginning of a new clause.
  RE2::GlobalReplace(&result, kPunctuationsRegex, "");
  RE2::GlobalReplace(&result, "\\s('t|'nt|n't|'d|'ll|'s|'m|'ve|'re)([\\s,;:/])",
                     "\\1\\2");
  RE2::GlobalReplace(&result, "\\s('t|'nt|n't|'d|'ll|'s|'m|'ve|'re)$", "\\1");
  for (auto iter = kRegexTransforms->begin(); iter != kRegexTransforms->end();
       iter++) {
    RE2::GlobalReplace(&result, iter->first, iter->second);
  }

  // Treat questions & interjections as special cases.
  RE2::GlobalReplace(&result, "([?])+", "\\1");
  RE2::GlobalReplace(&result, "([!])+", "\\1");
  RE2::GlobalReplace(&result, "([^?!]+)([?!])", "\\1 \\2 ");
  RE2::GlobalReplace(&result, "([?!])([?!])", "\\1 \\2");

  RE2::GlobalReplace(&result, "[\\s,:;\\-&'\"]+$", "");
  RE2::GlobalReplace(&result, "^[\\s,:;\\-&'\"]+", "");
  absl::StripAsciiWhitespace(&result);

  // Add start and end token.
  // Truncate input to maximum allowed size.
  if (result.length() <= kMaxInputChars) {
    absl::StrAppend(&result, " ", kEndToken);
  } else {
    result = result.substr(0, kMaxInputChars);
  }
  result = absl::StrCat(kStartToken, " ", result);

  tflite::DynamicBuffer buf;
  buf.AddString(result.data(), result.length());
  buf.WriteToTensor(GetOutput(context, node, 0));
  return kTfLiteOk;
}

}  // namespace normalize

TfLiteRegistration* Register_NORMALIZE() {
  static TfLiteRegistration r = {nullptr, nullptr, nullptr, normalize::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
