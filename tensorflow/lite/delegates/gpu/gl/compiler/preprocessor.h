/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_PREPROCESSOR_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_PREPROCESSOR_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace gl {

enum class RewriteStatus {
  SUCCESS = 0,
  NOT_RECOGNIZED = 1,
  ERROR = 2,
};

// Inline rewrite matches a string and rewrites it.
class InlineRewrite {
 public:
  virtual ~InlineRewrite() = default;

  virtual RewriteStatus Rewrite(absl::string_view input,
                                std::string* output) = 0;
};

// Text preprocessor runs a collection of registered rewrites.
// It uses a single character prefix as inline delimiter that needs to quote
// text to be rewritten.
class TextPreprocessor {
 public:
  // @param keep_unknown_rewrites if true, will keep unhandled rewrites as is
  // instead of reporting an error.
  TextPreprocessor(char inline_delimiter, bool keep_unknown_rewrites)
      : inline_delimiter_(inline_delimiter),
        keep_unknown_rewrites_(keep_unknown_rewrites) {}

  void AddRewrite(InlineRewrite* rewrite) {
    inline_rewrites_.push_back(rewrite);
  }

  // input and output may point to the same object.
  Status Rewrite(const std::string& input, std::string* output);

 private:
  const char inline_delimiter_;
  const bool keep_unknown_rewrites_;

  std::vector<InlineRewrite*> inline_rewrites_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_PREPROCESSOR_H_
