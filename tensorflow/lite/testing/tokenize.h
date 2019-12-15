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
#ifndef TENSORFLOW_LITE_TESTING_TOKENIZE_H_
#define TENSORFLOW_LITE_TESTING_TOKENIZE_H_

#include <istream>
#include <string>

namespace tflite {
namespace testing {

// Process tokens coming from Tokenize().
class TokenProcessor {
 public:
  virtual ~TokenProcessor() {}
  // Process a single token. The token won't be reused, so it is OK to call
  // token.swap().
  virtual void ConsumeToken(std::string* token) = 0;
};

// Tokenize a stream on whitespaces, colons and curly braces. Whitespaces are
// removed from the tokens and double-quotes can be used to avoid that. Note
// that there is no way to escape double-quotes, so there's no way to have a
// double-quote inside a token.
void Tokenize(std::istream* input, TokenProcessor* processor);

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_TOKENIZE_H_
