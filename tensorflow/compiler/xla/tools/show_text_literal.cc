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

// Usage: show_text_literal <path-to-serialized-literal-text>

#include <stdio.h>
#include <algorithm>
#include <memory>
#include <string>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/text_literal_reader.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

int main(int argc, char **argv) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);

  if (argc < 2) {
    LOG(QFATAL) << "Usage: " << argv[0] << " <path-to-serialized-literal-text>";
  }

  xla::Literal literal =
      xla::TextLiteralReader::ReadPath(argv[1]).ConsumeValueOrDie();

  LOG(INFO) << "literal: " << literal;
  fprintf(stderr, "%s\n", literal.ToString().c_str());
  if (literal.shape().element_type() == xla::F32) {
    float min = *std::min_element(literal.data<float>().begin(),
                                  literal.data<float>().end());
    float max = *std::max_element(literal.data<float>().begin(),
                                  literal.data<float>().end());
    fprintf(stderr, "min: %a=%f\n", min, min);
    fprintf(stderr, "max: %a=%f\n", max, max);
  }
}
