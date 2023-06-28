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

#include <fuzzer/FuzzedDataProvider.h>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/text_literal_reader.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/env.h"

namespace xla {
namespace {

void FuzzFileRead(std::string data) {
  std::string fname = "/tmp/ReadsR3File.txt";
  if (!tsl::WriteStringToFile(tsl::Env::Default(), fname, data.c_str()).ok()) {
    return;
  }

  if (TextLiteralReader::ReadPath(fname).ok()) {
    return;
  }
}
FUZZ_TEST(TextReaderFuzzer, FuzzFileRead);

}  // namespace
}  // namespace xla
