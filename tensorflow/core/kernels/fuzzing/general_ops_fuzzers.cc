/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "fuzztest/fuzztest.h"
#include "tensorflow/cc/ops/audio_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

// DecodePng
class FuzzDecodePng : public FuzzStringInputOp {
  SINGLE_INPUT_OP_BUILDER(DT_STRING, DecodePng);
};
STANDARD_TF_FUZZ_FUZZTEST_FUNCTION(FuzzDecodePng, FuzzDecodePngEntry);
FUZZ_TEST(GeneralOpsFuzzer, FuzzDecodePngEntry);

// DecodeWav
class FuzzDecodeWav : public FuzzStringInputOp {
  SINGLE_INPUT_OP_BUILDER(DT_STRING, DecodeWav);
};
STANDARD_TF_FUZZ_FUZZTEST_FUNCTION(FuzzDecodeWav, FuzzDecodeWavEntry);
FUZZ_TEST(GeneralOpsFuzzer, FuzzDecodeWavEntry);

// DecodeBmp
class FuzzDecodeBmp : public FuzzStringInputOp {
  SINGLE_INPUT_OP_BUILDER(DT_STRING, DecodeBmp);
};
STANDARD_TF_FUZZ_FUZZTEST_FUNCTION(FuzzDecodeBmp, FuzzDecodeBmpEntry);
FUZZ_TEST(GeneralOpsFuzzer, FuzzDecodeBmpEntry);

// DecodeGif
class FuzzDecodeGif : public FuzzStringInputOp {
  SINGLE_INPUT_OP_BUILDER(DT_STRING, DecodeGif);
};
STANDARD_TF_FUZZ_FUZZTEST_FUNCTION(FuzzDecodeGif, FuzzDecodeGifEntry);
FUZZ_TEST(GeneralOpsFuzzer, FuzzDecodeGifEntry);

// DecodeJpeg
class FuzzDecodeJpeg : public FuzzStringInputOp {
  SINGLE_INPUT_OP_BUILDER(DT_STRING, DecodeJpeg);
};
STANDARD_TF_FUZZ_FUZZTEST_FUNCTION(FuzzDecodeJpeg, FuzzDecodeJpegEntry);
FUZZ_TEST(GeneralOpsFuzzer, FuzzDecodeJpegEntry);

// DecodeImage
class FuzzDecodeImage : public FuzzStringInputOp {
  SINGLE_INPUT_OP_BUILDER(DT_STRING, DecodeImage);
};
STANDARD_TF_FUZZ_FUZZTEST_FUNCTION(FuzzDecodeImage, FuzzDecodeImageEntry);
FUZZ_TEST(GeneralOpsFuzzer, FuzzDecodeImageEntry);

// EncodeBase64
class FuzzEncodeBase64 : public FuzzStringInputOp {
  SINGLE_INPUT_OP_BUILDER(DT_STRING, EncodeBase64);
};
STANDARD_TF_FUZZ_FUZZTEST_FUNCTION(FuzzEncodeBase64, FuzzEncodeBase64Entry);
FUZZ_TEST(GeneralOpsFuzzer, FuzzEncodeBase64Entry);

// DecodeCsv
class FuzzDecodeCsv : public FuzzStringInputOp {
  void BuildGraph(const Scope& scope) override {
    auto input =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    // For now, assume we want CSVs with 4 columns, as we need a refactoring
    // of the entire infrastructure to support the more complex usecase due to
    // the fact that graph generation and fuzzing data are at separate steps.
    InputList defaults = {Input("a"), Input("b"), Input("c"), Input("d")};
    (void)tensorflow::ops::DecodeCSV(scope.WithOpName("output"), input,
                                     defaults);
  }
};
STANDARD_TF_FUZZ_FUZZTEST_FUNCTION(FuzzDecodeCsv, FuzzDecodeCsvEntry);
FUZZ_TEST(GeneralOpsFuzzer, FuzzDecodeCsvEntry);

// DecodeCompressed
class FuzzDecodeCompressed : public FuzzStringInputOp {
  void BuildGraph(const Scope& scope) override {
    auto input =
        tensorflow::ops::Placeholder(scope.WithOpName("input"), DT_STRING);
    auto d1 = tensorflow::ops::DecodeCompressed(
        scope.WithOpName("d1"), input,
        tensorflow::ops::DecodeCompressed::CompressionType(""));
    auto d2 = tensorflow::ops::DecodeCompressed(
        scope.WithOpName("d2"), input,
        tensorflow::ops::DecodeCompressed::CompressionType("ZLIB"));
    auto d3 = tensorflow::ops::DecodeCompressed(
        scope.WithOpName("d3"), input,
        tensorflow::ops::DecodeCompressed::CompressionType("GZIP"));
    Scope grouper =
        scope.WithControlDependencies(std::vector<tensorflow::Operation>{
            d1.output.op(), d2.output.op(), d3.output.op()});
    (void)tensorflow::ops::NoOp(grouper.WithOpName("output"));
  }
};
STANDARD_TF_FUZZ_FUZZTEST_FUNCTION(FuzzDecodeCompressed,
                                   FuzzDecodeCompressedEntry);
FUZZ_TEST(GeneralOpsFuzzer, FuzzDecodeCompressedEntry);

}  // end namespace fuzzing
}  // end namespace tensorflow
