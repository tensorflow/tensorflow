// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "fuzztest/fuzztest.h"
#include "absl/log/log.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/audio_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/security/fuzzing/cc/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

// Image op fuzzers
// DecodePng
class FuzzDecodePng : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("contents"), DT_STRING);
    tensorflow::ops::DecodePng(scope.WithOpName("output"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(DT_STRING, {});
    input_tensor.scalar<tstring>()() =
        string(input_string.c_str(), input_string.size());
    Status s = RunInputsWithStatus({{"contents", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }

 public:
  void FuzzValidInput(const std::string& input_string) { Fuzz(input_string); }

  void FuzzArbitraryInput(const std::string& input_string) {
    Fuzz(input_string);
  }
};
FUZZ_TEST_F(FuzzDecodePng, FuzzValidInput)
    .WithDomains(fuzztest::InRegexp("[-.0-9]+"));
FUZZ_TEST_F(FuzzDecodePng, FuzzArbitraryInput);

// DecodeJpeg
class FuzzDecodeJpeg : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("contents"), DT_STRING);
    tensorflow::ops::DecodeJpeg(scope.WithOpName("output"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(DT_STRING, {});
    input_tensor.scalar<tstring>()() =
        string(input_string.c_str(), input_string.size());
    Status s = RunInputsWithStatus({{"contents", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzDecodeJpeg, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

// DecodeGif
class FuzzDecodeGif : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("contents"), DT_STRING);
    tensorflow::ops::DecodeGif(scope.WithOpName("output"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(DT_STRING, {});
    input_tensor.scalar<tstring>()() =
        string(input_string.c_str(), input_string.size());
    Status s = RunInputsWithStatus({{"contents", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzDecodeGif, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

// DecodeJpeg
class FuzzDecodeImage : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("contents"), DT_STRING);
    tensorflow::ops::DecodeImage(scope.WithOpName("output"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(DT_STRING, {});
    input_tensor.scalar<tstring>()() =
        string(input_string.c_str(), input_string.size());
    Status s = RunInputsWithStatus({{"contents", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzDecodeImage, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

// DecodeBmp
class FuzzDecodeBmp : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("contents"), DT_STRING);
    tensorflow::ops::DecodeBmp(scope.WithOpName("output"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(DT_STRING, {});
    input_tensor.scalar<tstring>()() =
        string(input_string.c_str(), input_string.size());
    Status s = RunInputsWithStatus({{"contents", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzDecodeBmp, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

// DecodeAndCropJpeg
class FuzzDecodeAndCropJpeg : public FuzzSession<std::string, int32> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("contents"), DT_STRING);
    auto crop_window =
        tensorflow::ops::Placeholder(scope.WithOpName("crop_window"), DT_INT32);
    tensorflow::ops::DecodeAndCropJpeg(scope.WithOpName("output"), op_node,
                                       crop_window);
  }

  void FuzzImpl(const std::string& input_string,
                const int32& crop_window_val) final {
    Tensor input_tensor(DT_STRING, {});
    input_tensor.scalar<tstring>()() =
        string(input_string.c_str(), input_string.size());
    Tensor crop_window(DT_INT32, {});
    crop_window.scalar<int32>()() = crop_window_val;

    Status s = RunInputsWithStatus(
        {{"contents", input_tensor}, {"crop_window", crop_window}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzDecodeAndCropJpeg, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()),
                 fuzztest::InRange<int32>(0, 4096));

// Audio decoder
class FuzzDecodeWav : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("contents"), DT_STRING);
    tensorflow::ops::DecodeWav(scope.WithOpName("output"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(DT_STRING, {});
    input_tensor.scalar<tstring>()() =
        string(input_string.c_str(), input_string.size());
    Status s = RunInputsWithStatus({{"contents", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzDecodeWav, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

}  // end namespace fuzzing
}  // end namespace tensorflow
