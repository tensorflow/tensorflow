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

#include "fuzztest/fuzztest.h"
#include "tensorflow/cc/ops/audio_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/security/fuzzing/cc/fuzz_session.h"

namespace tensorflow {
namespace fuzzing {

// Image op fuzzers
// DecodePng
class FuzzDecodePng : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("contents"), DT_STRING);
    tensorflow::ops::DecodePng(scope.WithOpName("image"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<tstring>()() =
        string(input_string.c_str(), input_string.size());
    Status s = RunInputsWithStatus({{"contents", input_tensor}});
    if (!s.ok()) {
      LOG(ERROR) << "Execution failed: " << s.message();
    }
  }
};
FUZZ_TEST_F(FuzzDecodePng, Fuzz)
    .WithDomains(fuzztest::OneOf(fuzztest::InRegexp("[-.0-9]+"),
                                 fuzztest::Arbitrary<std::string>()));

// DecodeJpeg
class FuzzDecodeJpeg : public FuzzSession<std::string> {
  void BuildGraph(const Scope& scope) override {
    auto op_node =
        tensorflow::ops::Placeholder(scope.WithOpName("contents"), DT_STRING);
    tensorflow::ops::DecodeJpeg(scope.WithOpName("image"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
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
    tensorflow::ops::DecodeGif(scope.WithOpName("image"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
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
    tensorflow::ops::DecodeImage(scope.WithOpName("image"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
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
    tensorflow::ops::DecodeBmp(scope.WithOpName("image"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
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
    tensorflow::ops::DecodeAndCropJpeg(scope.WithOpName("image"), op_node,
                                       crop_window);
  }

  void FuzzImpl(const std::string& input_string,
                const int32& crop_window_val) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<tstring>()() =
        string(input_string.c_str(), input_string.size());
    Tensor crop_window(tensorflow::DT_INT32, TensorShape({}));
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
    tensorflow::ops::DecodeWav(scope.WithOpName("image"), op_node);
  }

  void FuzzImpl(const std::string& input_string) final {
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
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
