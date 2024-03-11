/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_CAST_TEST_COMMON_H_
#define TENSORFLOW_LITE_KERNELS_CAST_TEST_COMMON_H_

#include <stdint.h>

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/interpreter_options.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

using ::testing::ElementsAreArray;

class CastOpModel : public SingleOpModel {
 public:
  CastOpModel(const TensorData& input, const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_CAST, BuiltinOptions_CastOptions,
                 CreateCastOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  template <class ConstInputData>
  CastOpModel(const TensorData& input, ConstInputData&& data,
              const TensorData& output) {
    input_ = AddConstInput(input, static_cast<ConstInputData>(data));
    output_ = AddOutput(output);
    SetBuiltinOp(BuiltinOperator_CAST, BuiltinOptions_CastOptions,
                 CreateCastOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)}, /*num_threads=*/-1,
                     /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/true, /*allocate_and_delegate=*/false,
                     /*use_simple_allocator=*/false);
    InterpreterOptions options;
    options.SetCacheConstantCastOp(true);
    interpreter_->ApplyOptions(&options);
    AllocateAndDelegate(/*apply_delegate=*/true);
  }

  void Set4BitInput(absl::Span<const int8_t> f) {
    PopulateTensor4bit(input_, 0, f.data(), f.data() + f.size());
  }

  int input() const { return input_; }
  int output() const { return output_; }

 protected:
  int input_;
  int output_;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_CAST_TEST_COMMON_H_
