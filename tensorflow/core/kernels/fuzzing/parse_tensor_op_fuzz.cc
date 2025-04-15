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

#include "absl/log/log.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/parsing_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/fuzzing/fuzz_session.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace fuzzing {

// Fuzz inputs to the serialized Tensor decoder.

class FuzzParseTensor : public FuzzSession {
  void BuildGraph(const Scope& scope) final {
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)
    // The serialized proto.
    auto input = Placeholder(scope.WithOpName("input"), DT_STRING);

    (void)ParseTensor(scope.WithOpName("output"), input, DT_FLOAT);
  }

  void FuzzImpl(const uint8_t* data, size_t size) final {
    // We need to be sure that we don't request too many elements (i.e., we
    // don't make ASAN OOM). In theory, a tensor shape can have arbitrary large
    // number of elements, up to the limit of the memory available to the OS.
    // However, due to the tracing done in ASAN, after 2^32 bytes of requested
    // memory we would get a crash in the fuzzer (see b/34190148). Hence, let's
    // try parsing the proto here, check that the size (if valid) is below a
    // maximum threshold (using 2^20 for convenience), and then run the
    // remainder of the fuzzer testing. Of course, this duplicates some work
    // but it's better than repeating the investigation whenever Autofuzz
    // detects another similar OOM.
    // After adding `-fsanitize=null` to ASAN (cl/317376103), the memory
    // footprint increased, so we lower the maximum threshold to 2^18.
    string as_string = string(reinterpret_cast<const char*>(data), size);
    TensorProto proto;
    if (!ParseProtoUnlimited(&proto, as_string)) {
      LOG(WARNING) << "Unable to parse proto of tensor\n";
      return;
    }
    if (!TensorShape::IsValid(proto.tensor_shape())) {
      LOG(WARNING) << "Invalid tensor shape\n";
      return;
    }
    TensorShape shape(proto.tensor_shape());
    const int64_t num_elements = shape.num_elements();
    const int64_t max_num_elements = 1 << 18;
    if (num_elements > max_num_elements) {
      LOG(WARNING) << "Requiring a tensor with too many elements\n";
      return;
    }

    // Now we can do the actual fuzz implementation
    Tensor input_tensor(tensorflow::DT_STRING, TensorShape({}));
    input_tensor.scalar<tstring>()() = as_string;
    RunInputs({{"input", input_tensor}});
  }
};

STANDARD_TF_FUZZ_FUNCTION(FuzzParseTensor);

}  // end namespace fuzzing
}  // end namespace tensorflow
