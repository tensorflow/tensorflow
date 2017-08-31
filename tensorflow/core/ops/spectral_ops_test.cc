/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {

TEST(MathOpsTest, FFT_ShapeFn) {
  for (const auto* op_name : {"FFT", "IFFT"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "?");
    INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[]");
    INFER_OK(op, "[?]", "in0");
    INFER_OK(op, "[1]", "in0");
    INFER_OK(op, "[1,2,3,4,5,6,7]", "in0");
  }

  for (const auto* op_name : {"FFT2D", "IFFT2D"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "?");
    INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
    INFER_OK(op, "[?,1]", "in0");
    INFER_OK(op, "[1,2]", "in0");
    INFER_OK(op, "[1,2,3,4,5,6,7]", "in0");
  }

  for (const auto* op_name : {"FFT3D", "IFFT3D"}) {
    ShapeInferenceTestOp op(op_name);
    INFER_OK(op, "?", "?");
    INFER_ERROR("Shape must be at least rank 3 but is rank 2", op, "[1,2]");
    INFER_OK(op, "[?,1,?]", "in0");
    INFER_OK(op, "[1,2,3]", "in0");
    INFER_OK(op, "[1,2,3,4,5,6,7]", "in0");
  }
}

TEST(MathOpsTest, RFFT_ShapeFn) {
  // Rank 1
  for (const bool forward : {true, false}) {
    ShapeInferenceTestOp op(forward ? "RFFT" : "IRFFT");

    // Unknown rank or shape of inputs.
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "?;[1]", "?");

    // Unknown fft_length (whether or not rank/shape is known) implies unknown
    // FFT shape.
    INFER_OK(op, "[1];?", "[?]");
    INFER_OK(op, "[1];[1]", "[?]");
    INFER_OK(op, "[?];[1]", "[?]");

    // Batch dimensions preserved.
    INFER_OK(op, "[1,2,3,4];[1]", "[d0_0,d0_1,d0_2,?]");

    INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "[];?");
    INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1];[1,1]");
    INFER_ERROR("Dimension must be 1 but is 2", op, "[1];[2]");

    // Tests with known values for fft_length input.
    op.input_tensors.resize(2);
    Tensor fft_length = test::AsTensor<int32>({10});
    op.input_tensors[1] = &fft_length;

    // The inner-most dimension of the RFFT is n/2+1 while for IRFFT it's n.
    if (forward) {
      INFER_OK(op, "[?];[1]", "[6]");
      INFER_OK(op, "[1];[1]", "[6]");
      INFER_OK(op, "[1,1];[1]", "[d0_0,6]");
    } else {
      INFER_OK(op, "[?];[1]", "[10]");
      INFER_OK(op, "[1];[1]", "[10]");
      INFER_OK(op, "[1,1];[1]", "[d0_0,10]");
    }

    fft_length = test::AsTensor<int32>({11});
    if (forward) {
      INFER_OK(op, "[?];[1]", "[6]");
      INFER_OK(op, "[1];[1]", "[6]");
      INFER_OK(op, "[1,1];[1]", "[d0_0,6]");
    } else {
      INFER_OK(op, "[?];[1]", "[11]");
      INFER_OK(op, "[1];[1]", "[11]");
      INFER_OK(op, "[1,1];[1]", "[d0_0,11]");
    }

    fft_length = test::AsTensor<int32>({12});
    if (forward) {
      INFER_OK(op, "[?];[1]", "[7]");
      INFER_OK(op, "[1];[1]", "[7]");
      INFER_OK(op, "[1,1];[1]", "[d0_0,7]");
    } else {
      INFER_OK(op, "[?];[1]", "[12]");
      INFER_OK(op, "[1];[1]", "[12]");
      INFER_OK(op, "[1,1];[1]", "[d0_0,12]");
    }
  }

  // Rank 2
  for (const bool forward : {true, false}) {
    ShapeInferenceTestOp op(forward ? "RFFT2D" : "IRFFT2D");

    // Unknown rank or shape of inputs.
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "?;[2]", "?");

    // Unknown fft_length (whether or not rank/shape is known) implies unknown
    // FFT shape.
    INFER_OK(op, "[1,1];?", "[?,?]");
    INFER_OK(op, "[1,1];[2]", "[?,?]");
    INFER_OK(op, "[?,?];[2]", "[?,?]");

    // Batch dimensions preserved.
    INFER_OK(op, "[1,2,3,4];[2]", "[d0_0,d0_1,?,?]");

    INFER_ERROR("Shape must be at least rank 2 but is rank 0", op, "[];?");
    INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1,1];[1,1]");
    INFER_ERROR("Dimension must be 2 but is 3", op, "[1,1];[3]");

    // Tests with known values for fft_length input.
    op.input_tensors.resize(2);
    Tensor fft_length = test::AsTensor<int32>({9, 10});
    op.input_tensors[1] = &fft_length;

    // The inner-most dimension of the RFFT is n/2+1 while for IRFFT it's n.
    if (forward) {
      INFER_OK(op, "[?,?];[2]", "[9,6]");
      INFER_OK(op, "[1,1];[2]", "[9,6]");
      INFER_OK(op, "[1,1,1];[2]", "[d0_0,9,6]");
    } else {
      INFER_OK(op, "[?,?];[2]", "[9,10]");
      INFER_OK(op, "[1,1];[2]", "[9,10]");
      INFER_OK(op, "[1,1,1];[2]", "[d0_0,9,10]");
    }

    fft_length = test::AsTensor<int32>({10, 11});
    if (forward) {
      INFER_OK(op, "[?,?];[2]", "[10,6]");
      INFER_OK(op, "[1,1];[2]", "[10,6]");
      INFER_OK(op, "[1,1,1];[2]", "[d0_0,10,6]");
    } else {
      INFER_OK(op, "[?,?];[2]", "[10,11]");
      INFER_OK(op, "[1,1];[2]", "[10,11]");
      INFER_OK(op, "[1,1,1];[2]", "[d0_0,10,11]");
    }

    fft_length = test::AsTensor<int32>({11, 12});
    if (forward) {
      INFER_OK(op, "[?,?];[2]", "[11,7]");
      INFER_OK(op, "[1,1];[2]", "[11,7]");
      INFER_OK(op, "[1,1,1];[2]", "[d0_0,11,7]");
    } else {
      INFER_OK(op, "[?,?];[2]", "[11,12]");
      INFER_OK(op, "[1,1];[2]", "[11,12]");
      INFER_OK(op, "[1,1,1];[2]", "[d0_0,11,12]");
    }
  }

  // Rank 3
  for (const bool forward : {true, false}) {
    ShapeInferenceTestOp op(forward ? "RFFT3D" : "IRFFT3D");

    // Unknown rank or shape of inputs.
    INFER_OK(op, "?;?", "?");
    INFER_OK(op, "?;[3]", "?");

    // Unknown fft_length (whether or not rank/shape is known) implies unknown
    // FFT shape.
    INFER_OK(op, "[1,1,1];?", "[?,?,?]");
    INFER_OK(op, "[1,1,1];[3]", "[?,?,?]");
    INFER_OK(op, "[?,?,?];[3]", "[?,?,?]");

    // Batch dimensions preserved.
    INFER_OK(op, "[1,2,3,4];[3]", "[d0_0,?,?,?]");

    INFER_ERROR("Shape must be at least rank 3 but is rank 0", op, "[];?");
    INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1,1,1];[1,1]");
    INFER_ERROR("Dimension must be 3 but is 4", op, "[1,1,1];[4]");

    // Tests with known values for fft_length input.
    op.input_tensors.resize(2);
    Tensor fft_length = test::AsTensor<int32>({10, 11, 12});
    op.input_tensors[1] = &fft_length;

    // The inner-most dimension of the RFFT is n/2+1 while for IRFFT it's n.
    if (forward) {
      INFER_OK(op, "[?,?,?];[3]", "[10,11,7]");
      INFER_OK(op, "[1,1,1];[3]", "[10,11,7]");
      INFER_OK(op, "[1,1,1,1];[3]", "[d0_0,10,11,7]");
    } else {
      INFER_OK(op, "[?,?,?];[3]", "[10,11,12]");
      INFER_OK(op, "[1,1,1];[3]", "[10,11,12]");
      INFER_OK(op, "[1,1,1,1];[3]", "[d0_0,10,11,12]");
    }

    fft_length = test::AsTensor<int32>({11, 12, 13});
    if (forward) {
      INFER_OK(op, "[?,?,?];[3]", "[11,12,7]");
      INFER_OK(op, "[1,1,1];[3]", "[11,12,7]");
      INFER_OK(op, "[1,1,1,1];[3]", "[d0_0,11,12,7]");
    } else {
      INFER_OK(op, "[?,?,?];[3]", "[11,12,13]");
      INFER_OK(op, "[1,1,1];[3]", "[11,12,13]");
      INFER_OK(op, "[1,1,1,1];[3]", "[d0_0,11,12,13]");
    }

    fft_length = test::AsTensor<int32>({12, 13, 14});
    if (forward) {
      INFER_OK(op, "[?,?,?];[3]", "[12,13,8]");
      INFER_OK(op, "[1,1,1];[3]", "[12,13,8]");
      INFER_OK(op, "[1,1,1,1];[3]", "[d0_0,12,13,8]");
    } else {
      INFER_OK(op, "[?,?,?];[3]", "[12,13,14]");
      INFER_OK(op, "[1,1,1];[3]", "[12,13,14]");
      INFER_OK(op, "[1,1,1,1];[3]", "[d0_0,12,13,14]");
    }
  }
}

}  // end namespace tensorflow
