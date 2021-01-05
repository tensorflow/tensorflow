/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL_H_

#ifdef INTEL_MKL

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/quantization_utils.h"

namespace tensorflow {
namespace {

// Helper class for converting MKL tensors to TF tensors
class ConvMklToTF : public OpsTestBase {
 public:
  template <typename T>
  void ConvertMKL2TF(DataType dtype, const Tensor& first, const Tensor& second,
                     Tensor& output) {
    // Create an MKL to TF conversion node and execute it
    TF_EXPECT_OK(NodeDefBuilder("mkl_to_tf_op", "_MklToTf")
                     .Input(FakeInput(dtype))     // Input
                     .Input(FakeInput(DT_UINT8))  // MKL second tensor
                     .Attr("T", dtype)
                     .Attr("_kernel", "MklLayoutDependentOp")
                     .Finalize(node_def()));
    TF_EXPECT_OK(InitOp());
    AddInputFromArray<T>(first.shape(), first.flat<T>());
    AddInputFromArray<uint8>(second.shape(), second.flat<uint8>());
    TF_ASSERT_OK(RunOpKernel());

    output = *GetOutput(0);
  }
  void TestBody() {}
};

// Helper function that converts tensor from MKL format to TF format.
template <typename T>
inline Tensor GetTFFormatTensor(DataType dtype,
                                const Tensor& mkl_quantized_tensor,
                                const Tensor* mkl_metadata_tensor_ptr) {
  DCHECK(mkl_metadata_tensor_ptr);
  Tensor converted_tensor;
  ConvMklToTF conv_comp;
  conv_comp.ConvertMKL2TF<T>(dtype, mkl_quantized_tensor,
                             *mkl_metadata_tensor_ptr, converted_tensor);
  return converted_tensor;
}

// Helper function that converts quantized tensor to float tensor and returns it
// in TF native format.
template <typename T>
inline Tensor QuantizedToFloatTFFormat(DataType dtype,
                                       const Tensor& mkl_quantized_tensor,
                                       const Tensor* mkl_metadata_tensor,
                                       const float output_min,
                                       const float output_max) {
  Tensor converted_tensor;
  if (!NativeFormatEnabled()) {
    converted_tensor =
        GetTFFormatTensor<T>(dtype, mkl_quantized_tensor, mkl_metadata_tensor);
  }
  return QuantizedTensorToFloat<T>(
      NativeFormatEnabled() ? mkl_quantized_tensor : converted_tensor,
      output_min, output_max);
}

}  // namespace
}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL_H_
