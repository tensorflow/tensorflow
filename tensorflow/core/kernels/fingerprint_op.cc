/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <string>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/fingerprint.h"

namespace tensorflow {
namespace {
template <typename T>
inline void CopyToBuffer(const T& value, uint8* output) {
  // Memcpy to string is endian-dependent. We choose little-endian as
  // standard. On big-endian machines, bytes should be reversed.
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  static_assert(port::kLittleEndian, "");
  std::memcpy(output, &value, sizeof(value));
#else
  static_assert(!port::kLittleEndian, "");
  std::reverse_copy(reinterpret_cast<const uint8*>(&value),
                    reinterpret_cast<const uint8*>(&value + 1), output);
#endif
}

void FarmhashFingerprint64(TTypes<uint8, 2>::ConstTensor input,
                           TTypes<uint8, 2>::Matrix output) {
  DCHECK_EQ(output.dimension(0), input.dimension(0));
  DCHECK_EQ(output.dimension(1), sizeof(uint64));
  for (int64_t i = 0; i < output.dimension(0); ++i) {
    const uint64 fingerprint =
        Fingerprint64({reinterpret_cast<const char*>(&input(i, 0)),
                       static_cast<std::size_t>(input.dimension(1))});
    CopyToBuffer(fingerprint, &output(i, 0));
  }
}

void FarmhashFingerprint64(TTypes<tstring>::ConstFlat input,
                           TTypes<uint8, 2>::Matrix output) {
  DCHECK_EQ(output.dimension(0), input.dimension(0));
  DCHECK_EQ(output.dimension(1), sizeof(uint64));
  for (int64_t i = 0; i < input.dimension(0); ++i) {
    const uint64 fingerprint =
        Fingerprint64({input(i).data(), input(i).size()});
    CopyToBuffer(fingerprint, &output(i, 0));
  }
}

class FingerprintOp : public OpKernel {
 public:
  explicit FingerprintOp(OpKernelConstruction* context) : OpKernel(context) {
    DataType dtype;
    OP_REQUIRES_OK(context, context->GetAttr("T", &dtype));
    OP_REQUIRES(context, DataTypeCanUseMemcpy(dtype) || dtype == DT_STRING,
                errors::InvalidArgument("Data type not supported: ",
                                        DataTypeString(dtype)));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    const Tensor& method_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(method_tensor.shape()),
                errors::InvalidArgument("`method` should be a scalar string: ",
                                        method_tensor.shape()));
    // For now, farmhash64 is the only function supported.
    const tstring& method = method_tensor.scalar<tstring>()();
    OP_REQUIRES(
        context, method == "farmhash64",
        errors::InvalidArgument("Unsupported fingerprint method: ", method));

    const Tensor& input = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVectorOrHigher(input.shape()),
        errors::InvalidArgument("`data` should have at least one dimension: ",
                                input.shape()));

    const int64_t dim0 = input.shape().dim_size(0);
    int64_t dim1;
    if (dim0 == 0) {
      dim1 = 0;
    } else {
      dim1 = input.shape().num_elements() / dim0;
    }

    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape{dim0, kFingerprintSize}, &output));

    if (input.dtype() == DT_STRING) {
      if (dim1 > 1) {
        Tensor temp;
        OP_REQUIRES_OK(context, context->allocate_temp(
                                    DT_UINT8,
                                    TensorShape{input.shape().num_elements(),
                                                kFingerprintSize},
                                    &temp));
        // `temp` is a matrix of shape {input.num_elements, fingerprint_size},
        // and each row contains the fingerprint value of corresponding string.
        // To compute fingerprints of multiple strings, this op fingerprints the
        // buffer containing the string fingerprints.
        FarmhashFingerprint64(input.flat<tstring>(), temp.tensor<uint8, 2>());
        FarmhashFingerprint64(static_cast<const Tensor&>(temp).shaped<uint8, 2>(
                                  {dim0, dim1 * kFingerprintSize}),
                              output->matrix<uint8>());
      } else {
        // In case dim1 == 1, each string computes into its own fingerprint
        // value. There is no need to fingerprint twice.
        FarmhashFingerprint64(input.flat<tstring>(), output->matrix<uint8>());
      }
    } else {
      auto data = input.bit_casted_shaped<uint8, 2>(
          {dim0, dim1 * DataTypeSize(input.dtype())});
      FarmhashFingerprint64(data, output->matrix<uint8>());
    }
  }

 private:
  static constexpr int kFingerprintSize = sizeof(uint64);
};

REGISTER_KERNEL_BUILDER(Name("Fingerprint").Device(tensorflow::DEVICE_CPU),
                        FingerprintOp);
}  // namespace
}  // namespace tensorflow
