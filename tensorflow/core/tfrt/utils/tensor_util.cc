/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/utils/tensor_util.h"

#include <assert.h>
#include <sys/types.h>

#include <cstring>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tensorflow/core/runtime_fallback/util/type_util.h"
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/dtype/dtype.h"  // from @tf_runtime
#include "tfrt/host_context/host_buffer.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/forward_decls.h"  // from @tf_runtime
#include "tfrt/tensor/dense_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/string_host_tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor.h"  // from @tf_runtime
#include "tfrt/tensor/tensor_shape.h"  // from @tf_runtime

namespace tfrt {
namespace {

using ::tensorflow::StatusOr;

llvm::Expected<tensorflow::Tensor> CopyScalarHostTensorToTFTensor(
    const AnyScalarHostTensor& tensor) {
  auto element_byte_size = GetHostSize(tensor.dtype());
  if (element_byte_size == 0) {
    return MakeStringError(
        "Failed to convert ScalarHostTensor to tensorflow::Tensor: "
        "unsupported dtype: ",
        tensor.dtype());
  }

  llvm::SmallVector<Index, 4> dims;
  tensor.shape().GetDimensions(&dims);

  auto tf_dtype = tensorflow::tfd::GetTfDataType(tensor.dtype());
  tensorflow::Tensor tf_tensor(
      tf_dtype, tensorflow::TensorShape(
                    llvm::SmallVector<int64_t, 4>(dims.begin(), dims.end())));

  // This can be a DCHECK instead of returninng an error because TFRT's
  // ScalarHostTensor only supports these types.
  DCHECK(DataTypeCanUseMemcpy(tf_dtype));

  // TODO(tfrt-devs): Hide the following the logic of generating a full size
  // buffer for the scalar host tensor under Tensor (and optimize if
  // necessary), so we don't have to re-implement it every time we need it.
  char* begin = reinterpret_cast<char*>(tf_tensor.data());
  for (int i = 0; i < tf_tensor.NumElements(); ++i) {
    std::memcpy(begin, tensor.data(), element_byte_size);
    begin += element_byte_size;
  }
  return tf_tensor;
}

StatusOr<DType> ConvertTFDTypeToTFRTDType(tensorflow::DataType dtype) {
  switch (dtype) {
#define DTYPE(TFRT_DTYPE, TF_DTYPE) \
  case tensorflow::TF_DTYPE:        \
    return DType(DType::TFRT_DTYPE);
#include "tensorflow/core/tfrt/utils/dtype.def"
    default:
      return tensorflow::errors::Internal(absl::StrCat(
          "unsupported tensorflow dtype: ", tensorflow::DataType_Name(dtype)));
  }
}

StatusOr<tensorflow::DataType> ConvertTFRTDTypeToTFDType(DType dtype) {
  switch (dtype) {
#define DTYPE(TFRT_DTYPE, TF_DTYPE) \
  case DType::TFRT_DTYPE:           \
    return tensorflow::TF_DTYPE;
#include "tensorflow/core/tfrt/utils/dtype.def"
    default:
      return tensorflow::errors::Internal(
          StrCat("unsupported tfrt dtype: ", dtype));
  }
}

}  // namespace

llvm::Expected<tensorflow::Tensor> TFRTTensorToTFTensor(const Tensor& tensor,
                                                        HostContext* host) {
  if (auto* knfbt = llvm::dyn_cast<tensorflow::KernelFallbackTensor>(&tensor)) {
    return *knfbt->GetTensor();
  }
  // TODO(tfrt-devs): The following logic should be better provided by
  // Tensor so we don't have to re-implement it.
  if (auto* dht = llvm::dyn_cast<DenseHostTensor>(&tensor)) {
    return tensorflow::tfd::MoveHostBufferToTfTensor(
        dht->buffer(), dht->dtype(), dht->shape());
  }
  if (auto* sht = llvm::dyn_cast<StringHostTensor>(&tensor)) {
    return tensorflow::tfd::CopyShtToTfTensor(*sht);
  }
  if (auto* scalar = llvm::dyn_cast<AnyScalarHostTensor>(&tensor)) {
    return CopyScalarHostTensorToTFTensor(*scalar);
  }
  return MakeStringError("Unsupported conversion format for ",
                         tensor.tensor_type().name());
}

AsyncValueRef<TensorHandle> TFTensorToTFRTTensorHandle(
    const tensorflow::Tensor& tf_tensor, HostContext* host_ctx) {
  auto knfbt =
      MakeAvailableAsyncValueRef<tensorflow::KernelFallbackTensor>(tf_tensor);
  return MakeAvailableAsyncValueRef<TensorHandle>(
      host_ctx->GetHostDeviceRef(), knfbt->metadata(), std::move(knfbt));
}

StatusOr<TensorHandle> CreateTensorHandleFromTFTensor(
    const tensorflow::Tensor& tensor, HostContext* host) {
  // TODO(chky): Handle non-trivial types such as strings.
  TF_ASSIGN_OR_RETURN(auto dtype, ConvertTFDTypeToTFRTDType(tensor.dtype()));
  auto shape = tensor.shape().dim_sizes();
  TensorMetadata metadata(dtype, TensorShape(llvm::SmallVector<Index, 4>(
                                     shape.begin(), shape.end())));

  if (dtype == DType::String) {
    auto sht_ref =
        StringHostTensor::MakeConstructedAsyncValueRef(metadata, host);
    auto to = sht_ref->strings();
    auto from = tensor.flat<tensorflow::tstring>();
    for (int i = 0, e = to.size(); i < e; ++i) {
      to[i] = from(i);
    }
    sht_ref.SetStateConcrete();
    return TensorHandle(host->GetHostDeviceRef(), metadata, std::move(sht_ref));
  }

  auto dht_ref = DenseHostTensor::MakeConstructedAsyncValueRef(metadata, host);

  auto& dht = dht_ref.get();
  assert(dht.DataSizeInBytes() ==
         tensor.NumElements() * tensorflow::DataTypeSize(tensor.dtype()));
  std::memcpy(dht_ref.get().data(), tensor.data(), dht.DataSizeInBytes());

  dht_ref.SetStateConcrete();
  return TensorHandle(host->GetHostDeviceRef(), metadata, std::move(dht_ref));
}

StatusOr<tensorflow::Tensor> CreateTFTensorFromTensorHandle(
    const TensorHandle& tensor_handle) {
  const auto& metadata = tensor_handle.GetAvailableMetadata();
  TF_ASSIGN_OR_RETURN(auto dtype, ConvertTFRTDTypeToTFDType(metadata.dtype));
  llvm::SmallVector<Index, 4> shape;
  metadata.shape.GetDimensions(&shape);
  const auto& host_tensor = tensor_handle.GetAsyncTensor()->get<HostTensor>();

  if (auto* kernel_fallback_tensor =
          llvm::dyn_cast<tensorflow::KernelFallbackTensor>(&host_tensor)) {
    return *kernel_fallback_tensor->GetTensor();
  }

  if (llvm::isa<StringHostTensor>(host_tensor)) {
    assert(dtype == tensorflow::DT_STRING);
    const auto& sht = llvm::cast<StringHostTensor>(host_tensor);
    tensorflow::Tensor tensor(
        tensorflow::DT_STRING,
        tensorflow::TensorShape(
            llvm::SmallVector<int64_t, 4>(shape.begin(), shape.end())));
    auto from = sht.strings();
    auto to = tensor.flat<tensorflow::tstring>();
    for (int i = 0, e = from.size(); i < e; ++i) {
      to(i).assign(from[i].data(), from[i].size());
    }
    return tensor;
  }

  if (llvm::isa<DenseHostTensor>(host_tensor)) {
    const auto& dht = llvm::cast<DenseHostTensor>(host_tensor);
    tensorflow::Tensor tensor(
        dtype, tensorflow::TensorShape(
                   llvm::SmallVector<int64_t, 4>(shape.begin(), shape.end())));

    assert(dht.DataSizeInBytes() ==
           tensor.NumElements() * tensorflow::DataTypeSize(tensor.dtype()));
    std::memcpy(tensor.data(), dht.data(), dht.DataSizeInBytes());
    return tensor;
  }

  return tensorflow::errors::Internal("unknown host tensor type");
}

}  // namespace tfrt
