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

#include <sstream>
#include <string>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace {

// Operators used to create a std::unique_ptr for TF_Tensor and TF_Status.
struct TFTensorDeleter {
  void operator()(TF_Tensor* tf_tensor) const { TF_DeleteTensor(tf_tensor); }
};

struct TFStatusDeleter {
  void operator()(TF_Status* tf_status) const { TF_DeleteStatus(tf_status); }
};

// Struct that wraps TF_Tensor and TF_Status to delete once out of scope.
using Safe_TF_TensorPtr = std::unique_ptr<TF_Tensor, TFTensorDeleter>;
using Safe_TF_StatusPtr = std::unique_ptr<TF_Status, TFStatusDeleter>;

// Used to pass the operation node name from kernel construction to
// kernel computation.
struct HistogramSummaryOp {
  std::string op_node_name;
};

void* HistogramSummaryOp_Create(TF_OpKernelConstruction* ctx) {
  HistogramSummaryOp* kernel = new HistogramSummaryOp;
  TF_StringView string_view_name = TF_OpKernelConstruction_GetName(ctx);
  kernel->op_node_name =
      std::string(string_view_name.data, string_view_name.len);
  return kernel;
}

void HistogramSummaryOp_Delete(void* kernel) {
  delete static_cast<HistogramSummaryOp*>(kernel);
}

template <typename T>
void HistogramSummaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  HistogramSummaryOp* k = static_cast<HistogramSummaryOp*>(kernel);
  TF_Tensor* tags;
  TF_Tensor* values;
  Safe_TF_StatusPtr status(TF_NewStatus());
  TF_GetInput(ctx, 0, &tags, status.get());
  Safe_TF_TensorPtr safe_tags_ptr(tags);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  TF_GetInput(ctx, 1, &values, status.get());
  Safe_TF_TensorPtr safe_values_ptr(values);
  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  if (TF_NumDims(safe_tags_ptr.get()) != 0) {
    TF_SetStatus(status.get(), TF_INVALID_ARGUMENT, "tags must be scalar");
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  // Cast values to array to access tensor elements by index
  auto values_array = static_cast<T*>(TF_TensorData(safe_values_ptr.get()));
  tensorflow::histogram::Histogram histo;
  for (int64_t i = 0; i < TF_TensorElementCount(safe_values_ptr.get()); ++i) {
    const double double_val = static_cast<double>(values_array[i]);
    if (Eigen::numext::isnan(double_val)) {
      std::ostringstream err;
      err << "Nan in summary histogram for: " << k->op_node_name;
      TF_SetStatus(status.get(), TF_INVALID_ARGUMENT, err.str().c_str());
      TF_OpKernelContext_Failure(ctx, status.get());
      return;
    } else if (Eigen::numext::isinf(double_val)) {
      std::ostringstream err;
      err << "Infinity in Histogram for: " << k->op_node_name;
      TF_SetStatus(status.get(), TF_INVALID_ARGUMENT, err.str().c_str());
      TF_OpKernelContext_Failure(ctx, status.get());
      return;
    }
    histo.Add(double_val);
  }
  tensorflow::Summary s;
  tensorflow::Summary::Value* v = s.add_value();
  const tensorflow::tstring& tag =
      *(static_cast<tensorflow::tstring*>(TF_TensorData(safe_tags_ptr.get())));
  v->set_tag(tag.data(), tag.size());
  histo.EncodeToProto(v->mutable_histo(), false /* Drop zero buckets */);

  Safe_TF_TensorPtr summary_tensor(TF_AllocateOutput(
      /*context=*/ctx, /*index=*/0, /*dtype=*/TF_ExpectedOutputDataType(ctx, 0),
      /*dims=*/nullptr, /*num_dims=*/0,
      /*len=*/sizeof(tensorflow::tstring), status.get()));

  if (TF_GetCode(status.get()) != TF_OK) {
    TF_OpKernelContext_Failure(ctx, status.get());
    return;
  }
  tensorflow::tstring* output_tstring = reinterpret_cast<tensorflow::tstring*>(
      TF_TensorData(summary_tensor.get()));
  CHECK(SerializeToTString(s, output_tstring));
}

template <typename T>
void RegisterHistogramSummaryOpKernel() {
  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder(
        "HistogramSummary", tensorflow::DEVICE_CPU, &HistogramSummaryOp_Create,
        &HistogramSummaryOp_Compute<T>, &HistogramSummaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(
        builder, "T",
        static_cast<TF_DataType>(tensorflow::DataTypeToEnum<T>::v()), status);
    CHECK_EQ(TF_OK, TF_GetCode(status)) << "Error while adding type constraint";
    TF_RegisterKernelBuilder("HistogramSummary", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Histogram Summmary kernel";
  }
  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the Histogram Summary kernel.
TF_ATTRIBUTE_UNUSED static bool IsHistogramSummaryOpKernelRegistered = []() {
  if (SHOULD_REGISTER_OP_KERNEL("HistogramSummary")) {
    RegisterHistogramSummaryOpKernel<int64_t>();
    RegisterHistogramSummaryOpKernel<tensorflow::uint64>();
    RegisterHistogramSummaryOpKernel<tensorflow::int32>();
    RegisterHistogramSummaryOpKernel<tensorflow::uint32>();
    RegisterHistogramSummaryOpKernel<tensorflow::uint16>();
    RegisterHistogramSummaryOpKernel<tensorflow::int16>();
    RegisterHistogramSummaryOpKernel<tensorflow::int8>();
    RegisterHistogramSummaryOpKernel<tensorflow::uint8>();
    RegisterHistogramSummaryOpKernel<Eigen::half>();
    RegisterHistogramSummaryOpKernel<tensorflow::bfloat16>();
    RegisterHistogramSummaryOpKernel<float>();
    RegisterHistogramSummaryOpKernel<double>();
  }
  return true;
}();
}  // namespace
