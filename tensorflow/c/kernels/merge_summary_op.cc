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
#include <memory>
#include <sstream>
#include <unordered_set>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/selective_registration.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"

namespace {

// Operators used to create a std::unique_ptr for TF_Tensor and TF_Status
struct TFTensorDeleter {
  void operator()(TF_Tensor* tf_tensor) const { TF_DeleteTensor(tf_tensor); }
};

struct TFStatusDeleter {
  void operator()(TF_Status* tf_status) const { TF_DeleteStatus(tf_status); }
};

// Struct that wraps TF_Tensor and TF_Status to delete once out of scope
using Safe_TF_TensorPtr = std::unique_ptr<TF_Tensor, TFTensorDeleter>;
using Safe_TF_StatusPtr = std::unique_ptr<TF_Status, TFStatusDeleter>;

// dummy functions used for kernel registration
void* MergeSummaryOp_Create(TF_OpKernelConstruction* ctx) { return nullptr; }

void MergeSummaryOp_Delete(void* kernel) {}

void MergeSummaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  tensorflow::Summary s;
  std::unordered_set<tensorflow::string> tags;
  Safe_TF_StatusPtr status(TF_NewStatus());
  for (int input_num = 0; input_num < TF_NumInputs(ctx); ++input_num) {
    TF_Tensor* input;
    TF_GetInput(ctx, input_num, &input, status.get());
    Safe_TF_TensorPtr safe_input_ptr(input);
    if (TF_GetCode(status.get()) != TF_OK) {
      TF_OpKernelContext_Failure(ctx, status.get());
      return;
    }
    auto tags_array =
        static_cast<tensorflow::tstring*>(TF_TensorData(safe_input_ptr.get()));
    for (int i = 0; i < TF_TensorElementCount(safe_input_ptr.get()); ++i) {
      const tensorflow::tstring& s_in = tags_array[i];
      tensorflow::Summary summary_in;
      if (!tensorflow::ParseProtoUnlimited(&summary_in, s_in)) {
        TF_SetStatus(status.get(), TF_INVALID_ARGUMENT,
                     "Could not parse one of the summary inputs");
        TF_OpKernelContext_Failure(ctx, status.get());
        return;
      }
      for (int v = 0; v < summary_in.value_size(); ++v) {
        // This tag is unused by the TensorSummary op, so no need to check for
        // duplicates.
        const tensorflow::string& tag = summary_in.value(v).tag();
        if ((!tag.empty()) && !tags.insert(tag).second) {
          std::ostringstream err;
          err << "Duplicate tag " << tag << " found in summary inputs ";
          TF_SetStatus(status.get(), TF_INVALID_ARGUMENT, err.str().c_str());
          TF_OpKernelContext_Failure(ctx, status.get());
          return;
        }
        *s.add_value() = summary_in.value(v);
      }
    }
  }
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

void RegisterMergeSummaryOpKernel() {
  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder(
        "MergeSummary", tensorflow::DEVICE_CPU, &MergeSummaryOp_Create,
        &MergeSummaryOp_Compute, &MergeSummaryOp_Delete);
    TF_RegisterKernelBuilder("MergeSummary", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Merge Summmary kernel";
  }
  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the Histogram Summary kernel.
TF_ATTRIBUTE_UNUSED static bool IsMergeSummaryOpKernelRegistered = []() {
  if (SHOULD_REGISTER_OP_KERNEL("MergeSummary")) {
    RegisterMergeSummaryOpKernel();
  }
  return true;
}();

}  // namespace
