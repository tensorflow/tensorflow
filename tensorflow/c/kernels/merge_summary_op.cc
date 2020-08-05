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
#include <unordered_set>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/selective_registration.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/default/logging.h"

namespace { 

// Struct that stores the status and TF_Tensor inputs to the opkernel. 
// Used to delete tensor and status in its destructor upon kernel return. 
struct TensorWrapper { 
  TF_Tensor* t; 
  TensorWrapper() : t(nullptr) {}
  ~TensorWrapper() { 
    TF_DeleteTensor(t);
  }
};

struct StatusWrapper { 
  TF_Status* s; 
  StatusWrapper() { 
    s = TF_NewStatus(); 
  }
  ~StatusWrapper() { 
    TF_DeleteStatus(s);
  }
};

// dummy functions used for kernel registration 
void* MergeSummaryOp_Create(TF_OpKernelConstruction* ctx) {}

void MergeSummaryOp_Delete(void* kernel) {
  return;
}

void MergeSummaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  tensorflow::Summary s; 
  std::unordered_set<tensorflow::string> tags; 
  StatusWrapper status_wrapper;
  for (int input_num = 0; input_num < TF_NumInputs(ctx); ++input_num) { 
    TensorWrapper input_wrapper; 
    TF_GetInput(ctx, input_num, &input_wrapper.t, status_wrapper.s); 
    if (TF_GetCode(status_wrapper.s) != TF_OK) {  
      TF_OpKernelContext_Failure(ctx, status_wrapper.s); 
      return;
    }

    auto tags_array = static_cast<tensorflow::tstring*>(
    		TF_TensorData(input_wrapper.t));
    for (int i = 0; i < TF_TensorElementCount(input_wrapper.t); ++i) { 
      const tensorflow::tstring& s_in = tags_array[i]; 
      tensorflow::Summary summary_in; 
      if (!tensorflow::ParseProtoUnlimited(&summary_in, s_in)) { 
        TF_SetStatus(status_wrapper.s, TF_INVALID_ARGUMENT, 
            "Could not parse one of the summary inputs");
        TF_OpKernelContext_Failure(ctx, status_wrapper.s);
        return; 
      }
      for (int v = 0; v < summary_in.value_size(); ++v) { 
        // This tag is unused by the TensorSummary op, so no need to check for 
        // duplicates.
        const tensorflow::string& tag = summary_in.value(v).tag(); 
        if ((!tag.empty()) && !tags.insert(tag).second) { 
          std::ostringstream err;
          err << "Duplicate tag " << tag << " found in summary inputs ";
          TF_SetStatus(status_wrapper.s, TF_INVALID_ARGUMENT, err.str().c_str());
          TF_OpKernelContext_Failure(ctx, status_wrapper.s);
          return; 
        }
        *s.add_value() = summary_in.value(v); 
      }
    }
  }
  TensorWrapper summary_tensor_wrapper; 
  summary_tensor_wrapper.t = TF_AllocateOutput(
      /*context=*/ctx, /*index=*/0, /*dtype=*/TF_ExpectedOutputDataType(
      ctx, 0), /*dims=*/nullptr, /*num_dims=*/0, 
      /*len=*/sizeof(tensorflow::tstring), status_wrapper.s);
  if (TF_GetCode(status_wrapper.s) != TF_OK){ 
    TF_OpKernelContext_Failure(ctx, status_wrapper.s);
    return; 
  }
  tensorflow::tstring* output_tstring = reinterpret_cast<tensorflow::tstring*>(
      TF_TensorData(summary_tensor_wrapper.t)); 
  CHECK(SerializeToTString(s, output_tstring));
}

void RegisterMergeSummaryOpKernel() {
  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder("MergeSummary", 
                                        tensorflow::DEVICE_CPU,
                                        &MergeSummaryOp_Create, 
                                        &MergeSummaryOp_Compute,
                                        &MergeSummaryOp_Delete);
    TF_RegisterKernelBuilder("MergeSummary", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Merge Summmary kernel";
  }
  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the Histogram Summary kernel.                                                          
TF_ATTRIBUTE_UNUSED static bool  IsMergeSummaryOpKernelRegistered = []() {                  
  if (SHOULD_REGISTER_OP_KERNEL("MergeSummary")) {
    RegisterMergeSummaryOpKernel();                                
  }                                                                           
  return true;                                                                
}();   

} // namespace 
