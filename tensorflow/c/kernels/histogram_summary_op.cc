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

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/selective_registration.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// Wrappers to delete resources once the resource is out of scope. 
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
typedef struct HistogramSummaryOp {
  std::string op_node_name; 
};

static void* HistogramSummaryOp_Create(TF_OpKernelConstruction* ctx) {
  HistogramSummaryOp* kernel = new HistogramSummaryOp; 
  TF_StringView string_view_name = TF_OpKernelConstruction_GetName(ctx); 
  kernel->op_node_name = std::string(string_view_name.data, 
      string_view_name.len);
  return kernel; 
}

static void HistogramSummaryOp_Delete(void* kernel) {
  delete static_cast<HistogramSummaryOp*>(kernel); 
}

template<typename T>
static void HistogramSummaryOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  HistogramSummaryOp* k = static_cast<HistogramSummaryOp*>(kernel);
  TensorWrapper tags_wrapper; 
  TensorWrapper values_wrapper; 
  StatusWrapper status_wrapper;
  TF_GetInput(ctx, 0, &tags_wrapper.t, status_wrapper.s);
  if (TF_GetCode(status_wrapper.s) != TF_OK) { 
    TF_OpKernelContext_Failure(ctx, status_wrapper.s);
    return; 
  }
  TF_GetInput(ctx, 1, &values_wrapper.t, status_wrapper.s);
  if (TF_GetCode(status_wrapper.s) != TF_OK) { 
    TF_OpKernelContext_Failure(ctx, status_wrapper.s);
    return; 
  }
  if (TF_NumDims(tags_wrapper.t) != 0) { 
    TF_SetStatus(status_wrapper.s, TF_INVALID_ARGUMENT, "tags must be scalar");
    TF_OpKernelContext_Failure(ctx, status_wrapper.s);
    return; 
  }
  // Cast values to array to access tensor elements by index 
  auto values_array = static_cast<T*>(TF_TensorData(values_wrapper.t)); 
  tensorflow::histogram::Histogram histo; 
  for (int i = 0; i < TF_TensorElementCount(values_wrapper.t); ++i) { 
    const double double_val = static_cast<double>(values_array[i]); 
    if (Eigen::numext::isnan(double_val)) { 
      std::ostringstream err; 
      err << "Nan in summary histogram for: " << k->op_node_name; 
      TF_SetStatus(status_wrapper.s, TF_INVALID_ARGUMENT, err.str().c_str());
      break;
    }
    else if (Eigen::numext::isinf(double_val)) { 
      std::ostringstream err; 
      err << "Infinity in Histogram for: " << k->op_node_name; 
      TF_SetStatus(status_wrapper.s, TF_INVALID_ARGUMENT, err.str().c_str());
      break; 
    }
    histo.Add(double_val);
  }
  tensorflow::Summary s; 
  tensorflow::Summary::Value* v = s.add_value(); 
  const tensorflow::tstring& tag = *(static_cast<tensorflow::tstring*>(
      TF_TensorData(tags_wrapper.t))); 
  v->set_tag(tag.data(), tag.size()); 
  histo.EncodeToProto(v->mutable_histo(), false /* Drop zero buckets */); 

  // Must use new status for TF_AllocateOutput if previous status is set 
  // because of an invalid values argument.  
  StatusWrapper allocation_status_wrapper;
  TensorWrapper summary_tensor_wrapper; 
  TF_Tensor* summary_tensor= TF_AllocateOutput(ctx, 0,
      TF_ExpectedOutputDataType(ctx, 0), nullptr, 0, 
      sizeof(tensorflow::tstring), allocation_status_wrapper.s);
  summary_tensor_wrapper.t = summary_tensor;

  if (TF_GetCode(allocation_status_wrapper.s) != TF_OK){ 
    TF_OpKernelContext_Failure(ctx, allocation_status_wrapper.s);
    return; 
  }
  tensorflow::tstring* output_tstring = reinterpret_cast<tensorflow::tstring*>(
      TF_TensorData(summary_tensor)); 
  CHECK(SerializeToTString(s, output_tstring));
}

template <typename T>
void RegisterHistogramSummaryOpKernel() {
  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder("HistogramSummary", 
                                        tensorflow::DEVICE_CPU,
                                        &HistogramSummaryOp_Create, 
                                        &HistogramSummaryOp_Compute<T>,
                                        &HistogramSummaryOp_Delete);
    TF_KernelBuilder_TypeConstraint(builder, "T", 
        static_cast<TF_DataType>(tensorflow::DataTypeToEnum<T>::v()), status); 
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while adding type constraint";
    TF_RegisterKernelBuilder("HistogramSummary", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Histogram Summmary kernel";
  }
  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the Histogram Summary kernel.                                                          
TF_ATTRIBUTE_UNUSED static bool  IsHistogramSummaryOpKernelRegistered = []() {                  
  if (SHOULD_REGISTER_OP_KERNEL("HistogramSummary")) {
    // RegisterHistogramSummaryOpKernel<tensorflow::int64>();    
    // RegisterHistogramSummaryOpKernel<tensorflow::uint64>();       
    // RegisterHistogramSummaryOpKernel<tensorflow::int32>();   
    // RegisterHistogramSummaryOpKernel<tensorflow::uint32>(); 
    // RegisterHistogramSummaryOpKernel<tensorflow::uint16>();   
    // RegisterHistogramSummaryOpKernel<tensorflow::int16>();   
    // RegisterHistogramSummaryOpKernel<tensorflow::int8>();  
    // RegisterHistogramSummaryOpKernel<tensorflow::uint8>();   
    RegisterHistogramSummaryOpKernel<Eigen::half>();   
    RegisterHistogramSummaryOpKernel<tensorflow::bfloat16>();   
    RegisterHistogramSummaryOpKernel<float>();   
    RegisterHistogramSummaryOpKernel<double>();                                  
  }                                                                           
  return true;                                                                
}(); 
