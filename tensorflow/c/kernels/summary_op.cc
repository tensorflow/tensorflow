
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

#include <sstream>

#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/selective_registration.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

// Struct that stores the status and TF_Tensor inputs to the opkernel. 
// Used to delete tensor and status in its destructor upon kernel return. 
typedef struct Params{ 
  TF_Tensor* tags; 
  TF_Tensor* values; 
  TF_Status* status; 
  Params(TF_OpKernelContext* ctx) {
    status = TF_NewStatus();
    TF_GetInput(ctx, 0, &tags, status);
    if (TF_GetCode(status) == TF_OK){ 
      TF_GetInput(ctx, 1, &values, status);
    }
    else{
      values = nullptr; 
    }
  }; 
  ~Params(){ 
    TF_DeleteStatus(status); 
    TF_DeleteTensor(tags); 
    // edge case if params fails to initialize 
    if (values != nullptr){
      TF_DeleteTensor(values); 
    }
  }
}; 

// dummy functions used for kernel registration 
static void* SummaryScalarOp_Create(TF_OpKernelConstruction* ctx) {
  void* ptr; 
  return ptr; 
}

static void SummaryScalarOp_Delete(void* kernel) {
  return;
}

// Helper functions for compute method 
bool IsSameSize(TF_Tensor* tensor1, TF_Tensor* tensor2);
static tensorflow::string SingleTag(TF_Tensor* tags); 

template<typename T>
static void SummaryScalarOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  Params params(ctx);
  if (TF_GetCode(params.status) != TF_OK){ 
    TF_OpKernelContext_Failure(ctx, params.status);
    return; 
  }
  if (!IsSameSize(params.tags, params.values)) {
    std::ostringstream err;
    err << "tags and values not the same shape: " 
        << TF_ShapeDebugString(params.tags) << " != " 
        << TF_ShapeDebugString(params.values)
        << SingleTag(params.tags); 
    TF_SetStatus(params.status, TF_INVALID_ARGUMENT, err.str().c_str());
  }
  if (TF_GetCode(params.status) != TF_OK){ 
    TF_OpKernelContext_Failure(ctx, params.status);
    return; 
  }

  // Convert tags and values tensor to array to access elements by index 
  tensorflow::Summary s; 
  auto tags_array = static_cast<tensorflow::tstring*>(
      TF_TensorData(params.tags));
  auto values_array = static_cast<T*>(TF_TensorData(params.values)); 
  // Copy tags and values into summary protobuf 
  for (int i = 0; i < TF_TensorElementCount(params.tags); ++i) { 
    tensorflow::Summary::Value* v = s.add_value();
    const tensorflow::tstring& Ttags_i = tags_array[i];  
    v->set_tag(Ttags_i.data(), Ttags_i.size());
    v->set_simple_value(float(values_array[i]));
  }
  TF_Tensor* summary_tensor = TF_AllocateOutput(ctx, 0,
      TF_ExpectedOutputDataType(ctx, 0), nullptr, 0, 
      sizeof(tensorflow::tstring), params.status);
  if (TF_GetCode(params.status) != TF_OK){ 
    TF_DeleteTensor(summary_tensor); 
    TF_OpKernelContext_Failure(ctx, params.status);
    return; 
  }
  tensorflow::tstring* output_tstring = reinterpret_cast<tensorflow::tstring*>(
      TF_TensorData(summary_tensor)); 
  SerializeToTString(s, output_tstring);
  TF_DeleteTensor(summary_tensor);
}

bool IsSameSize(TF_Tensor* tensor1, TF_Tensor* tensor2){ 
  if (TF_NumDims(tensor1) != TF_NumDims(tensor2)) {
    return false; 
  }
  for (int d = 0; d < TF_NumDims(tensor1); d++) {
    if (TF_Dim(tensor1, d) != TF_Dim(tensor2, d)) {
      return false; 
    }
  }
  return true; 
}

static tensorflow::string SingleTag(TF_Tensor* tags){ 
  if (TF_TensorElementCount(tags) == 1) { 
    const char* single_tag = static_cast<tensorflow::tstring*>(
        TF_TensorData(tags))->c_str(); 
    return tensorflow::strings::StrCat(" (tag '", single_tag, "')");
  } 
  else {
    return ""; 
  }
}

template <typename T>
void RegisterSummaryScalarOpKernel() {
  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder("SummaryScalar", 
                                        tensorflow::DEVICE_CPU,
                                        &SummaryScalarOp_Create, 
                                        &SummaryScalarOp_Compute<T>,
                                        &SummaryScalarOp_Delete);
    TF_KernelBuilder_TypeConstraint(builder, "T", 
        static_cast<TF_DataType>(tensorflow::DataTypeToEnum<T>::v()), status); 
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while adding type constraint";
    TF_RegisterKernelBuilder("SummaryScalar", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Summary Scalar kernel";
  }
#if GOOGLE_CUDA
  {
    auto* builder = TF_NewKernelBuilder("SummaryScalar", 
                                        tensorflow::DEVICE_GPU,
                                        &SummaryScalarOp_Create, 
                                        &SummaryScalarOp_Compute<T>,
                                        &SummaryScalarOp_Delete);
    TF_RegisterKernelBuilder("SummaryScalar", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering CUDA SummaryScalar kernel";
  }
#endif
  TF_DeleteStatus(status);
}

// A dummy static variable initialized by a lambda whose side-effect is to
// register the bitcast kernel.                                                          
TF_ATTRIBUTE_UNUSED static bool  IsSummaryScalarOpKernelRegistered = []() {                  
  if (SHOULD_REGISTER_OP_KERNEL("SummaryScalar")) {                                                                           
    RegisterSummaryScalarOpKernel<tensorflow::int64>();          
    RegisterSummaryScalarOpKernel<tensorflow::int32>();   
    RegisterSummaryScalarOpKernel<tensorflow::uint16>();   
    RegisterSummaryScalarOpKernel<tensorflow::int16>();   
    RegisterSummaryScalarOpKernel<tensorflow::uint8>();   
    RegisterSummaryScalarOpKernel<Eigen::half>();   
    RegisterSummaryScalarOpKernel<tensorflow::bfloat16>();   
    RegisterSummaryScalarOpKernel<float>();   
    RegisterSummaryScalarOpKernel<double>();                                  
  }                                                                           
  return true;                                                                
}();                                                                          

