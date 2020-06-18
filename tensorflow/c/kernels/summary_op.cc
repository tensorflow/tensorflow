
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

// TODO: Copy over Summary Scalar Op Doc 

static void* SummaryScalarOp_Create(TF_OpKernelConstruction* ctx) {
  // TODO: replace with a void* pointer type later 
  int a = 4; 
  return static_cast<void*>(&a); 
}

static void SummaryScalarOp_Delete(void* kernel) {
  return;
}

bool IsSameSize(TF_Tensor* tensor1, TF_Tensor* tensor2){ 
  if (TF_NumDims(tensor1) != TF_NumDims(tensor2)){
    return false; 
  }
  for(int d = 0; d < TF_NumDims(tensor1); d++){
    if (TF_Dim(tensor1, d) != TF_Dim(tensor2, d)){
      return false; 
    }
  }
  return true; 
}

template<typename T>
static void SummaryScalarOp_Compute(void* kernel, TF_OpKernelContext* ctx) {
  TF_Tensor* tags;
  TF_Tensor* values; 
  TF_Status* status = TF_NewStatus();
  TF_GetInput(ctx, 0, &tags, status);
          CHECK_EQ(TF_OK, TF_GetCode(status))
    << "Error while getting input"; 
  if (TF_GetCode(status) == TF_OK){
    TF_GetInput(ctx, 1, &values, status);
  } 
  CHECK_EQ(TF_OK, TF_GetCode(status))
    << "Error while getting input";
  if (TF_GetCode(status) == TF_OK) {
    if (!IsSameSize(tags, values)) {
      std::ostringstream err;
      err << "tags and values not the same shape: "; 
      TF_SetStatus(status, TF_INVALID_ARGUMENT, err.str().c_str());
    }
  }
 // Copy tag and string data into summary protobuf 
  tensorflow::Summary s; 
  if (TF_GetCode(status) == TF_OK) {
    auto Ttags_array = static_cast<TF_TString**>(TF_TensorData(tags)); 
    auto values_array = static_cast<T*>(TF_TensorData(values)); 
    for (int i = 0; i < TF_TensorElementCount(tags); ++i){ 
      tensorflow::Summary::Value* v = s.add_value(); 
      TF_TString_Init(Ttags_array[i]); 
      v->set_tag(TF_TString_GetDataPointer(Ttags_array[i]), TF_TString_GetSize(Ttags_array[i]));
      v->set_simple_value(float(values_array[i]));
    }


    // TF_Tensor* summary_tensor = TF_AllocateOutput(ctx, 0, TF_ExpectedOutputDataType(ctx, 0), 0, 0)  

    // TF_Tensor* output = TF_AllocateTensor(k->output_data_type, dims, 0,
    //                                       TF_DataTypeSize(k->output_data_type));
    // if (TF_GetCode(status) == TF_OK) {
    //   TF_SetOutput(ctx, 0, output, status);
    // }
    // TF_DeleteTensor(output);
  }

  // if (TF_GetCode(status) != TF_OK) {
  //   TF_OpKernelContext_Failure(ctx, status);
  // }
  // TF_DeleteStatus(status);
  // TF_DeleteTensor(tags);
}

template <typename T>
void RegisterSummaryScalarOpKernel() {
  TF_Status* status = TF_NewStatus();
  {
    auto* builder = TF_NewKernelBuilder("SummaryScalar", tensorflow::DEVICE_CPU,
                                        &SummaryScalarOp_Create, &SummaryScalarOp_Compute<T>,
                                        &SummaryScalarOp_Delete);
    TF_KernelBuilder_TypeConstraint(builder, "T", static_cast<TF_DataType>(tensorflow::DataTypeToEnum<T>::v()), status); 
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while adding type constraint";
    TF_RegisterKernelBuilder("SummaryScalar", builder, status);
    CHECK_EQ(TF_OK, TF_GetCode(status))
        << "Error while registering Summary Scalar kernel";
  }
// template <typename T>
// #if GOOGLE_CUDA
//   {
//     auto* builder = TF_NewKernelBuilder("SummaryScalar", tensorflow::DEVICE_GPU,
//                                         &SummaryScalarOp_Create, &SummaryScalarOp_Compute,
//                                         &SummaryScalarOp_Delete);
//     TF_RegisterKernelBuilder("SummaryScalar", builder, status);
//     CHECK_EQ(TF_OK, TF_GetCode(status))
//         << "Error while registering CUDA SummaryScalar kernel";
//   }
// #endif

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

