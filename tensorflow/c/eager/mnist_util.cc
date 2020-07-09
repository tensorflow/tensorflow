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


#include "tensorflow/c/eager/mnist_util.h"
#include "tensorflow/c/eager/c_api_unified_experimental.h"

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/cluster.pb.h"

#include <memory>
#include <iostream>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
//#include "tensorflow/c/eager/c_api_test_util.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::string;
using namespace std;



// Tracing function that traces a MatMul operation in graph mode
TF_AbstractTensor* AbstractMatMul(TF_AbstractTensor* A, TF_AbstractTensor* B, const char* op_name, TF_ExecutionContext* graph_ctx, TF_Status* s) {
    auto* mm_op = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(mm_op, "MatMul", s);
    //ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_AbstractOpSetOpName(mm_op, op_name, s);
    //ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    
    TF_AbstractTensor* inputs[2] = {A, B};
    TF_OutputList* mm_outputs = TF_NewOutputList();
    TF_OutputListSetNumOutputs(mm_outputs, 1, s);
    //ASSERT_EQ(TF_OK, TF_GetCode(s) << TF_Message(s));
    
    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(mm_op, 2, inputs, mm_outputs, s);
    //ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    TF_DeleteAbstractOp(mm_op);
    
    // Extract the resulting tensor.
    TF_AbstractTensor* mm_out = TF_OutputListGet(mm_outputs, 0);
    TF_DeleteOutputList(mm_outputs);

    return mm_out;
}

// Tracing function that traces an Add operation in graph mode
TF_AbstractTensor* AbstractAdd(TF_AbstractTensor* A, TF_AbstractTensor* B, const char* op_name, TF_ExecutionContext* graph_ctx, TF_Status* s) {
    auto* add_op = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(add_op, "Add", s);
    TF_AbstractOpSetOpName(add_op, op_name, s);

    TF_AbstractTensor* inputs[2] = {A, B};
    TF_OutputList* add_outputs = TF_NewOutputList();
    TF_OutputListSetNumOutputs(add_outputs, 1, s);

    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(add_op, 2, inputs, add_outputs, s);
    TF_DeleteAbstractOp(add_op);
    
    // Extract the resulting tensor.
    TF_AbstractTensor* add_out = TF_OutputListGet(add_outputs, 0);
    TF_DeleteOutputList(add_outputs);

    return add_out;
}

TF_AbstractFunction* AbstractFinalizeFunction(TF_OutputList* func_outputs, TF_ExecutionContext* graph_ctx, TF_Status* s){

    TF_AbstractFunction* func = TF_FinalizeFunction(graph_ctx, func_outputs, s); // This also frees the graph_ctx
    int num_outputs =  TF_OutputListNumOutputs(func_outputs);
    
    for(int i = 0; i < num_outputs; i++){
        TF_AbstractTensor* at = TF_OutputListGet(func_outputs, i);
        TF_DeleteAbstractTensor(at);
    }
    
    TF_DeleteOutputList(func_outputs); 
    return func;    
}

TF_AbstractTensor* AbstractRelu(TF_AbstractTensor* A, const char* op_name, TF_ExecutionContext* graph_ctx, TF_Status* s){

    // Build an abstract operation, inputs and output.
    auto* relu_op = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(relu_op, "Relu", s);


    TF_AbstractOpSetOpName(relu_op, "relu", s);
    TF_AbstractTensor* inputs[1] = {A};
    TF_OutputList* relu_outputs = TF_NewOutputList();
    TF_OutputListSetNumOutputs(relu_outputs, 1, s);

    
    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(relu_op, 1, inputs, relu_outputs, s);
    TF_DeleteAbstractOp(relu_op);
    
    // Extract the resulting tensor.
    TF_AbstractTensor* relu_out = TF_OutputListGet(relu_outputs, 0);
    TF_DeleteOutputList(relu_outputs);

    return relu_out; 
}

TF_AbstractTensor* AbstractSparseSoftmaxCrossEntropyLoss(TF_AbstractTensor* scores, TF_AbstractTensor* y_labels, const char* op_name, TF_ExecutionContext* graph_ctx, TF_Status* s){
    
    // Build an abstract operation, inputs and output.
    auto* sm = TF_NewAbstractOp(graph_ctx);
    TF_AbstractOpSetOpType(sm, "SparseSoftmaxCrossEntropyWithLogits", s);
   
   
    TF_AbstractOpSetOpName(sm, op_name, s);
    TF_AbstractTensor* inputs[2] = {scores,y_labels}; 
    TF_OutputList* softmax_outputs = TF_NewOutputList();
    TF_OutputListSetNumOutputs(softmax_outputs, 2, s);

    
    // Trace the operation now (create a node in the graph).
    TF_ExecuteOperation(sm, 2, inputs, softmax_outputs, s);
  
    TF_DeleteAbstractOp(sm);
    
    // Extract the resulting tensor.
    TF_AbstractTensor* softmax_loss = TF_OutputListGet(softmax_outputs, 0);
    TF_AbstractTensor* backprop = TF_OutputListGet(softmax_outputs, 1); // Don't need this for forward pass
    
    //TF_DeleteAbstractTensor(backprop); // getting error when I try to delete this tensor?
    TF_DeleteOutputList(softmax_outputs);

    
    return softmax_loss;

}




