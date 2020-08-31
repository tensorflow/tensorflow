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


#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/micro/memory_arena_threshold_test_refvalues.h"

// NOTE: These values are measured on x86-64:
// TODO(b/158651472): Consider auditing these values on non-64 bit systems.

// We need different values for portable_optimized as different ScratchBuffer /
// PersistentBuffer are allocated result in different Head/Tail usage.

const unsigned int TfLiteEvalTensor_size_measured = 24u;
const int kKeywordModelTensorCount = 54;
const int kKeywordModelHeadSize = 672;
#ifdef TF_LITE_STATIC_MEMORY
const int kKeywordModelTailSize = 13936+kKeywordModelTensorCount*(sizeof(TfLiteEvalTensor)-TfLiteEvalTensor_size_measured);
#else
const int kKeywordModelTailSize = 14304+kKeywordModelTensorCount*(sizeof(TfLiteEvalTensor)-TfLiteEvalTensor_size_measured);
#endif
const int kKeywordModellAdditionalOpTailAllocations = 1024+512;

const int kTestConvModelTensorCount = 15;
const int kTestConvModelHeadSize = 7744;
#ifdef TF_LITE_STATIC_MEMORY
const int kTestConvModelTailSize = 2080+kTestConvModelTensorCount*(sizeof(TfLiteEvalTensor)-TfLiteEvalTensor_size_measured);
#else
const int kTestConvModelTailSize = 2240+kTestConvModelTensorCount*(sizeof(TfLiteEvalTensor)-TfLiteEvalTensor_size_measured);    
#endif

