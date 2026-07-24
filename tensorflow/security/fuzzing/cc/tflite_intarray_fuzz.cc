/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <cassert>
#include <cstddef>

#include "fuzztest/fuzztest.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"

// This fuzzer exercises the TfLiteIntArray/TfLiteFloatArray size handling and
// the delegate setup path (`ReplaceNodeSubsetsWithDelegateKernels`) against
// malformed array sizes. Negative sizes must be rejected without crashes or
// undefined behavior, and valid sizes (including zero) must keep working.

namespace {

// Sizes above this bound are only run through the size arithmetic, not
// through allocation, to keep the fuzzer's memory usage bounded.
constexpr int kMaxAllocatedSize = 1 << 16;

void FuzzArraySizeHandling(int size) {
  const size_t int_array_bytes = TfLiteIntArrayGetSizeInBytes(size);
  if (size < 0) {
    assert(int_array_bytes == 0);
  }
  (void)int_array_bytes;

  if (size > kMaxAllocatedSize) {
    return;
  }

  TfLiteIntArray* int_array = TfLiteIntArrayCreate(size);
  if (size < 0) {
    assert(int_array == nullptr);
  }
  if (int_array != nullptr) {
    assert(int_array->size == size);
    // Touch every element so that sanitizers verify the allocation really
    // covers the declared size.
    for (int i = 0; i < size; ++i) {
      int_array->data[i] = i;
    }
    TfLiteIntArrayFree(int_array);
  }

  TfLiteFloatArray* float_array = TfLiteFloatArrayCreate(size);
  if (size < 0) {
    assert(float_array == nullptr);
  }
  if (float_array != nullptr) {
    assert(float_array->size == size);
    for (int i = 0; i < size; ++i) {
      float_array->data[i] = 0.0f;
    }
    TfLiteFloatArrayFree(float_array);
  }
}
FUZZ_TEST(TfLiteArrayFuzz, FuzzArraySizeHandling);

TfLiteOperator* CreateDelegateKernelRegistration() {
  TfLiteOperator* registration = TfLiteOperatorCreate(
      kTfLiteBuiltinDelegate, "FUZZ DELEGATE KERNEL", /*version=*/1,
      /*user_data=*/nullptr);
  TfLiteOperatorSetInitWithData(
      registration,
      [](void* user_data, TfLiteOpaqueContext* context, const char* buffer,
         size_t length) -> void* { return nullptr; });
  return registration;
}

void FuzzDelegateNodesToReplace(int raw_size) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add.bin");
  assert(model != nullptr);

  struct DelegateState {
    int raw_size;
  } delegate_state{raw_size};

  TfLiteOpaqueDelegateBuilder opaque_delegate_builder{};
  opaque_delegate_builder.data = &delegate_state;
  opaque_delegate_builder.Prepare = [](TfLiteOpaqueContext* opaque_context,
                                       TfLiteOpaqueDelegate* opaque_delegate,
                                       void* data) -> TfLiteStatus {
    const int raw_size = static_cast<DelegateState*>(data)->raw_size;

    TfLiteIntArray* execution_plan = nullptr;
    TfLiteStatus status =
        TfLiteOpaqueContextGetExecutionPlan(opaque_context, &execution_plan);
    assert(status == kTfLiteOk);

    // Build an array whose allocation always covers the full execution plan
    // and whose contents are valid node indices, then declare a fuzzed size:
    // negative values must be rejected, values in [0, plan size] must be
    // handled without crashing. Out-of-range positive values are not
    // distinguishable from valid sizes at this API level, so they are folded
    // into the valid range.
    TfLiteIntArray* nodes_to_replace = TfLiteIntArrayCopy(execution_plan);
    assert(nodes_to_replace != nullptr);
    const int declared_size =
        raw_size < 0 ? raw_size : raw_size % (execution_plan->size + 1);
    nodes_to_replace->size = declared_size;

    status = TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
        opaque_context, CreateDelegateKernelRegistration(), nodes_to_replace,
        opaque_delegate);
    if (declared_size < 0) {
      assert(status == kTfLiteError);
    }
    (void)status;

    // Restore the real size so the free call sees the array it allocated.
    nodes_to_replace->size = execution_plan->size;
    TfLiteIntArrayFree(nodes_to_replace);
    return kTfLiteOk;
  };

  TfLiteOpaqueDelegate* opaque_delegate =
      TfLiteOpaqueDelegateCreate(&opaque_delegate_builder);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsAddDelegate(options, opaque_delegate);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  TfLiteInterpreterDelete(interpreter);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteOpaqueDelegateDelete(opaque_delegate);
  TfLiteModelDelete(model);
}
FUZZ_TEST(TfLiteDelegateFuzz, FuzzDelegateNodesToReplace);

}  // namespace
