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

#include "tensorflow/lite/c/common.h"

#include "tensorflow/lite/c/c_api_types.h"
#ifdef TF_LITE_TENSORFLOW_PROFILER
#include "tensorflow/lite/tensorflow_profiler_logger.h"
#endif

#ifndef TF_LITE_STATIC_MEMORY
#include <stdlib.h>
#include <string.h>
#endif  // TF_LITE_STATIC_MEMORY

extern "C" {

size_t TfLiteIntArrayGetSizeInBytes(int size) {
  static TfLiteIntArray dummy;

  size_t computed_size = sizeof(dummy) + sizeof(dummy.data[0]) * size;
#if defined(_MSC_VER)
  // Context for why this is needed is in http://b/189926408#comment21
  computed_size -= sizeof(dummy.data[0]);
#endif
  return computed_size;
}

int TfLiteIntArrayEqual(const TfLiteIntArray* a, const TfLiteIntArray* b) {
  if (a == b) return 1;
  if (a == nullptr || b == nullptr) return 0;
  return TfLiteIntArrayEqualsArray(a, b->size, b->data);
}

int TfLiteIntArrayEqualsArray(const TfLiteIntArray* a, int b_size,
                              const int b_data[]) {
  if (a == nullptr) return (b_size == 0);
  if (a->size != b_size) return 0;
  int i = 0;
  for (; i < a->size; i++)
    if (a->data[i] != b_data[i]) return 0;
  return 1;
}

#ifndef TF_LITE_STATIC_MEMORY

TfLiteIntArray* TfLiteIntArrayCreate(int size) {
  size_t alloc_size = TfLiteIntArrayGetSizeInBytes(size);
  if (alloc_size <= 0) return nullptr;
  TfLiteIntArray* ret = (TfLiteIntArray*)malloc(alloc_size);
  if (!ret) return ret;
  ret->size = size;
  return ret;
}

TfLiteIntArray* TfLiteIntArrayCopy(const TfLiteIntArray* src) {
  if (!src) return nullptr;
  TfLiteIntArray* ret = TfLiteIntArrayCreate(src->size);
  if (ret) {
    memcpy(ret->data, src->data, src->size * sizeof(int));
  }
  return ret;
}

void TfLiteIntArrayFree(TfLiteIntArray* a) { free(a); }

#endif  // TF_LITE_STATIC_MEMORY

int TfLiteFloatArrayGetSizeInBytes(int size) {
  static TfLiteFloatArray dummy;

  int computed_size = sizeof(dummy) + sizeof(dummy.data[0]) * size;
#if defined(_MSC_VER)
  // Context for why this is needed is in http://b/189926408#comment21
  computed_size -= sizeof(dummy.data[0]);
#endif
  return computed_size;
}

#ifndef TF_LITE_STATIC_MEMORY

TfLiteFloatArray* TfLiteFloatArrayCreate(int size) {
  TfLiteFloatArray* ret =
      (TfLiteFloatArray*)malloc(TfLiteFloatArrayGetSizeInBytes(size));
  ret->size = size;
  return ret;
}

void TfLiteFloatArrayFree(TfLiteFloatArray* a) { free(a); }

void TfLiteTensorDataFree(TfLiteTensor* t) {
  if (t->allocation_type == kTfLiteDynamic ||
      t->allocation_type == kTfLitePersistentRo) {
    if (t->data.raw) {
#ifdef TF_LITE_TENSORFLOW_PROFILER
      tflite::OnTfLiteTensorDealloc(t);
#endif
      free(t->data.raw);
    }
  }
  t->data.raw = nullptr;
}

void TfLiteQuantizationFree(TfLiteQuantization* quantization) {
  if (quantization->type == kTfLiteAffineQuantization) {
    TfLiteAffineQuantization* q_params =
        (TfLiteAffineQuantization*)(quantization->params);
    if (q_params->scale) {
      TfLiteFloatArrayFree(q_params->scale);
      q_params->scale = nullptr;
    }
    if (q_params->zero_point) {
      TfLiteIntArrayFree(q_params->zero_point);
      q_params->zero_point = nullptr;
    }
    free(q_params);
  }
  quantization->params = nullptr;
  quantization->type = kTfLiteNoQuantization;
}

void TfLiteSparsityFree(TfLiteSparsity* sparsity) {
  if (sparsity == nullptr) {
    return;
  }

  if (sparsity->traversal_order) {
    TfLiteIntArrayFree(sparsity->traversal_order);
    sparsity->traversal_order = nullptr;
  }

  if (sparsity->block_map) {
    TfLiteIntArrayFree(sparsity->block_map);
    sparsity->block_map = nullptr;
  }

  if (sparsity->dim_metadata) {
    int i = 0;
    for (; i < sparsity->dim_metadata_size; i++) {
      TfLiteDimensionMetadata metadata = sparsity->dim_metadata[i];
      if (metadata.format == kTfLiteDimSparseCSR) {
        TfLiteIntArrayFree(metadata.array_segments);
        metadata.array_segments = nullptr;
        TfLiteIntArrayFree(metadata.array_indices);
        metadata.array_indices = nullptr;
      }
    }
    free(sparsity->dim_metadata);
    sparsity->dim_metadata = nullptr;
  }

  free(sparsity);
}

void TfLiteTensorFree(TfLiteTensor* t) {
  TfLiteTensorDataFree(t);
  if (t->dims) TfLiteIntArrayFree(t->dims);
  t->dims = nullptr;

  if (t->dims_signature) {
    TfLiteIntArrayFree((TfLiteIntArray*)t->dims_signature);
  }
  t->dims_signature = nullptr;

  TfLiteQuantizationFree(&t->quantization);
  TfLiteSparsityFree(t->sparsity);
  t->sparsity = nullptr;
}

void TfLiteTensorReset(TfLiteType type, const char* name, TfLiteIntArray* dims,
                       TfLiteQuantizationParams quantization, char* buffer,
                       size_t size, TfLiteAllocationType allocation_type,
                       const void* allocation, bool is_variable,
                       TfLiteTensor* tensor) {
  TfLiteTensorFree(tensor);
  tensor->type = type;
  tensor->name = name;
  tensor->dims = dims;
  tensor->params = quantization;
  tensor->data.raw = buffer;
  tensor->bytes = size;
  tensor->allocation_type = allocation_type;
  tensor->allocation = allocation;
  tensor->is_variable = is_variable;

  tensor->quantization.type = kTfLiteNoQuantization;
  tensor->quantization.params = nullptr;
}

TfLiteStatus TfLiteTensorCopy(const TfLiteTensor* src, TfLiteTensor* dst) {
  if (!src || !dst) return kTfLiteOk;
  if (src->bytes != dst->bytes) return kTfLiteError;
  if (src == dst) return kTfLiteOk;

  dst->type = src->type;
  if (dst->dims) TfLiteIntArrayFree(dst->dims);
  dst->dims = TfLiteIntArrayCopy(src->dims);
  memcpy(dst->data.raw, src->data.raw, src->bytes);
  dst->buffer_handle = src->buffer_handle;
  dst->data_is_stale = src->data_is_stale;
  dst->delegate = src->delegate;

  return kTfLiteOk;
}

void TfLiteTensorRealloc(size_t num_bytes, TfLiteTensor* tensor) {
  if (tensor->allocation_type != kTfLiteDynamic &&
      tensor->allocation_type != kTfLitePersistentRo) {
    return;
  }
  // TODO(b/145340303): Tensor data should be aligned.
  if (!tensor->data.raw) {
    tensor->data.raw = (char*)malloc(num_bytes);
#ifdef TF_LITE_TENSORFLOW_PROFILER
    tflite::OnTfLiteTensorAlloc(tensor, num_bytes);
#endif
  } else if (num_bytes > tensor->bytes) {
#ifdef TF_LITE_TENSORFLOW_PROFILER
    tflite::OnTfLiteTensorDealloc(tensor);
#endif
    tensor->data.raw = (char*)realloc(tensor->data.raw, num_bytes);
#ifdef TF_LITE_TENSORFLOW_PROFILER
    tflite::OnTfLiteTensorAlloc(tensor, num_bytes);
#endif
  }
  tensor->bytes = num_bytes;
}
#endif  // TF_LITE_STATIC_MEMORY

const char* TfLiteTypeGetName(TfLiteType type) {
  switch (type) {
    case kTfLiteNoType:
      return "NOTYPE";
    case kTfLiteFloat32:
      return "FLOAT32";
    case kTfLiteUInt16:
      return "UINT16";
    case kTfLiteInt16:
      return "INT16";
    case kTfLiteInt32:
      return "INT32";
    case kTfLiteUInt32:
      return "UINT32";
    case kTfLiteUInt8:
      return "UINT8";
    case kTfLiteInt8:
      return "INT8";
    case kTfLiteInt64:
      return "INT64";
    case kTfLiteUInt64:
      return "UINT64";
    case kTfLiteBool:
      return "BOOL";
    case kTfLiteComplex64:
      return "COMPLEX64";
    case kTfLiteComplex128:
      return "COMPLEX128";
    case kTfLiteString:
      return "STRING";
    case kTfLiteFloat16:
      return "FLOAT16";
    case kTfLiteFloat64:
      return "FLOAT64";
    case kTfLiteResource:
      return "RESOURCE";
    case kTfLiteVariant:
      return "VARIANT";
  }
  return "Unknown type";
}

TfLiteDelegate TfLiteDelegateCreate() { return TfLiteDelegate{}; }

}  // extern "C"
