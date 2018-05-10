/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/java/src/main/native/tensor_jni.h"
#include <cstring>
#include <memory>
#include "tensorflow/contrib/lite/java/src/main/native/exception_jni.h"

namespace {

TfLiteTensor* convertLongToTensor(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to TfLiteTensor.");
    return nullptr;
  }
  return reinterpret_cast<TfLiteTensor*>(handle);
}

size_t writeOneDimensionalArray(JNIEnv* env, jobject object, TfLiteType type,
                                void* dst, size_t dst_size) {
  jarray array = static_cast<jarray>(object);
  const int num_elements = env->GetArrayLength(array);
  size_t to_copy = num_elements * elementByteSize(type);
  if (to_copy > dst_size) {
    throwException(env, kIllegalStateException,
                   "Internal error: cannot write Java array of %d bytes to "
                   "Tensor of %d bytes",
                   to_copy, dst_size);
    return 0;
  }
  switch (type) {
    case kTfLiteFloat32: {
      jfloatArray a = static_cast<jfloatArray>(array);
      jfloat* values = env->GetFloatArrayElements(a, nullptr);
      memcpy(dst, values, to_copy);
      env->ReleaseFloatArrayElements(a, values, JNI_ABORT);
      return to_copy;
    }
    case kTfLiteInt32: {
      jintArray a = static_cast<jintArray>(array);
      jint* values = env->GetIntArrayElements(a, nullptr);
      memcpy(dst, values, to_copy);
      env->ReleaseIntArrayElements(a, values, JNI_ABORT);
      return to_copy;
    }
    case kTfLiteInt64: {
      jlongArray a = static_cast<jlongArray>(array);
      jlong* values = env->GetLongArrayElements(a, nullptr);
      memcpy(dst, values, to_copy);
      env->ReleaseLongArrayElements(a, values, JNI_ABORT);
      return to_copy;
    }
    case kTfLiteUInt8: {
      jbyteArray a = static_cast<jbyteArray>(array);
      jbyte* values = env->GetByteArrayElements(a, nullptr);
      memcpy(dst, values, to_copy);
      env->ReleaseByteArrayElements(a, values, JNI_ABORT);
      return to_copy;
    }
    default: {
      throwException(env, kUnsupportedOperationException,
                     "DataType error: TensorFlowLite currently supports float "
                     "(32 bits), int (32 bits), byte (8 bits), and long "
                     "(64 bits), support for other types (DataType %d in this "
                     "case) will be added in the future",
                     kTfLiteFloat32, type);
      return 0;
    }
  }
}

size_t readOneDimensionalArray(JNIEnv* env, TfLiteType data_type,
                               const void* src, size_t src_size, jarray dst) {
  const int len = env->GetArrayLength(dst);
  const size_t size = len * elementByteSize(data_type);
  if (size > src_size) {
    throwException(
        env, kIllegalStateException,
        "Internal error: cannot fill a Java array of %d bytes with a Tensor of "
        "%d bytes",
        size, src_size);
    return 0;
  }
  switch (data_type) {
    case kTfLiteFloat32: {
      jfloatArray float_array = static_cast<jfloatArray>(dst);
      env->SetFloatArrayRegion(float_array, 0, len,
                               static_cast<const jfloat*>(src));
      return size;
    }
    case kTfLiteInt32: {
      jintArray int_array = static_cast<jintArray>(dst);
      env->SetIntArrayRegion(int_array, 0, len, static_cast<const jint*>(src));
      return size;
    }
    case kTfLiteInt64: {
      jlongArray long_array = static_cast<jlongArray>(dst);
      env->SetLongArrayRegion(long_array, 0, len,
                              static_cast<const jlong*>(src));
      return size;
    }
    case kTfLiteUInt8: {
      jbyteArray byte_array = static_cast<jbyteArray>(dst);
      env->SetByteArrayRegion(byte_array, 0, len,
                              static_cast<const jbyte*>(src));
      return size;
    }
    default: {
      throwException(env, kIllegalStateException,
                     "DataType error: invalid DataType(%d)", data_type);
    }
  }
  return 0;
}

size_t readMultiDimensionalArray(JNIEnv* env, TfLiteType data_type, char* src,
                                 size_t src_size, int dims_left, jarray dst) {
  if (dims_left == 1) {
    return readOneDimensionalArray(env, data_type, src, src_size, dst);
  } else {
    jobjectArray ndarray = static_cast<jobjectArray>(dst);
    int len = env->GetArrayLength(ndarray);
    size_t size = 0;
    for (int i = 0; i < len; ++i) {
      jarray row = static_cast<jarray>(env->GetObjectArrayElement(ndarray, i));
      size += readMultiDimensionalArray(env, data_type, src + size,
                                        src_size - size, dims_left - 1, row);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return size;
    }
    return size;
  }
}

}  // namespace

size_t elementByteSize(TfLiteType data_type) {
  // The code in this file makes the assumption that the
  // TensorFlow TF_DataTypes and the Java primitive types
  // have the same byte sizes. Validate that:
  switch (data_type) {
    case kTfLiteFloat32:
      static_assert(sizeof(jfloat) == 4,
                    "Interal error: Java float not compatible with "
                    "kTfLiteFloat");
      return 4;
    case kTfLiteInt32:
      static_assert(sizeof(jint) == 4,
                    "Interal error: Java int not compatible with kTfLiteInt");
      return 4;
    case kTfLiteUInt8:
      static_assert(sizeof(jbyte) == 1,
                    "Interal error: Java byte not compatible with "
                    "kTfLiteUInt8");
      return 1;
    case kTfLiteInt64:
      static_assert(sizeof(jlong) == 8,
                    "Interal error: Java long not compatible with "
                    "kTfLiteInt64");
      return 8;
    default:
      return 0;
  }
}

size_t writeByteBuffer(JNIEnv* env, jobject object, char** dst, int dst_size) {
  char* buf = static_cast<char*>(env->GetDirectBufferAddress(object));
  if (!buf) {
    throwException(env, kIllegalArgumentException,
                   "Input ByteBuffer is not a direct buffer");
    return 0;
  }
  *dst = buf;
  return dst_size;
}

size_t writeMultiDimensionalArray(JNIEnv* env, jobject src, TfLiteType type,
                                  int dims_left, char** dst, int dst_size) {
  if (dims_left <= 1) {
    return writeOneDimensionalArray(env, src, type, *dst, dst_size);
  } else {
    jobjectArray ndarray = static_cast<jobjectArray>(src);
    int len = env->GetArrayLength(ndarray);
    size_t sz = 0;
    for (int i = 0; i < len; ++i) {
      jobject row = env->GetObjectArrayElement(ndarray, i);
      char* next_dst = *dst + sz;
      sz += writeMultiDimensionalArray(env, row, type, dims_left - 1, &next_dst,
                                       dst_size - sz);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return sz;
    }
    return sz;
  }
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Tensor_readMultiDimensionalArray(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle,
                                                          jobject value) {
  TfLiteTensor* tensor = convertLongToTensor(env, handle);
  if (tensor == nullptr) return;
  int num_dims = tensor->dims->size;
  if (num_dims == 0) {
    throwException(env, kIllegalArgumentException,
                   "Internal error: Cannot copy empty/scalar Tensors.");
    return;
  }
  readMultiDimensionalArray(env, tensor->type, tensor->data.raw, tensor->bytes,
                            num_dims, static_cast<jarray>(value));
}

JNIEXPORT jint JNICALL Java_org_tensorflow_lite_Tensor_dtype(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle) {
  TfLiteTensor* tensor = convertLongToTensor(env, handle);
  if (tensor == nullptr) return 0;
  return static_cast<jint>(tensor->type);
}

JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_Tensor_shape(JNIEnv* env, jclass clazz, jlong handle) {
  TfLiteTensor* tensor = convertLongToTensor(env, handle);
  if (tensor == nullptr) return nullptr;
  int num_dims = tensor->dims->size;
  jintArray result = env->NewIntArray(num_dims);
  env->SetIntArrayRegion(result, 0, num_dims, tensor->dims->data);
  return result;
}
