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

#include <jni.h>

#include <cstring>
#include <memory>
#include <string>

#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow/lite/core/shims/cc/interpreter.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"
#include "tensorflow/lite/string_util.h"

using tflite::jni::ThrowException;
using tflite_shims::Interpreter;

namespace {

static const char* kByteArrayClassPath = "[B";
static const char* kStringClassPath = "java/lang/String";

// Convenience handle for obtaining a TfLiteTensor given an interpreter and
// tensor index.
//
// Historically, the Java Tensor class used a TfLiteTensor pointer as its native
// handle. However, this approach isn't generally safe, as the interpreter may
// invalidate all TfLiteTensor* handles during inference or allocation.
class TensorHandle {
 public:
  TensorHandle(Interpreter* interpreter, int tensor_index)
      : interpreter_(interpreter), tensor_index_(tensor_index) {}

  TfLiteTensor* tensor() const { return interpreter_->tensor(tensor_index_); }
  int index() const { return tensor_index_; }

 private:
  Interpreter* const interpreter_;
  const int tensor_index_;
};

TfLiteTensor* GetTensorFromHandle(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Invalid handle to TfLiteTensor.");
    return nullptr;
  }
  return reinterpret_cast<TensorHandle*>(handle)->tensor();
}

int GetTensorIndexFromHandle(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Invalid handle to TfLiteTensor.");
    return -1;
  }
  return reinterpret_cast<TensorHandle*>(handle)->index();
}

size_t ElementByteSize(TfLiteType data_type) {
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
    case kTfLiteInt8:
      static_assert(sizeof(jbyte) == 1,
                    "Interal error: Java byte not compatible with "
                    "kTfLiteUInt8");
      return 1;
    case kTfLiteBool:
      static_assert(sizeof(jboolean) == 1,
                    "Interal error: Java boolean not compatible with "
                    "kTfLiteBool");
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

size_t WriteOneDimensionalArray(JNIEnv* env, jobject object, TfLiteType type,
                                void* dst, size_t dst_size) {
  jarray array = static_cast<jarray>(object);
  const int num_elements = env->GetArrayLength(array);
  size_t to_copy = num_elements * ElementByteSize(type);
  if (to_copy > dst_size) {
    ThrowException(env, tflite::jni::kIllegalStateException,
                   "Internal error: cannot write Java array of %d bytes to "
                   "Tensor of %d bytes",
                   to_copy, dst_size);
    return 0;
  }
  switch (type) {
    case kTfLiteFloat32: {
      jfloatArray float_array = static_cast<jfloatArray>(array);
      jfloat* float_dst = static_cast<jfloat*>(dst);
      env->GetFloatArrayRegion(float_array, 0, num_elements, float_dst);
      return to_copy;
    }
    case kTfLiteInt32: {
      jintArray int_array = static_cast<jintArray>(array);
      jint* int_dst = static_cast<jint*>(dst);
      env->GetIntArrayRegion(int_array, 0, num_elements, int_dst);
      return to_copy;
    }
    case kTfLiteInt64: {
      jlongArray long_array = static_cast<jlongArray>(array);
      jlong* long_dst = static_cast<jlong*>(dst);
      env->GetLongArrayRegion(long_array, 0, num_elements, long_dst);
      return to_copy;
    }
    case kTfLiteInt8:
    case kTfLiteUInt8: {
      jbyteArray byte_array = static_cast<jbyteArray>(array);
      jbyte* byte_dst = static_cast<jbyte*>(dst);
      env->GetByteArrayRegion(byte_array, 0, num_elements, byte_dst);
      return to_copy;
    }
    case kTfLiteBool: {
      jbooleanArray bool_array = static_cast<jbooleanArray>(array);
      jboolean* bool_dst = static_cast<jboolean*>(dst);
      env->GetBooleanArrayRegion(bool_array, 0, num_elements, bool_dst);
      return to_copy;
    }
    default: {
      ThrowException(
          env, tflite::jni::kUnsupportedOperationException,
          "DataType error: TensorFlowLite currently supports float "
          "(32 bits), int (32 bits), byte (8 bits), bool (8 bits), and long "
          "(64 bits), support for other types (DataType %d in this "
          "case) will be added in the future",
          kTfLiteFloat32, type);
      return 0;
    }
  }
}

size_t ReadOneDimensionalArray(JNIEnv* env, TfLiteType data_type,
                               const void* src, size_t src_size, jarray dst) {
  const int len = env->GetArrayLength(dst);
  const size_t size = len * ElementByteSize(data_type);
  if (size > src_size) {
    ThrowException(
        env, tflite::jni::kIllegalStateException,
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
    case kTfLiteInt8:
    case kTfLiteUInt8: {
      jbyteArray byte_array = static_cast<jbyteArray>(dst);
      env->SetByteArrayRegion(byte_array, 0, len,
                              static_cast<const jbyte*>(src));
      return size;
    }
    case kTfLiteBool: {
      jbooleanArray bool_array = static_cast<jbooleanArray>(dst);
      env->SetBooleanArrayRegion(bool_array, 0, len,
                                 static_cast<const jboolean*>(src));
      return size;
    }
    default: {
      ThrowException(env, tflite::jni::kIllegalStateException,
                     "DataType error: invalid DataType(%d)", data_type);
    }
  }
  return 0;
}

size_t ReadMultiDimensionalArray(JNIEnv* env, TfLiteType data_type, char* src,
                                 size_t src_size, int dims_left, jarray dst) {
  if (dims_left == 1) {
    return ReadOneDimensionalArray(env, data_type, src, src_size, dst);
  } else {
    jobjectArray ndarray = static_cast<jobjectArray>(dst);
    int len = env->GetArrayLength(ndarray);
    size_t size = 0;
    for (int i = 0; i < len; ++i) {
      jarray row = static_cast<jarray>(env->GetObjectArrayElement(ndarray, i));
      size += ReadMultiDimensionalArray(env, data_type, src + size,
                                        src_size - size, dims_left - 1, row);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return size;
    }
    return size;
  }
}

// Returns the total number of strings read.
int ReadMultiDimensionalStringArray(JNIEnv* env, TfLiteTensor* tensor,
                                    int dims_left, int start_str_index,
                                    jarray dst) {
  jobjectArray object_array = static_cast<jobjectArray>(dst);
  int len = env->GetArrayLength(object_array);
  int num_strings_read = 0;

  // If dst is a 1-dimensional array, copy the strings into it. Else
  // recursively call ReadMultiDimensionalStringArray over sub-dimensions.
  if (dims_left == 1) {
    for (int i = 0; i < len; ++i) {
      const tflite::StringRef strref =
          tflite::GetString(tensor, start_str_index + num_strings_read);
      // Makes sure the string is null terminated before passing to
      // NewStringUTF.
      std::string str(strref.str, strref.len);
      jstring string_dest = env->NewStringUTF(str.data());
      env->SetObjectArrayElement(object_array, i, string_dest);
      env->DeleteLocalRef(string_dest);
      ++num_strings_read;
    }
  } else {
    for (int i = 0; i < len; ++i) {
      jarray row =
          static_cast<jarray>(env->GetObjectArrayElement(object_array, i));
      num_strings_read += ReadMultiDimensionalStringArray(
          env, tensor, dims_left - 1, start_str_index + num_strings_read, row);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return num_strings_read;
    }
  }

  return num_strings_read;
}

size_t WriteMultiDimensionalArray(JNIEnv* env, jobject src, TfLiteType type,
                                  int dims_left, char** dst, int dst_size) {
  if (dims_left <= 1) {
    return WriteOneDimensionalArray(env, src, type, *dst, dst_size);
  } else {
    jobjectArray ndarray = static_cast<jobjectArray>(src);
    int len = env->GetArrayLength(ndarray);
    size_t sz = 0;
    for (int i = 0; i < len; ++i) {
      jobject row = env->GetObjectArrayElement(ndarray, i);
      char* next_dst = *dst + sz;
      sz += WriteMultiDimensionalArray(env, row, type, dims_left - 1, &next_dst,
                                       dst_size - sz);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return sz;
    }
    return sz;
  }
}

void AddStringDynamicBuffer(JNIEnv* env, jobject src,
                            tflite::DynamicBuffer* dst_buffer) {
  if (env->IsInstanceOf(src, env->FindClass(kStringClassPath))) {
    jstring str = static_cast<jstring>(src);
    const char* chars = env->GetStringUTFChars(str, nullptr);
    // + 1 for terminating character.
    const int byte_len = env->GetStringUTFLength(str) + 1;
    dst_buffer->AddString(chars, byte_len);
    env->ReleaseStringUTFChars(str, chars);
  }
  if (env->IsInstanceOf(src, env->FindClass(kByteArrayClassPath))) {
    jbyteArray byte_array = static_cast<jbyteArray>(src);
    jsize byte_array_length = env->GetArrayLength(byte_array);
    jbyte* bytes = env->GetByteArrayElements(byte_array, nullptr);
    dst_buffer->AddString(reinterpret_cast<const char*>(bytes),
                          byte_array_length);
    env->ReleaseByteArrayElements(byte_array, bytes, JNI_ABORT);
  }
}

void PopulateStringDynamicBuffer(JNIEnv* env, jobject src,
                                 tflite::DynamicBuffer* dst_buffer,
                                 int dims_left) {
  jobjectArray object_array = static_cast<jobjectArray>(src);
  const int num_elements = env->GetArrayLength(object_array);

  // If src is a 1-dimensional array, add the strings into dst_buffer. Else
  // recursively call populateStringDynamicBuffer over sub-dimensions.
  if (dims_left <= 1) {
    for (int i = 0; i < num_elements; ++i) {
      jobject obj = env->GetObjectArrayElement(object_array, i);
      AddStringDynamicBuffer(env, obj, dst_buffer);
      env->DeleteLocalRef(obj);
    }
  } else {
    for (int i = 0; i < num_elements; ++i) {
      jobject row = env->GetObjectArrayElement(object_array, i);
      PopulateStringDynamicBuffer(env, row, dst_buffer, dims_left - 1);
      env->DeleteLocalRef(row);
      if (env->ExceptionCheck()) return;
    }
  }
}

void WriteMultiDimensionalStringArray(JNIEnv* env, jobject src,
                                      TfLiteTensor* tensor) {
  tflite::DynamicBuffer dst_buffer;
  PopulateStringDynamicBuffer(env, src, &dst_buffer, tensor->dims->size);
  if (!env->ExceptionCheck()) {
    dst_buffer.WriteToTensor(tensor, /*new_shape=*/nullptr);
  }
}

void WriteScalar(JNIEnv* env, jobject src, TfLiteType type, void* dst,
                 int dst_size) {
  size_t src_size = ElementByteSize(type);
  if (src_size != dst_size) {
    ThrowException(
        env, tflite::jni::kIllegalStateException,
        "Scalar (%d bytes) not compatible with allocated tensor (%d bytes)",
        src_size, dst_size);
    return;
  }
  switch (type) {
// env->FindClass and env->GetMethodID are expensive and JNI best practices
// suggest that they should be cached. However, until the creation of scalar
// valued tensors seems to become a noticeable fraction of program execution,
// ignore that cost.
#define CASE(type, jtype, method_name, method_signature, call_type)            \
  case type: {                                                                 \
    jclass clazz = env->FindClass("java/lang/Number");                         \
    jmethodID method = env->GetMethodID(clazz, method_name, method_signature); \
    jtype v = env->Call##call_type##Method(src, method);                       \
    memcpy(dst, &v, src_size);                                                 \
    return;                                                                    \
  }
    CASE(kTfLiteFloat32, jfloat, "floatValue", "()F", Float);
    CASE(kTfLiteInt32, jint, "intValue", "()I", Int);
    CASE(kTfLiteInt64, jlong, "longValue", "()J", Long);
    CASE(kTfLiteInt8, jbyte, "byteValue", "()B", Byte);
    CASE(kTfLiteUInt8, jbyte, "byteValue", "()B", Byte);
#undef CASE
    case kTfLiteBool: {
      jclass clazz = env->FindClass("java/lang/Boolean");
      jmethodID method = env->GetMethodID(clazz, "booleanValue", "()Z");
      jboolean v = env->CallBooleanMethod(src, method);
      *(static_cast<unsigned char*>(dst)) = v ? 1 : 0;
      return;
    }
    default:
      ThrowException(env, tflite::jni::kIllegalStateException,
                     "Invalid DataType(%d)", type);
      return;
  }
}

void WriteScalarString(JNIEnv* env, jobject src, TfLiteTensor* tensor) {
  tflite::DynamicBuffer dst_buffer;
  AddStringDynamicBuffer(env, src, &dst_buffer);
  if (!env->ExceptionCheck()) {
    dst_buffer.WriteToTensor(tensor, /*new_shape=*/nullptr);
  }
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_org_tensorflow_lite_Tensor_create(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jint tensor_index) {
  Interpreter* interpreter = reinterpret_cast<Interpreter*>(interpreter_handle);
  return reinterpret_cast<jlong>(new TensorHandle(interpreter, tensor_index));
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_Tensor_delete(JNIEnv* env,
                                                              jclass clazz,
                                                              jlong handle) {
  delete reinterpret_cast<TensorHandle*>(handle);
}

JNIEXPORT jobject JNICALL Java_org_tensorflow_lite_Tensor_buffer(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle) {
  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return nullptr;
  if (tensor->data.raw == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Tensor hasn't been allocated.");
    return nullptr;
  }
  return env->NewDirectByteBuffer(static_cast<void*>(tensor->data.raw),
                                  static_cast<jlong>(tensor->bytes));
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_Tensor_writeDirectBuffer(
    JNIEnv* env, jclass clazz, jlong handle, jobject src) {
  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return;

  void* src_data_raw = env->GetDirectBufferAddress(src);
  if (!src_data_raw) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Input ByteBuffer is not a direct buffer");
    return;
  }

  if (!tensor->data.data) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Tensor hasn't been allocated.");
    return;
  }

  // Historically, we would simply overwrite the tensor buffer pointer with
  // the direct Buffer address. However, that is generally unsafe, and
  // specifically wrong if the graph happens to have dynamic shapes where
  // arena-allocated input buffers will be refreshed during invocation.
  // TODO(b/156094015): Explore whether this is actually faster than
  // using ByteBuffer.put(ByteBuffer).
  memcpy(tensor->data.data, src_data_raw, tensor->bytes);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Tensor_readMultiDimensionalArray(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle,
                                                          jobject value) {
  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return;
  int num_dims = tensor->dims->size;
  if (num_dims == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Cannot copy empty/scalar Tensors.");
    return;
  }
  if (tensor->type == kTfLiteString) {
    ReadMultiDimensionalStringArray(env, tensor, num_dims, 0,
                                    static_cast<jarray>(value));
  } else {
    ReadMultiDimensionalArray(env, tensor->type, tensor->data.raw,
                              tensor->bytes, num_dims,
                              static_cast<jarray>(value));
  }
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_Tensor_writeMultiDimensionalArray(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle,
                                                           jobject src) {
  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return;
  if (tensor->type != kTfLiteString && tensor->data.raw == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Target Tensor hasn't been allocated.");
    return;
  }
  if (tensor->dims->size == 0) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Cannot copy empty/scalar Tensors.");
    return;
  }
  if (tensor->type == kTfLiteString) {
    WriteMultiDimensionalStringArray(env, src, tensor);
  } else {
    WriteMultiDimensionalArray(env, src, tensor->type, tensor->dims->size,
                               &tensor->data.raw, tensor->bytes);
  }
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_Tensor_writeScalar(
    JNIEnv* env, jclass clazz, jlong handle, jobject src) {
  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return;
  if ((tensor->type != kTfLiteString) && (tensor->data.raw == nullptr)) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Target Tensor hasn't been allocated.");
    return;
  }
  if ((tensor->dims->size != 0) && (tensor->dims->data[0] != 1)) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Cannot write Java scalar to non-scalar "
                   "Tensor.");
    return;
  }
  if (tensor->type == kTfLiteString) {
    WriteScalarString(env, src, tensor);
  } else {
    WriteScalar(env, src, tensor->type, tensor->data.data, tensor->bytes);
  }
}

JNIEXPORT jint JNICALL Java_org_tensorflow_lite_Tensor_dtype(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle) {
  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return 0;
  return static_cast<jint>(tensor->type);
}

JNIEXPORT jstring JNICALL Java_org_tensorflow_lite_Tensor_name(JNIEnv* env,
                                                               jclass clazz,
                                                               jlong handle) {
  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Target Tensor doesn't exist.");
    return nullptr;
  }

  if (tensor->name == nullptr) {
    return env->NewStringUTF("");
  }

  jstring tensor_name = env->NewStringUTF(tensor->name);
  if (tensor_name == nullptr) {
    return env->NewStringUTF("");
  }
  return tensor_name;
}

JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_Tensor_shape(JNIEnv* env, jclass clazz, jlong handle) {
  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return nullptr;
  int num_dims = tensor->dims->size;
  jintArray result = env->NewIntArray(num_dims);
  env->SetIntArrayRegion(result, 0, num_dims, tensor->dims->data);
  return result;
}

JNIEXPORT jintArray JNICALL Java_org_tensorflow_lite_Tensor_shapeSignature(
    JNIEnv* env, jclass clazz, jlong handle) {
  TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return nullptr;

  int num_dims = 0;
  int const* data = nullptr;
  if (tensor->dims_signature != nullptr && tensor->dims_signature->size != 0) {
    num_dims = tensor->dims_signature->size;
    data = tensor->dims_signature->data;
  } else {
    num_dims = tensor->dims->size;
    data = tensor->dims->data;
  }
  jintArray result = env->NewIntArray(num_dims);
  env->SetIntArrayRegion(result, 0, num_dims, data);
  return result;
}

JNIEXPORT jint JNICALL Java_org_tensorflow_lite_Tensor_numBytes(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  const TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return 0;
  return static_cast<jint>(tensor->bytes);
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_Tensor_hasDelegateBufferHandle(JNIEnv* env,
                                                        jclass clazz,
                                                        jlong handle) {
  const TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  if (tensor == nullptr) return false;
  return tensor->delegate && (tensor->buffer_handle != kTfLiteNullBufferHandle)
             ? JNI_TRUE
             : JNI_FALSE;
}

JNIEXPORT jint JNICALL Java_org_tensorflow_lite_Tensor_index(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle) {
  return GetTensorIndexFromHandle(env, handle);
}

JNIEXPORT jfloat JNICALL Java_org_tensorflow_lite_Tensor_quantizationScale(
    JNIEnv* env, jclass clazz, jlong handle) {
  const TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  return static_cast<jfloat>(tensor ? tensor->params.scale : 0.f);
}

JNIEXPORT jint JNICALL Java_org_tensorflow_lite_Tensor_quantizationZeroPoint(
    JNIEnv* env, jclass clazz, jlong handle) {
  const TfLiteTensor* tensor = GetTensorFromHandle(env, handle);
  return static_cast<jint>(tensor ? tensor->params.zero_point : 0);
}

}  // extern "C"
