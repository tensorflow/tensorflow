/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/java/src/main/native/eager_operation_builder_jni.h"

#include <cstring>
#include <memory>
#include <set>

#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"

// This value should be >= to the maximum number of outputs in any op
#define MAX_OUTPUTS_PER_OP 8

namespace {

TFE_Op* requireOp(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "Operation has already been built");
    return nullptr;
  }
  return reinterpret_cast<TFE_Op*>(handle);
}

TFE_Context* requireContext(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalStateException, "Context has been deleted");
    return nullptr;
  }
  return reinterpret_cast<TFE_Context*>(handle);
}

TF_Tensor* requireTensor(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "close() has been called on the Tensor");
    return nullptr;
  }
  return reinterpret_cast<TF_Tensor*>(handle);
}

TFE_TensorHandle* requireTensorHandle(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalStateException,
                   "Tensor handle has been deleted");
    return nullptr;
  }
  return reinterpret_cast<TFE_TensorHandle*>(handle);
}

}  // namespace

JNIEXPORT jlong JNICALL Java_org_tensorflow_EagerOperationBuilder_allocate(
    JNIEnv* env, jclass clazz, jlong context_handle, jstring name) {
  TFE_Context* context = requireContext(env, context_handle);
  if (context == nullptr) return 0;
  const char* op_or_function_name = env->GetStringUTFChars(name, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_Op* op = TFE_NewOp(context, op_or_function_name, status);
  env->ReleaseStringUTFChars(name, op_or_function_name);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return 0;
  }
  TF_DeleteStatus(status);
  static_assert(sizeof(jlong) >= sizeof(TFE_Op*),
                "Cannot represent a C TFE_Op as a Java long");
  return reinterpret_cast<jlong>(op);
}

JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_delete(
    JNIEnv* env, jclass clazz, jlong op_handle) {
  if (op_handle == 0) return;
  TFE_DeleteOp(reinterpret_cast<TFE_Op*>(op_handle));
}

JNIEXPORT jlongArray JNICALL Java_org_tensorflow_EagerOperationBuilder_execute(
    JNIEnv* env, jclass clazz, jlong op_handle) {
  TFE_Op* op = requireOp(env, op_handle);
  if (op == nullptr) return 0;
  int num_retvals = MAX_OUTPUTS_PER_OP;
  std::unique_ptr<TFE_TensorHandle*[]> retvals(
      new TFE_TensorHandle*[num_retvals]);
  TF_Status* status = TF_NewStatus();
  TFE_Execute(op, retvals.get(), &num_retvals, status);
  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  TF_DeleteStatus(status);
  jlongArray rethandles = env->NewLongArray(num_retvals);
  if (num_retvals > 0) {
    jlong* retval = env->GetLongArrayElements(rethandles, nullptr);
    for (int i = 0; i < num_retvals; ++i) {
      retval[i] = reinterpret_cast<jlong>(retvals[i]);
    }
    env->ReleaseLongArrayElements(rethandles, retval, 0);
  }
  return rethandles;
}

JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setDevice(
    JNIEnv* env, jclass clazz, jlong op_handle, jstring device_name) {
  TFE_Op* op = requireOp(env, op_handle);
  if (op == nullptr) return;
  const char* cname = env->GetStringUTFChars(device_name, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_OpSetDevice(op, cname, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  env->ReleaseStringUTFChars(device_name, cname);
}

JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_addInput(
    JNIEnv* env, jclass clazz, jlong op_handle, jlong input_handle) {
  TFE_Op* op = requireOp(env, op_handle);
  if (op == nullptr) return;
  TFE_TensorHandle* tensor_handle = requireTensorHandle(env, input_handle);
  if (tensor_handle == nullptr) return;
  TF_Status* status = TF_NewStatus();
  TFE_OpAddInput(op, tensor_handle, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_addInputList(
    JNIEnv* env, jclass clazz, jlong op_handle, jlongArray input_handles) {
  TFE_Op* op = requireOp(env, op_handle);
  if (op == nullptr) return;
  jlong* cinput_handles = env->GetLongArrayElements(input_handles, nullptr);
  size_t num_inputs = static_cast<size_t>(env->GetArrayLength(input_handles));
  std::unique_ptr<TFE_TensorHandle*[]> tensor_handles(
      new TFE_TensorHandle*[num_inputs]);
  for (int i = 0; i < num_inputs; ++i) {
    tensor_handles[i] = requireTensorHandle(env, cinput_handles[i]);
    if (tensor_handles[i] == nullptr) {
      env->ReleaseLongArrayElements(input_handles, cinput_handles, JNI_ABORT);
      return;
    }
  }
  env->ReleaseLongArrayElements(input_handles, cinput_handles, JNI_ABORT);
  TF_Status* status = TF_NewStatus();
  TFE_OpAddInputList(op, tensor_handles.get(), num_inputs, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrString(
    JNIEnv* env, jclass clazz, jlong op_handle, jstring attr_name,
    jbyteArray value) {
  static_assert(sizeof(jbyte) == 1,
                "Require Java byte to be represented as a single byte");
  TFE_Op* op = requireOp(env, op_handle);
  if (op == nullptr) return;
  const char* cname = env->GetStringUTFChars(attr_name, nullptr);
  jbyte* cvalue = env->GetByteArrayElements(value, nullptr);
  TFE_OpSetAttrString(op, cname, cvalue, env->GetArrayLength(value));
  env->ReleaseByteArrayElements(value, cvalue, JNI_ABORT);
  env->ReleaseStringUTFChars(attr_name, cname);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_EagerOperationBuilder_setAttrStringList(
    JNIEnv* env, jclass object, jlong op_handle, jstring attr_name,
    jobjectArray values) {
  TFE_Op* op = requireOp(env, op_handle);
  if (op == nullptr) return;
  const char* cname = env->GetStringUTFChars(attr_name, nullptr);
  int num_values = env->GetArrayLength(values);
  static_assert(sizeof(jbyte) == 1,
                "Require Java byte to be represented as a single byte");
  std::unique_ptr<jbyteArray[]> jarrays(new jbyteArray[num_values]);
  std::unique_ptr<jbyte*[]> jvalues(new jbyte*[num_values]);
  std::unique_ptr<void*[]> cvalues(new void*[num_values]);
  std::unique_ptr<size_t[]> lengths(new size_t[num_values]);

  for (int i = 0; i < num_values; ++i) {
    jbyteArray v =
        static_cast<jbyteArray>(env->GetObjectArrayElement(values, i));
    jarrays[i] = v;
    jvalues[i] = env->GetByteArrayElements(v, nullptr);
    cvalues[i] = jvalues[i];
    lengths[i] = static_cast<size_t>(env->GetArrayLength(v));
  }
  TFE_OpSetAttrStringList(op, cname, cvalues.get(), lengths.get(), num_values);
  for (int i = 0; i < num_values; ++i) {
    env->ReleaseByteArrayElements(jarrays[i], jvalues[i], JNI_ABORT);
  }
  env->ReleaseStringUTFChars(attr_name, cname);
}

#define DEFINE_SET_ATTR_SCALAR(name, jtype, ctype)                       \
  JNIEXPORT void JNICALL                                                 \
      Java_org_tensorflow_EagerOperationBuilder_setAttr##name(           \
          JNIEnv* env, jclass clazz, jlong op_handle, jstring attr_name, \
          jtype value) {                                                 \
    static_assert(                                                       \
        sizeof(ctype) >= sizeof(jtype),                                  \
        "Information loss when converting between Java and C types");    \
    TFE_Op* op = requireOp(env, op_handle);                              \
    if (op == nullptr) return;                                           \
    const char* cname = env->GetStringUTFChars(attr_name, nullptr);      \
    TFE_OpSetAttr##name(op, cname, static_cast<ctype>(value));           \
    env->ReleaseStringUTFChars(attr_name, cname);                        \
  }

#define DEFINE_SET_ATTR_LIST(name, jname, jtype, ctype)                  \
  JNIEXPORT void JNICALL                                                 \
      Java_org_tensorflow_EagerOperationBuilder_setAttr##name##List(     \
          JNIEnv* env, jclass clazz, jlong op_handle, jstring attr_name, \
          jtype##Array value) {                                          \
    TFE_Op* op = requireOp(env, op_handle);                              \
    if (op == nullptr) return;                                           \
    const char* cname = env->GetStringUTFChars(attr_name, nullptr);      \
    /* Make a copy of the array to paper over any differences */         \
    /* in byte representations of the jtype and ctype */                 \
    /* For example, jint vs TF_DataType. */                              \
    /* If this copy turns out to be a problem in practice */             \
    /* can avoid it for many types. */                                   \
    const int n = env->GetArrayLength(value);                            \
    std::unique_ptr<ctype[]> cvalue(new ctype[n]);                       \
    jtype* elems = env->Get##jname##ArrayElements(value, nullptr);       \
    for (int i = 0; i < n; ++i) {                                        \
      cvalue[i] = static_cast<ctype>(elems[i]);                          \
    }                                                                    \
    TFE_OpSetAttr##name##List(op, cname, cvalue.get(), n);               \
    env->Release##jname##ArrayElements(value, elems, JNI_ABORT);         \
    env->ReleaseStringUTFChars(attr_name, cname);                        \
  }

#define DEFINE_SET_ATTR(name, jname, jtype, ctype) \
  DEFINE_SET_ATTR_SCALAR(name, jtype, ctype)       \
  DEFINE_SET_ATTR_LIST(name, jname, jtype, ctype)

DEFINE_SET_ATTR(Int, Long, jlong, int64_t);
DEFINE_SET_ATTR(Float, Float, jfloat, float);
DEFINE_SET_ATTR(Bool, Boolean, jboolean, unsigned char);
DEFINE_SET_ATTR(Type, Int, jint, TF_DataType);
#undef DEFINE_SET_ATTR
#undef DEFINE_SET_ATTR_LIST
#undef DEFINE_SET_ATTR_SCALAR

JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrTensor(
    JNIEnv* env, jclass clazz, jlong handle, jstring attr_name,
    jlong tensor_handle) {
  TFE_Op* op = requireOp(env, handle);
  if (op == nullptr) return;
  TF_Tensor* t = requireTensor(env, tensor_handle);
  if (t == nullptr) return;
  const char* cname = env->GetStringUTFChars(attr_name, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_OpSetAttrTensor(op, cname, t, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  env->ReleaseStringUTFChars(attr_name, cname);
}

JNIEXPORT void JNICALL Java_org_tensorflow_EagerOperationBuilder_setAttrShape(
    JNIEnv* env, jclass clazz, jlong op_handle, jstring attr_name,
    jlongArray shape, jint num_dims) {
  TFE_Op* op = requireOp(env, op_handle);
  if (op == nullptr) return;
  std::unique_ptr<int64_t[]> cvalue;
  // num_dims and env->GetArrayLength(shape) are assumed to be consistent.
  // i.e., either num_dims < 0 or num_dims == env->GetArrayLength(shape).
  if (num_dims > 0) {
    cvalue.reset(new int64_t[num_dims]);
    jlong* elems = env->GetLongArrayElements(shape, nullptr);
    for (int i = 0; i < num_dims; ++i) {
      cvalue[i] = static_cast<int64_t>(elems[i]);
    }
    env->ReleaseLongArrayElements(shape, elems, JNI_ABORT);
  }
  const char* cname = env->GetStringUTFChars(attr_name, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_OpSetAttrShape(op, cname, cvalue.get(), static_cast<int>(num_dims),
                     status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  env->ReleaseStringUTFChars(attr_name, cname);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_EagerOperationBuilder_setAttrShapeList(
    JNIEnv* env, jclass clazz, jlong op_handle, jstring attr_name,
    jlongArray shapes, jintArray num_dims) {
  TFE_Op* op = requireOp(env, op_handle);
  if (op == nullptr) return;
  std::unique_ptr<int64_t[]> cshapes;
  std::unique_ptr<const int64_t*[]> cdims;
  std::unique_ptr<int[]> cnum_dims;
  const int num_dims_length = env->GetArrayLength(num_dims);
  if (num_dims_length > 0) {
    const int shapes_length = env->GetArrayLength(shapes);
    cshapes.reset(new int64_t[shapes_length]);
    cdims.reset(new const int64_t*[num_dims_length]);
    cnum_dims.reset(new int[num_dims_length]);
    jlong* shapes_elems =
        static_cast<jlong*>(env->GetPrimitiveArrayCritical(shapes, nullptr));
    std::memcpy(cshapes.get(), shapes_elems, shapes_length << 3);
    env->ReleasePrimitiveArrayCritical(shapes, shapes_elems, JNI_ABORT);
    int64_t* cshapes_ptr = cshapes.get();
    jint* num_dims_elems =
        static_cast<jint*>(env->GetPrimitiveArrayCritical(num_dims, nullptr));
    for (int i = 0; i < num_dims_length; ++i) {
      cnum_dims[i] = static_cast<int>(num_dims_elems[i]);
      cdims[i] = cshapes_ptr;
      if (cnum_dims[i] > 0) {
        cshapes_ptr += cnum_dims[i];
      }
    }
    env->ReleasePrimitiveArrayCritical(num_dims, num_dims_elems, JNI_ABORT);
  }
  const char* cname = env->GetStringUTFChars(attr_name, nullptr);
  TF_Status* status = TF_NewStatus();
  TFE_OpSetAttrShapeList(op, cname, cdims.get(), cnum_dims.get(),
                         num_dims_length, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
  env->ReleaseStringUTFChars(attr_name, cname);
}
