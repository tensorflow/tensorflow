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

#include <dlfcn.h>
#include <jni.h>
#include <stdio.h>
#include <time.h>

#include <atomic>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/create_op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/verifier_internal.h"
#if TFLITE_DISABLE_SELECT_JAVA_APIS
#include "tensorflow/lite/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/acceleration/configuration/c/xnnpack_plugin.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"
#else
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif
#include "tensorflow/lite/java/src/main/native/jni_utils.h"
#include "tensorflow/lite/java/src/main/native/op_resolver_lazy_delegate_proxy.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/util.h"

using tflite::FlatBufferModel;
using tflite::Interpreter;
using tflite::InterpreterBuilder;
using tflite::OpResolver;
using tflite::jni::AreDimsDifferent;
using tflite::jni::BufferErrorReporter;
using tflite::jni::CastLongToPointer;
using tflite::jni::ConvertJIntArrayToVector;
using tflite::jni::ThrowException;

namespace {

Interpreter* convertLongToInterpreter(JNIEnv* env, jlong handle) {
  return CastLongToPointer<Interpreter>(env, handle);
}

FlatBufferModel* convertLongToModel(JNIEnv* env, jlong handle) {
  return CastLongToPointer<FlatBufferModel>(env, handle);
}

BufferErrorReporter* convertLongToErrorReporter(JNIEnv* env, jlong handle) {
  return CastLongToPointer<BufferErrorReporter>(env, handle);
}

int getDataType(TfLiteType data_type) {
  switch (data_type) {
    case kTfLiteFloat32:
      return 1;
    case kTfLiteInt32:
      return 2;
    case kTfLiteUInt8:
      return 3;
    case kTfLiteInt64:
      return 4;
    case kTfLiteString:
      return 5;
    case kTfLiteBool:
      return 6;
    default:
      return -1;
  }
}

// TODO(yichengfan): evaluate the benefit to use tflite verifier.
bool VerifyModel(const void* buf, size_t length) {
  return tflite::internal::VerifyFlatBufferAndGetModel(buf, length);
}

// Verifies whether the model is a flatbuffer file.
class JNIFlatBufferVerifier : public tflite::TfLiteVerifier {
 public:
  bool Verify(const char* data, int length,
              tflite::ErrorReporter* reporter) override {
    if (!VerifyModel(data, length)) {
      TF_LITE_REPORT_ERROR(reporter,
                           "The model is not a valid Flatbuffer file");
      return false;
    }
    return true;
  }
};

// Like JNIEnv's FindClass method, but converts the result to a
// JNI global reference rather than returning a local reference.
jclass FindClassAndMakeGlobalRef(JNIEnv* env, const char* class_name) {
  jclass local_ref = env->FindClass(class_name);
  jclass global_ref = static_cast<jclass>(env->NewGlobalRef(local_ref));
  env->DeleteLocalRef(local_ref);
  return global_ref;
}

}  // namespace

extern "C" {

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputNames(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return nullptr;

  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  static jclass string_class =
      FindClassAndMakeGlobalRef(env, "java/lang/String");
  if (string_class == nullptr) {
    if (!env->ExceptionCheck()) {
      ThrowException(env, tflite::jni::kUnsupportedOperationException,
                     "Internal error: Can not find java/lang/String class to "
                     "get input names.");
    }
    return nullptr;
  }
  size_t size = interpreter->inputs().size();
  jobjectArray names = static_cast<jobjectArray>(
      env->NewObjectArray(size, string_class, env->NewStringUTF("")));
  for (int i = 0; i < size; ++i) {
    env->SetObjectArrayElement(names, i,
                               env->NewStringUTF(interpreter->GetInputName(i)));
  }
  return names;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allocateTensors(
    JNIEnv* env, jclass clazz, jlong handle, jlong error_handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return;

  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ThrowException(env, tflite::jni::kIllegalStateException,
                   "Internal error: Unexpected failure when preparing tensor "
                   "allocations: %s",
                   error_reporter->CachedErrorMessage());
  }
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_hasUnresolvedFlexOp(
    JNIEnv* env, jclass clazz, jlong handle) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  TFLITE_LOG(tflite::TFLITE_LOG_WARNING, "Not supported: hasUnresolvedFlexOp");
  return JNI_FALSE;
#else
  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return JNI_FALSE;

  // TODO(b/132995737): Remove this logic by caching whether an unresolved
  // Flex op is present during Interpreter creation.
  for (size_t subgraph_i = 0; subgraph_i < interpreter->subgraphs_size();
       ++subgraph_i) {
    const auto* subgraph = interpreter->subgraph(static_cast<int>(subgraph_i));
    for (int node_i : subgraph->execution_plan()) {
      const auto& registration =
          subgraph->node_and_registration(node_i)->second;
      if (tflite::IsUnresolvedCustomOp(registration) &&
          tflite::IsFlexOp(registration.custom_name)) {
        return JNI_TRUE;
      }
    }
  }
  return JNI_FALSE;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getSignatureKeys(
    JNIEnv* env, jclass clazz, jlong handle) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  TFLITE_LOG(tflite::TFLITE_LOG_WARNING, "Not supported: getSignatureKeys");
  return nullptr;
#else
  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  static jclass string_class =
      FindClassAndMakeGlobalRef(env, "java/lang/String");
  if (string_class == nullptr) {
    if (!env->ExceptionCheck()) {
      ThrowException(env, tflite::jni::kUnsupportedOperationException,
                     "Internal error: Can not find java/lang/String class to "
                     "get SignatureDef keys.");
    }
    return nullptr;
  }
  const auto& signature_keys = interpreter->signature_keys();
  jobjectArray keys = static_cast<jobjectArray>(env->NewObjectArray(
      signature_keys.size(), string_class, env->NewStringUTF("")));
  for (int i = 0; i < signature_keys.size(); ++i) {
    env->SetObjectArrayElement(keys, i,
                               env->NewStringUTF(signature_keys[i]->c_str()));
  }
  return keys;
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputTensorIndex(
    JNIEnv* env, jclass clazz, jlong handle, jint input_index) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return interpreter->inputs()[input_index];
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputTensorIndex(
    JNIEnv* env, jclass clazz, jlong handle, jint output_index) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return interpreter->outputs()[output_index];
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getExecutionPlanLength(
    JNIEnv* env, jclass clazz, jlong handle) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: getExecutionPlanLength");
  return -1;
#else
  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return static_cast<jint>(interpreter->execution_plan().size());
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputCount(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return static_cast<jint>(interpreter->inputs().size());
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputCount(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return static_cast<jint>(interpreter->outputs().size());
}

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputNames(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return nullptr;

  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  static jclass string_class =
      FindClassAndMakeGlobalRef(env, "java/lang/String");
  if (string_class == nullptr) {
    if (!env->ExceptionCheck()) {
      ThrowException(env, tflite::jni::kUnsupportedOperationException,
                     "Internal error: Can not find java/lang/String class to "
                     "get output names.");
    }
    return nullptr;
  }
  size_t size = interpreter->outputs().size();
  jobjectArray names = static_cast<jobjectArray>(
      env->NewObjectArray(size, string_class, env->NewStringUTF("")));
  for (int i = 0; i < size; ++i) {
    env->SetObjectArrayElement(
        names, i, env->NewStringUTF(interpreter->GetOutputName(i)));
  }
  return names;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allowFp16PrecisionForFp32(
    JNIEnv* env, jclass clazz, jlong handle, jboolean allow) {
  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  if (allow) {
    ThrowException(env, tflite::jni::kUnsupportedOperationException,
                   "Not supported: SetAllowFp16PrecisionForFp32(true)");
  }
#else
  interpreter->SetAllowFp16PrecisionForFp32(static_cast<bool>(allow));
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allowBufferHandleOutput(
    JNIEnv* env, jclass clazz, jlong handle, jboolean allow) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  if (allow) {
    ThrowException(env, tflite::jni::kUnsupportedOperationException,
                   "Not supported: allowBufferHandleOutput(true)");
  }
#else
  Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  interpreter->SetAllowBufferHandleOutput(allow);
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createErrorReporter(
    JNIEnv* env, jclass clazz, jint size) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

  BufferErrorReporter* error_reporter =
      new BufferErrorReporter(env, static_cast<int>(size));
  return reinterpret_cast<jlong>(error_reporter);
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createModel(
    JNIEnv* env, jclass clazz, jstring model_file, jlong error_handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;
  const char* path = env->GetStringUTFChars(model_file, nullptr);

  std::unique_ptr<tflite::TfLiteVerifier> verifier;
  verifier = std::make_unique<JNIFlatBufferVerifier>();

  auto model = FlatBufferModel::VerifyAndBuildFromFile(path, verifier.get(),
                                                       error_reporter);
  if (!model) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Contents of %s does not encode a valid "
                   "TensorFlow Lite model: %s",
                   path, error_reporter->CachedErrorMessage());
    env->ReleaseStringUTFChars(model_file, path);
    return 0;
  }
  env->ReleaseStringUTFChars(model_file, path);
  return reinterpret_cast<jlong>(model.release());
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createModelWithBuffer(
    JNIEnv* env, jclass /*clazz*/, jobject model_buffer, jlong error_handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;
  const char* buf =
      static_cast<char*>(env->GetDirectBufferAddress(model_buffer));
  jlong capacity = env->GetDirectBufferCapacity(model_buffer);
  if (!VerifyModel(buf, capacity)) {
    ThrowException(
        env, tflite::jni::kIllegalArgumentException,
        "ByteBuffer is not a valid TensorFlow Lite model flatbuffer");
    return 0;
  }

  auto model = FlatBufferModel::BuildFromBuffer(
      buf, static_cast<size_t>(capacity), error_reporter);
  if (!model) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "ByteBuffer does not encode a valid model: %s",
                   error_reporter->CachedErrorMessage());
    return 0;
  }
  return reinterpret_cast<jlong>(model.release());
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createInterpreter(
    JNIEnv* env, jclass clazz, jlong model_handle, jlong error_handle,
    jint num_threads, jboolean useXnnpack, jobject delegate_handle_list) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return 0;

  static jclass list_class = FindClassAndMakeGlobalRef(env, "java/util/List");
  if (list_class == nullptr) {
    if (!env->ExceptionCheck()) {
      ThrowException(env, tflite::jni::kUnsupportedOperationException,
                     "Internal error: Can't find java.util.List class.");
    }
    return 0;
  }
  static jmethodID list_size_method =
      env->GetMethodID(list_class, "size", "()I");
  if (list_size_method == nullptr) {
    if (!env->ExceptionCheck()) {
      ThrowException(env, tflite::jni::kUnsupportedOperationException,
                     "Internal error: Can't find java.util.List.size method.");
    }
    return 0;
  }
  static jmethodID list_get_method =
      env->GetMethodID(list_class, "get", "(I)Ljava/lang/Object;");
  if (list_get_method == nullptr) {
    if (!env->ExceptionCheck()) {
      ThrowException(env, tflite::jni::kUnsupportedOperationException,
                     "Internal error: Can't find java.util.List.get method.");
    }
    return 0;
  }
  static jclass long_class = FindClassAndMakeGlobalRef(env, "java/lang/Long");
  if (long_class == nullptr) {
    if (!env->ExceptionCheck()) {
      ThrowException(env, tflite::jni::kUnsupportedOperationException,
                     "Internal error: "
                     "Can't find java.lang.Long class.");
    }
    return 0;
  }
  static jmethodID long_value_method =
      env->GetMethodID(long_class, "longValue", "()J");
  if (long_value_method == nullptr) {
    if (!env->ExceptionCheck()) {
      ThrowException(env, tflite::jni::kUnsupportedOperationException,
                     "Internal error: "
                     "Can't find java.lang.Long longValue method.");
    }
    return 0;
  }

  FlatBufferModel* model = convertLongToModel(env, model_handle);
  if (model == nullptr) return 0;

  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;

  std::unique_ptr<OpResolver> resolver =
      std::make_unique<tflite::jni::OpResolverLazyDelegateProxy>(
          tflite::CreateOpResolver(), useXnnpack != JNI_FALSE);

  InterpreterBuilder interpreter_builder(*model, *resolver);
  interpreter_builder.SetNumThreads(static_cast<int>(num_threads));

  // Add delegate_list to interpreter_builder.

  // Java: int size = delegate_list.size();
  jint size = env->CallIntMethod(delegate_handle_list, list_size_method);
  for (jint i = 0; i < size; ++i) {
    // Java: Long jdelegate_handle = delegate_handle_list->get(i);
    jobject jdelegate_handle =
        env->CallObjectMethod(delegate_handle_list, list_get_method, i);
    if (jdelegate_handle == nullptr) {
      if (!env->ExceptionCheck()) {
        ThrowException(env, tflite::jni::kIllegalArgumentException,
                       "Internal error: null object in Delegate handle list");
      }
      return 0;
    }
    // Java: long delegate_handle = jdelegate_handle.longValue();
    jlong delegate_handle =
        env->CallLongMethod(jdelegate_handle, long_value_method);
    if (delegate_handle == 0) {
      if (!env->ExceptionCheck()) {
        ThrowException(env, tflite::jni::kIllegalArgumentException,
                       "Internal error: Found invalid handle");
      }
      return 0;
    }
    auto delegate = reinterpret_cast<TfLiteOpaqueDelegate*>(delegate_handle);
    interpreter_builder.AddDelegate(delegate);
  }

  // Create the Interpreter.
  std::unique_ptr<Interpreter> interpreter;
  TfLiteStatus status = interpreter_builder(&interpreter);
  if (status != kTfLiteOk) {
    if (status == kTfLiteDelegateError) {
      ThrowException(env, tflite::jni::kIllegalArgumentException,
                     "Internal error: Failed to apply delegate: %s",
                     error_reporter->CachedErrorMessage());
    } else if (status == kTfLiteApplicationError) {
      ThrowException(env, tflite::jni::kIllegalArgumentException,
                     "Internal error: Error applying delegate: %s",
                     error_reporter->CachedErrorMessage());
    } else {
      const char* error_message = error_reporter->CachedErrorMessage();
      if (std::strcmp(
              error_message,
              "Restored original execution plan after delegate application "
              "failure.") == 0 ||
          std::strcmp(
              error_message,
              "Restored original execution plan after delegate application "
              "failure.\n"
              "Restored original execution plan after delegate application "
              "failure.") == 0) {
        ThrowException(env, tflite::jni::kIllegalArgumentException,
                       "Internal error: Failed to apply delegate.");
      } else {
        ThrowException(env, tflite::jni::kIllegalArgumentException,
                       "Internal error: Cannot create interpreter: %s",
                       error_message);
      }
    }
    return 0;
  }

  // Note that tensor allocation is performed explicitly by the owning Java
  // NativeInterpreterWrapper instance.
  return reinterpret_cast<jlong>(interpreter.release());
}

// Sets inputs, runs inference, and returns outputs as long handles.
JNIEXPORT void JNICALL Java_org_tensorflow_lite_NativeInterpreterWrapper_run(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return;

  Interpreter* interpreter = convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return;
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return;

  if (interpreter->Invoke() != kTfLiteOk) {
    // TODO(b/168266570): Return InterruptedException.
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Failed to run on the given Interpreter: %s",
                   error_reporter->CachedErrorMessage());
    return;
  }
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_resizeInput(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle,
    jint input_idx, jintArray dims, jboolean strict) {
  if (!tflite::jni::CheckJniInitializedOrThrow(env)) return JNI_FALSE;

  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return JNI_FALSE;
  Interpreter* interpreter = convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return JNI_FALSE;
  if (input_idx < 0 || input_idx >= interpreter->inputs().size()) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Input error: Can not resize %d-th input for a model having "
                   "%d inputs.",
                   input_idx, interpreter->inputs().size());
    return JNI_FALSE;
  }
  const int tensor_idx = interpreter->inputs()[input_idx];
  // check whether it is resizing with the same dimensions.
  TfLiteTensor* target = interpreter->tensor(tensor_idx);
  bool is_changed = AreDimsDifferent(env, target, dims);
  if (is_changed) {
    TfLiteStatus status;
    if (strict) {
      status = interpreter->ResizeInputTensorStrict(
          tensor_idx, ConvertJIntArrayToVector(env, dims));
    } else {
      status = interpreter->ResizeInputTensor(
          tensor_idx, ConvertJIntArrayToVector(env, dims));
    }
    if (status != kTfLiteOk) {
      ThrowException(env, tflite::jni::kIllegalArgumentException,
                     "Internal error: Failed to resize %d-th input: %s",
                     input_idx, error_reporter->CachedErrorMessage());
      return JNI_FALSE;
    }
  }
  return is_changed ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createCancellationFlag(
    JNIEnv* env, jclass clazz, jlong interpreter_handle) {
  Interpreter* interpreter = convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) {
    ThrowException(env, tflite::jni::kIllegalArgumentException,
                   "Internal error: Invalid handle to interpreter.");
    return 0;
  }
  std::atomic_bool* cancellation_flag = new std::atomic_bool(false);
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: cancellation");
#else
  interpreter->SetCancellationFunction(cancellation_flag, [](void* payload) {
    std::atomic_bool* cancellation_flag =
        reinterpret_cast<std::atomic_bool*>(payload);
    return cancellation_flag->load() == true;
  });
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
  return reinterpret_cast<jlong>(cancellation_flag);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_deleteCancellationFlag(
    JNIEnv* env, jclass clazz, jlong flag_handle) {
  std::atomic_bool* cancellation_flag =
      reinterpret_cast<std::atomic_bool*>(flag_handle);
  delete cancellation_flag;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_setCancelled(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong flag_handle,
    jboolean value) {
#if TFLITE_DISABLE_SELECT_JAVA_APIS
  ThrowException(env, tflite::jni::kUnsupportedOperationException,
                 "Not supported: cancellation");
#else
  std::atomic_bool* cancellation_flag =
      reinterpret_cast<std::atomic_bool*>(flag_handle);
  if (cancellation_flag != nullptr) {
    cancellation_flag->store(static_cast<bool>(value));
  }
#endif  // TFLITE_DISABLE_SELECT_JAVA_APIS
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_NativeInterpreterWrapper_delete(
    JNIEnv* env, jclass clazz, jlong error_handle, jlong model_handle,
    jlong interpreter_handle) {
  if (interpreter_handle != 0) {
    delete convertLongToInterpreter(env, interpreter_handle);
  }
  if (model_handle != 0) {
    delete convertLongToModel(env, model_handle);
  }
  if (error_handle != 0) {
    delete convertLongToErrorReporter(env, error_handle);
  }
}

}  // extern "C"
