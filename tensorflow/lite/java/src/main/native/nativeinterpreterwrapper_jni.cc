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
#include <stdio.h>
#include <time.h>

#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/tflite_api_dispatcher/tflite_api_dispatcher.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"
#include "tensorflow/lite/util.h"

namespace tflite {
// This is to be provided at link-time by a library.
extern std::unique_ptr<OpResolver> CreateOpResolver();
}  // namespace tflite

using tflite::jni::BufferErrorReporter;
using tflite::jni::ThrowException;

namespace {

tflite_api_dispatcher::Interpreter* convertLongToInterpreter(JNIEnv* env,
                                                             jlong handle) {
  if (handle == 0) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to Interpreter.");
    return nullptr;
  }
  return reinterpret_cast<tflite_api_dispatcher::Interpreter*>(handle);
}

tflite_api_dispatcher::TfLiteModel* convertLongToModel(JNIEnv* env,
                                                       jlong handle) {
  if (handle == 0) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to model.");
    return nullptr;
  }
  return reinterpret_cast<tflite_api_dispatcher::TfLiteModel*>(handle);
}

BufferErrorReporter* convertLongToErrorReporter(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to ErrorReporter.");
    return nullptr;
  }
  return reinterpret_cast<BufferErrorReporter*>(handle);
}

TfLiteDelegate* convertLongToDelegate(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to delegate.");
    return nullptr;
  }
  return reinterpret_cast<TfLiteDelegate*>(handle);
}

std::vector<int> convertJIntArrayToVector(JNIEnv* env, jintArray inputs) {
  int size = static_cast<int>(env->GetArrayLength(inputs));
  std::vector<int> outputs(size, 0);
  jint* ptr = env->GetIntArrayElements(inputs, nullptr);
  if (ptr == nullptr) {
    ThrowException(env, kIllegalArgumentException,
                   "Array has empty dimensions.");
    return {};
  }
  for (int i = 0; i < size; ++i) {
    outputs[i] = ptr[i];
  }
  env->ReleaseIntArrayElements(inputs, ptr, JNI_ABORT);
  return outputs;
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
    default:
      return -1;
  }
}

void printDims(char* buffer, int max_size, int* dims, int num_dims) {
  if (max_size <= 0) return;
  buffer[0] = '?';
  int size = 1;
  for (int i = 1; i < num_dims; ++i) {
    if (max_size > size) {
      int written_size =
          snprintf(buffer + size, max_size - size, ",%d", dims[i]);
      if (written_size < 0) return;
      size += written_size;
    }
  }
}

// Checks whether there is any difference between dimensions of a tensor and a
// given dimensions. Returns true if there is difference, else false.
bool AreDimsDifferent(JNIEnv* env, TfLiteTensor* tensor, jintArray dims) {
  int num_dims = static_cast<int>(env->GetArrayLength(dims));
  jint* ptr = env->GetIntArrayElements(dims, nullptr);
  if (ptr == nullptr) {
    ThrowException(env, kIllegalArgumentException,
                   "Empty dimensions of input array.");
    return true;
  }
  bool is_different = false;
  if (tensor->dims->size != num_dims) {
    is_different = true;
  } else {
    for (int i = 0; i < num_dims; ++i) {
      if (ptr[i] != tensor->dims->data[i]) {
        is_different = true;
        break;
      }
    }
  }
  env->ReleaseIntArrayElements(dims, ptr, JNI_ABORT);
  return is_different;
}

// TODO(yichengfan): evaluate the benefit to use tflite verifier.
bool VerifyModel(const void* buf, size_t len) {
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  return tflite::VerifyModelBuffer(verifier);
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputNames(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    ThrowException(env, kUnsupportedOperationException,
                   "Internal error: Can not find java/lang/String class to get "
                   "input names.");
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
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    ThrowException(
        env, kIllegalStateException,
        "Internal error: Unexpected failure when preparing tensor allocations:"
        " %s",
        error_reporter->CachedErrorMessage());
  }
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_hasUnresolvedFlexOp(
    JNIEnv* env, jclass clazz, jlong handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
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
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputTensorIndex(
    JNIEnv* env, jclass clazz, jlong handle, jint input_index) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return interpreter->inputs()[input_index];
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputTensorIndex(
    JNIEnv* env, jclass clazz, jlong handle, jint output_index) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return interpreter->outputs()[output_index];
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getExecutionPlanLength(
    JNIEnv* env, jclass clazz, jlong handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return static_cast<jint>(interpreter->execution_plan().size());
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputCount(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return static_cast<jint>(interpreter->inputs().size());
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputCount(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  return static_cast<jint>(interpreter->outputs().size());
}

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputNames(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    ThrowException(env, kUnsupportedOperationException,
                   "Internal error: Can not find java/lang/String class to get "
                   "output names.");
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
Java_org_tensorflow_lite_NativeInterpreterWrapper_useNNAPI(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle,
                                                           jboolean state) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  interpreter->UseNNAPI(static_cast<bool>(state));
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allowFp16PrecisionForFp32(
    JNIEnv* env, jclass clazz, jlong handle, jboolean allow) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  interpreter->SetAllowFp16PrecisionForFp32(static_cast<bool>(allow));
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allowBufferHandleOutput(
    JNIEnv* env, jclass clazz, jlong handle, jboolean allow) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  interpreter->SetAllowBufferHandleOutput(allow);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_numThreads(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle,
                                                             jint num_threads) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  interpreter->SetNumThreads(static_cast<int>(num_threads));
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createErrorReporter(
    JNIEnv* env, jclass clazz, jint size) {
  BufferErrorReporter* error_reporter =
      new BufferErrorReporter(env, static_cast<int>(size));
  return reinterpret_cast<jlong>(error_reporter);
}

// Verifies whether the model is a flatbuffer file.
class JNIFlatBufferVerifier : public tflite_api_dispatcher::TfLiteVerifier {
 public:
  bool Verify(const char* data, int length,
              tflite::ErrorReporter* reporter) override {
    if (!VerifyModel(data, length)) {
      reporter->Report("The model is not a valid Flatbuffer file");
      return false;
    }
    return true;
  }
};

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createModel(
    JNIEnv* env, jclass clazz, jstring model_file, jlong error_handle) {
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;
  const char* path = env->GetStringUTFChars(model_file, nullptr);

  std::unique_ptr<tflite_api_dispatcher::TfLiteVerifier> verifier;
  verifier.reset(new JNIFlatBufferVerifier());

  auto model = tflite_api_dispatcher::TfLiteModel::VerifyAndBuildFromFile(
      path, verifier.get(), error_reporter);
  if (!model) {
    ThrowException(env, kIllegalArgumentException,
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
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;
  const char* buf =
      static_cast<char*>(env->GetDirectBufferAddress(model_buffer));
  jlong capacity = env->GetDirectBufferCapacity(model_buffer);
  if (!VerifyModel(buf, capacity)) {
    ThrowException(env, kIllegalArgumentException,
                   "ByteBuffer is not a valid flatbuffer model");
    return 0;
  }

  auto model = tflite_api_dispatcher::TfLiteModel::BuildFromBuffer(
      buf, static_cast<size_t>(capacity), error_reporter);
  if (!model) {
    ThrowException(env, kIllegalArgumentException,
                   "ByteBuffer does not encode a valid model: %s",
                   error_reporter->CachedErrorMessage());
    return 0;
  }
  return reinterpret_cast<jlong>(model.release());
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createInterpreter(
    JNIEnv* env, jclass clazz, jlong model_handle, jlong error_handle,
    jint num_threads) {
  tflite_api_dispatcher::TfLiteModel* model =
      convertLongToModel(env, model_handle);
  if (model == nullptr) return 0;
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;
  auto resolver = ::tflite::CreateOpResolver();
  std::unique_ptr<tflite_api_dispatcher::Interpreter> interpreter;
  TfLiteStatus status = tflite_api_dispatcher::InterpreterBuilder(
      *model, *(resolver.get()))(&interpreter, static_cast<int>(num_threads));
  if (status != kTfLiteOk) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Cannot create interpreter: %s",
                   error_reporter->CachedErrorMessage());
    return 0;
  }
  // Note that tensor allocation is performed explicitly by the owning Java
  // NativeInterpreterWrapper instance.
  return reinterpret_cast<jlong>(interpreter.release());
}

// Sets inputs, runs inference, and returns outputs as long handles.
JNIEXPORT void JNICALL Java_org_tensorflow_lite_NativeInterpreterWrapper_run(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return;
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return;

  if (interpreter->Invoke() != kTfLiteOk) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Failed to run on the given Interpreter: %s",
                   error_reporter->CachedErrorMessage());
    return;
  }
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputDataType(
    JNIEnv* env, jclass clazz, jlong handle, jint output_idx) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return -1;
  const int idx = static_cast<int>(output_idx);
  if (output_idx < 0 || output_idx >= interpreter->outputs().size()) {
    ThrowException(env, kIllegalArgumentException,
                   "Failed to get %d-th output out of %d outputs", output_idx,
                   interpreter->outputs().size());
    return -1;
  }
  TfLiteTensor* target = interpreter->tensor(interpreter->outputs()[idx]);
  int type = getDataType(target->type);
  return static_cast<jint>(type);
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_resizeInput(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle,
    jint input_idx, jintArray dims, jboolean strict) {
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return JNI_FALSE;
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return JNI_FALSE;
  if (input_idx < 0 || input_idx >= interpreter->inputs().size()) {
    ThrowException(env, kIllegalArgumentException,
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
          tensor_idx, convertJIntArrayToVector(env, dims));
    } else {
      status = interpreter->ResizeInputTensor(
          tensor_idx, convertJIntArrayToVector(env, dims));
    }
    if (status != kTfLiteOk) {
      ThrowException(env, kIllegalArgumentException,
                     "Internal error: Failed to resize %d-th input: %s",
                     input_idx, error_reporter->CachedErrorMessage());
      return JNI_FALSE;
    }
  }
  return is_changed ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_applyDelegate(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle,
    jlong delegate_handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return;

  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return;

  TfLiteDelegate* delegate = convertLongToDelegate(env, delegate_handle);
  if (delegate == nullptr) return;

  TfLiteStatus status = interpreter->ModifyGraphWithDelegate(delegate);
  if (status != kTfLiteOk) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Failed to apply delegate: %s",
                   error_reporter->CachedErrorMessage());
  }
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_resetVariableTensors(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return;

  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return;

  TfLiteStatus status = interpreter->ResetVariableTensors();
  if (status != kTfLiteOk) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Failed to reset variable tensors: %s",
                   error_reporter->CachedErrorMessage());
  }
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

#ifdef __cplusplus
}  // extern "C"
#endif
