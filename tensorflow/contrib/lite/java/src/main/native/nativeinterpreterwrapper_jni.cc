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

#include "tensorflow/contrib/lite/java/src/main/native/nativeinterpreterwrapper_jni.h"
namespace {

const int kByteBufferValue = 999;
const int kBufferSize = 256;

tflite::Interpreter* convertLongToInterpreter(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to Interpreter.");
    return nullptr;
  }
  return reinterpret_cast<tflite::Interpreter*>(handle);
}

tflite::FlatBufferModel* convertLongToModel(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to model.");
    return nullptr;
  }
  return reinterpret_cast<tflite::FlatBufferModel*>(handle);
}

BufferErrorReporter* convertLongToErrorReporter(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    throwException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to ErrorReporter.");
    return nullptr;
  }
  return reinterpret_cast<BufferErrorReporter*>(handle);
}

std::vector<int> convertJIntArrayToVector(JNIEnv* env, jintArray inputs) {
  int size = static_cast<int>(env->GetArrayLength(inputs));
  std::vector<int> outputs(size, 0);
  jint* ptr = env->GetIntArrayElements(inputs, nullptr);
  if (ptr == nullptr) {
    throwException(env, kIllegalArgumentException,
                   "Array has empty dimensions.");
    return {};
  }
  for (int i = 0; i < size; ++i) {
    outputs[i] = ptr[i];
  }
  env->ReleaseIntArrayElements(inputs, ptr, JNI_ABORT);
  return outputs;
}

bool isByteBuffer(jint data_type) { return data_type == kByteBufferValue; }

TfLiteType resolveDataType(jint data_type) {
  switch (data_type) {
    case 1:
      return kTfLiteFloat32;
    case 2:
      return kTfLiteInt32;
    case 3:
      return kTfLiteUInt8;
    case 4:
      return kTfLiteInt64;
    default:
      return kTfLiteNoType;
  }
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

TfLiteStatus checkInputs(JNIEnv* env, tflite::Interpreter* interpreter,
                         const int input_size, jintArray data_types,
                         jintArray nums_of_bytes, jobjectArray values,
                         jobjectArray sizes) {
  if (input_size != interpreter->inputs().size()) {
    throwException(env, kIllegalArgumentException,
                   "Input error: Expected num of inputs is %d but got %d",
                   interpreter->inputs().size(), input_size);
    return kTfLiteError;
  }
  if (input_size != env->GetArrayLength(data_types) ||
      input_size != env->GetArrayLength(nums_of_bytes) ||
      input_size != env->GetArrayLength(values)) {
    throwException(env, kIllegalArgumentException,
                   "Internal error: Arrays in arguments should be of the same "
                   "length, but got %d sizes, %d data_types, %d nums_of_bytes, "
                   "and %d values",
                   input_size, env->GetArrayLength(data_types),
                   env->GetArrayLength(nums_of_bytes),
                   env->GetArrayLength(values));
    return kTfLiteError;
  }
  for (int i = 0; i < input_size; ++i) {
    int input_idx = interpreter->inputs()[i];
    TfLiteTensor* target = interpreter->tensor(input_idx);
    jintArray dims =
        static_cast<jintArray>(env->GetObjectArrayElement(sizes, i));
    int num_dims = static_cast<int>(env->GetArrayLength(dims));
    if (target->dims->size != num_dims) {
      throwException(env, kIllegalArgumentException,
                     "Input error: %d-th input should have %d dimensions, but "
                     "found %d dimensions",
                     i, target->dims->size, num_dims);
      return kTfLiteError;
    }
    jint* ptr = env->GetIntArrayElements(dims, nullptr);
    for (int j = 1; j < num_dims; ++j) {
      if (target->dims->data[j] != ptr[j]) {
        std::unique_ptr<char[]> expected_dims(new char[kBufferSize]);
        std::unique_ptr<char[]> obtained_dims(new char[kBufferSize]);
        printDims(expected_dims.get(), kBufferSize, target->dims->data,
                  num_dims);
        printDims(obtained_dims.get(), kBufferSize, ptr, num_dims);
        throwException(env, kIllegalArgumentException,
                       "Input error: %d-th input dimension should be [%s], but "
                       "found [%s]",
                       i, expected_dims.get(), obtained_dims.get());
        env->ReleaseIntArrayElements(dims, ptr, JNI_ABORT);
        return kTfLiteError;
      }
    }
    env->ReleaseIntArrayElements(dims, ptr, JNI_ABORT);
    env->DeleteLocalRef(dims);
    if (env->ExceptionCheck()) return kTfLiteError;
  }
  return kTfLiteOk;
}

// Checks whether there is any difference between dimensions of a tensor and a
// given dimensions. Returns true if there is difference, else false.
bool areDimsDifferent(JNIEnv* env, TfLiteTensor* tensor, jintArray dims) {
  int num_dims = static_cast<int>(env->GetArrayLength(dims));
  jint* ptr = env->GetIntArrayElements(dims, nullptr);
  if (ptr == nullptr) {
    throwException(env, kIllegalArgumentException,
                   "Empty dimensions of input array.");
    return true;
  }
  if (tensor->dims->size != num_dims) {
    return true;
  }
  for (int i = 0; i < num_dims; ++i) {
    if (ptr[i] != tensor->dims->data[i]) {
      return true;
    }
  }
  env->ReleaseIntArrayElements(dims, ptr, JNI_ABORT);
  return false;
}

bool areInputDimensionsTheSame(JNIEnv* env, tflite::Interpreter* interpreter,
                               int input_size, jobjectArray sizes) {
  if (interpreter->inputs().size() != input_size) {
    return false;
  }
  for (int i = 0; i < input_size; ++i) {
    int input_idx = interpreter->inputs()[i];
    jintArray dims =
        static_cast<jintArray>(env->GetObjectArrayElement(sizes, i));
    TfLiteTensor* target = interpreter->tensor(input_idx);
    if (areDimsDifferent(env, target, dims)) return false;
    env->DeleteLocalRef(dims);
    if (env->ExceptionCheck()) return false;
  }
  return true;
}

TfLiteStatus resizeInputs(JNIEnv* env, tflite::Interpreter* interpreter,
                          int input_size, jobjectArray sizes) {
  for (int i = 0; i < input_size; ++i) {
    int input_idx = interpreter->inputs()[i];
    jintArray dims =
        static_cast<jintArray>(env->GetObjectArrayElement(sizes, i));
    TfLiteStatus status = interpreter->ResizeInputTensor(
        input_idx, convertJIntArrayToVector(env, dims));
    if (status != kTfLiteOk) {
      return status;
    }
    env->DeleteLocalRef(dims);
    if (env->ExceptionCheck()) return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus setInputs(JNIEnv* env, tflite::Interpreter* interpreter,
                       int input_size, jintArray data_types,
                       jintArray nums_of_bytes, jobjectArray values) {
  jint* data_type = env->GetIntArrayElements(data_types, nullptr);
  jint* num_bytes = env->GetIntArrayElements(nums_of_bytes, nullptr);
  for (int i = 0; i < input_size; ++i) {
    int input_idx = interpreter->inputs()[i];
    TfLiteTensor* target = interpreter->tensor(input_idx);
    jobject value = env->GetObjectArrayElement(values, i);
    bool is_byte_buffer = isByteBuffer(data_type[i]);
    if (is_byte_buffer) {
      writeByteBuffer(env, value, &(target->data.raw),
                      static_cast<int>(num_bytes[i]));
    } else {
      TfLiteType type = resolveDataType(data_type[i]);
      if (type != target->type) {
        throwException(env, kIllegalArgumentException,
                       "Input error: DataType (%d) of input data does not "
                       "match with the DataType (%d) of model inputs.",
                       type, target->type);
        return kTfLiteError;
      }
      writeMultiDimensionalArray(env, value, target->type, target->dims->size,
                                 &(target->data.raw),
                                 static_cast<int>(num_bytes[i]));
    }
    env->DeleteLocalRef(value);
    if (env->ExceptionCheck()) return kTfLiteError;
  }
  env->ReleaseIntArrayElements(data_types, data_type, JNI_ABORT);
  env->ReleaseIntArrayElements(nums_of_bytes, num_bytes, JNI_ABORT);
  return kTfLiteOk;
}

// TODO(yichengfan): evaluate the benefit to use tflite verifier.
bool VerifyModel(const void* buf, size_t len) {
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  return tflite::VerifyModelBuffer(verifier);
}

}  // namespace

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputNames(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle) {
  tflite::Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    throwException(env, kUnsupportedOperationException,
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

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputNames(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle) {
  tflite::Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    throwException(env, kUnsupportedOperationException,
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
  tflite::Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  interpreter->UseNNAPI(static_cast<bool>(state));
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_numThreads(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong handle,
                                                           jint num_threads) {
  tflite::Interpreter* interpreter = convertLongToInterpreter(env, handle);
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
class JNIFlatBufferVerifier : public tflite::TfLiteVerifier {
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

  std::unique_ptr<tflite::TfLiteVerifier> verifier;
  verifier.reset(new JNIFlatBufferVerifier());

  auto model = tflite::FlatBufferModel::VerifyAndBuildFromFile(
      path, verifier.get(), error_reporter);
  if (!model) {
    throwException(env, kIllegalArgumentException,
                   "Contents of %s does not encode a valid "
                   "TensorFlowLite model: %s",
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
    throwException(env, kIllegalArgumentException,
                   "MappedByteBuffer is not a valid flatbuffer model");
    return 0;
  }

  auto model = tflite::FlatBufferModel::BuildFromBuffer(
      buf, static_cast<size_t>(capacity), error_reporter);
  if (!model) {
    throwException(env, kIllegalArgumentException,
                   "MappedByteBuffer does not encode a valid "
                   "TensorFlowLite model: %s",
                   error_reporter->CachedErrorMessage());
    return 0;
  }
  return reinterpret_cast<jlong>(model.release());
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createInterpreter(
    JNIEnv* env, jclass clazz, jlong model_handle, jlong error_handle,
    jint num_threads) {
  tflite::FlatBufferModel* model = convertLongToModel(env, model_handle);
  if (model == nullptr) return 0;
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;
  auto resolver = ::tflite::CreateOpResolver();
  std::unique_ptr<tflite::Interpreter> interpreter;
  TfLiteStatus status = tflite::InterpreterBuilder(*model, *(resolver.get()))(
      &interpreter, static_cast<int>(num_threads));
  if (status != kTfLiteOk) {
    throwException(env, kIllegalArgumentException,
                   "Internal error: Cannot create interpreter: %s",
                   error_reporter->CachedErrorMessage());
    return 0;
  }
  // allocates memory
  status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    throwException(env, kNullPointerException,
                   "Internal error: Cannot allocate memory for the interpreter",
                   error_reporter->CachedErrorMessage());
    return 0;
  }
  return reinterpret_cast<jlong>(interpreter.release());
}

// Sets inputs, runs inference, and returns outputs as long handles.
JNIEXPORT jlongArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_run(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle,
    jobjectArray sizes, jintArray data_types, jintArray nums_of_bytes,
    jobjectArray values, jobject wrapper, jboolean memory_allocated) {
  tflite::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return nullptr;
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return nullptr;
  const int input_size = env->GetArrayLength(sizes);
  // validates inputs
  TfLiteStatus status = checkInputs(env, interpreter, input_size, data_types,
                                    nums_of_bytes, values, sizes);
  if (status != kTfLiteOk) return nullptr;
  if (!memory_allocated ||
      !areInputDimensionsTheSame(env, interpreter, input_size, sizes)) {
    // resizes inputs
    status = resizeInputs(env, interpreter, input_size, sizes);
    if (status != kTfLiteOk) {
      throwException(env, kNullPointerException,
                     "Internal error: Can not resize the input: %s",
                     error_reporter->CachedErrorMessage());
      return nullptr;
    }
    // allocates memory
    status = interpreter->AllocateTensors();
    if (status != kTfLiteOk) {
      throwException(env, kNullPointerException,
                     "Internal error: Can not allocate memory for the given "
                     "inputs: %s",
                     error_reporter->CachedErrorMessage());
      return nullptr;
    }
  }
  // sets inputs
  status = setInputs(env, interpreter, input_size, data_types, nums_of_bytes,
                     values);
  if (status != kTfLiteOk) return nullptr;
  timespec beforeInference = ::tflite::getCurrentTime();
  // runs inference
  if (interpreter->Invoke() != kTfLiteOk) {
    throwException(env, kIllegalArgumentException,
                   "Internal error: Failed to run on the given Interpreter: %s",
                   error_reporter->CachedErrorMessage());
    return nullptr;
  }
  timespec afterInference = ::tflite::getCurrentTime();
  jclass wrapper_clazz = env->GetObjectClass(wrapper);
  jfieldID fid =
      env->GetFieldID(wrapper_clazz, "inferenceDurationNanoseconds", "J");
  if (env->ExceptionCheck()) {
    env->ExceptionClear();
  } else if (fid != nullptr) {
    env->SetLongField(
        wrapper, fid,
        ::tflite::timespec_diff_nanoseconds(&beforeInference, &afterInference));
  }
  // returns outputs
  const std::vector<int>& results = interpreter->outputs();
  if (results.empty()) {
    throwException(
        env, kIllegalArgumentException,
        "Internal error: The Interpreter does not have any outputs.");
    return nullptr;
  }
  jlongArray outputs = env->NewLongArray(results.size());
  size_t size = results.size();
  for (int i = 0; i < size; ++i) {
    TfLiteTensor* source = interpreter->tensor(results[i]);
    jlong output = reinterpret_cast<jlong>(source);
    env->SetLongArrayRegion(outputs, i, 1, &output);
  }
  return outputs;
}

JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputDims(
    JNIEnv* env, jclass clazz, jlong handle, jint input_idx, jint num_bytes) {
  tflite::Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  const int idx = static_cast<int>(input_idx);
  if (input_idx < 0 || input_idx >= interpreter->inputs().size()) {
    throwException(env, kIllegalArgumentException,
                   "Input error: Out of range: Failed to get %d-th input out of"
                   " %d inputs",
                   input_idx, interpreter->inputs().size());
    return nullptr;
  }
  TfLiteTensor* target = interpreter->tensor(interpreter->inputs()[idx]);
  int size = target->dims->size;
  if (num_bytes >= 0) {  // verifies num of bytes matches if num_bytes if valid.
    int expected_num_bytes = elementByteSize(target->type);
    for (int i = 0; i < size; ++i) {
      expected_num_bytes *= target->dims->data[i];
    }
    if (num_bytes != expected_num_bytes) {
      throwException(env, kIllegalArgumentException,
                     "Input error: Failed to get input dimensions. %d-th input "
                     "should have %d bytes, but found %d bytes.",
                     idx, expected_num_bytes, num_bytes);
      return nullptr;
    }
  }
  jintArray outputs = env->NewIntArray(size);
  env->SetIntArrayRegion(outputs, 0, size, &(target->dims->data[0]));
  return outputs;
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputDataType(
    JNIEnv* env, jclass clazz, jlong handle, jint output_idx) {
  tflite::Interpreter* interpreter = convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return -1;
  const int idx = static_cast<int>(output_idx);
  if (output_idx < 0 || output_idx >= interpreter->outputs().size()) {
    throwException(env, kIllegalArgumentException,
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
    jint input_idx, jintArray dims) {
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return JNI_FALSE;
  tflite::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return JNI_FALSE;
  const int idx = static_cast<int>(input_idx);
  if (idx < 0 || idx >= interpreter->inputs().size()) {
    throwException(env, kIllegalArgumentException,
                   "Input error: Can not resize %d-th input for a model having "
                   "%d inputs.",
                   idx, interpreter->inputs().size());
    return JNI_FALSE;
  }
  // check whether it is resizing with the same dimensions.
  TfLiteTensor* target = interpreter->tensor(input_idx);
  bool is_changed = areDimsDifferent(env, target, dims);
  if (is_changed) {
    TfLiteStatus status = interpreter->ResizeInputTensor(
        interpreter->inputs()[idx], convertJIntArrayToVector(env, dims));
    if (status != kTfLiteOk) {
      throwException(env, kIllegalArgumentException,
                     "Internal error: Failed to resize %d-th input: %s", idx,
                     error_reporter->CachedErrorMessage());
      return JNI_FALSE;
    }
  }
  return is_changed ? JNI_TRUE : JNI_FALSE;
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
