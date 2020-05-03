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

#include <limits>
#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"
#include "tensorflow/java/src/main/native/saved_model_bundle_jni.h"

JNIEXPORT jobject JNICALL Java_org_tensorflow_SavedModelBundle_load(
    JNIEnv* env, jclass clazz, jstring export_dir, jobjectArray tags,
    jbyteArray config, jbyteArray run_options) {
  TF_Status* status = TF_NewStatus();
  jobject bundle = nullptr;

  // allocate parameters for TF_LoadSessionFromSavedModel
  TF_SessionOptions* opts = TF_NewSessionOptions();
  if (config != nullptr) {
    size_t sz = env->GetArrayLength(config);
    if (sz > 0) {
      jbyte* config_data = env->GetByteArrayElements(config, nullptr);
      TF_SetConfig(opts, static_cast<void*>(config_data), sz, status);
      env->ReleaseByteArrayElements(config, config_data, JNI_ABORT);
      if (!throwExceptionIfNotOK(env, status)) {
        TF_DeleteSessionOptions(opts);
        TF_DeleteStatus(status);
        return nullptr;
      }
    }
  }
  TF_Buffer* crun_options = nullptr;
  if (run_options != nullptr) {
    size_t sz = env->GetArrayLength(run_options);
    if (sz > 0) {
      jbyte* run_options_data = env->GetByteArrayElements(run_options, nullptr);
      crun_options =
          TF_NewBufferFromString(static_cast<void*>(run_options_data), sz);
      env->ReleaseByteArrayElements(run_options, run_options_data, JNI_ABORT);
    }
  }
  const char* cexport_dir = env->GetStringUTFChars(export_dir, nullptr);
  std::unique_ptr<const char* []> tags_ptrs;
  size_t tags_len = env->GetArrayLength(tags);
  tags_ptrs.reset(new const char*[tags_len]);
  for (size_t i = 0; i < tags_len; ++i) {
    jstring tag = static_cast<jstring>(env->GetObjectArrayElement(tags, i));
    tags_ptrs[i] = env->GetStringUTFChars(tag, nullptr);
    env->DeleteLocalRef(tag);
  }

  // load the session
  TF_Graph* graph = TF_NewGraph();
  TF_Buffer* metagraph_def = TF_NewBuffer();
  TF_Session* session = TF_LoadSessionFromSavedModel(
      opts, crun_options, cexport_dir, tags_ptrs.get(), tags_len, graph,
      metagraph_def, status);

  // release the parameters
  TF_DeleteSessionOptions(opts);
  if (crun_options != nullptr) {
    TF_DeleteBuffer(crun_options);
  }
  env->ReleaseStringUTFChars(export_dir, cexport_dir);
  for (size_t i = 0; i < tags_len; ++i) {
    jstring tag = static_cast<jstring>(env->GetObjectArrayElement(tags, i));
    env->ReleaseStringUTFChars(tag, tags_ptrs[i]);
    env->DeleteLocalRef(tag);
  }

  // handle the result
  if (throwExceptionIfNotOK(env, status)) {
    // sizeof(jsize) is less than sizeof(size_t) on some platforms.
    if (metagraph_def->length > std::numeric_limits<jint>::max()) {
      throwException(
          env, kIndexOutOfBoundsException,
          "MetaGraphDef is too large to serialize into a byte[] array");
    } else {
      static_assert(sizeof(jbyte) == 1, "unexpected size of the jbyte type");
      jint jmetagraph_len = static_cast<jint>(metagraph_def->length);
      jbyteArray jmetagraph_def = env->NewByteArray(jmetagraph_len);
      env->SetByteArrayRegion(jmetagraph_def, 0, jmetagraph_len,
                              static_cast<const jbyte*>(metagraph_def->data));

      jmethodID method = env->GetStaticMethodID(
          clazz, "fromHandle", "(JJ[B)Lorg/tensorflow/SavedModelBundle;");
      bundle = env->CallStaticObjectMethod(
          clazz, method, reinterpret_cast<jlong>(graph),
          reinterpret_cast<jlong>(session), jmetagraph_def);
      graph = nullptr;
      session = nullptr;
      env->DeleteLocalRef(jmetagraph_def);
    }
  }

  if (session != nullptr) {
    TF_CloseSession(session, status);
    // Result of close is ignored, delete anyway.
    TF_DeleteSession(session, status);
  }
  if (graph != nullptr) {
    TF_DeleteGraph(graph);
  }
  TF_DeleteBuffer(metagraph_def);
  TF_DeleteStatus(status);

  return bundle;
}
