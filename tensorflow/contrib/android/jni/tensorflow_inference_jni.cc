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

#include "tensorflow/contrib/android/jni/tensorflow_inference_jni.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include <jni.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include <map>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/contrib/android/jni/jni_utils.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/stat_summarizer.h"

using namespace tensorflow;

typedef std::map<std::string, std::pair<std::string, tensorflow::Tensor> >
    InputMap;

// Variables associated with a single TF session.
struct SessionVariables {
  std::unique_ptr<tensorflow::Session> session;

  long id = -1;  // Copied from Java field for convenience.
  int num_runs = 0;
  int64 timing_total_us = 0;

  InputMap input_tensors;
  std::vector<std::string> output_tensor_names;
  std::vector<tensorflow::Tensor> output_tensors;
};

static tensorflow::mutex mutex_(tensorflow::LINKER_INITIALIZED);

std::map<int64, SessionVariables*>* GetSessionsSingleton() {
  static std::map<int64, SessionVariables*>* sessions PT_GUARDED_BY(mutex_) =
      new std::map<int64, SessionVariables*>();
  return sessions;
}

inline static SessionVariables* GetSessionVars(JNIEnv* env, jobject thiz) {
  jclass clazz = env->GetObjectClass(thiz);
  assert(clazz != nullptr);
  jfieldID fid = env->GetFieldID(clazz, "id", "J");
  assert(fid != nullptr);
  const int64 id = env->GetLongField(thiz, fid);

  // This method is thread-safe as we support working with multiple
  // sessions simultaneously. However care must be taken at the calling
  // level on a per-session basis.
  mutex_lock l(mutex_);
  std::map<int64, SessionVariables*>& sessions = *GetSessionsSingleton();
  if (sessions.find(id) == sessions.end()) {
    LOG(INFO) << "Creating new session variables for " << std::hex << id;
    SessionVariables* vars = new SessionVariables;
    vars->id = id;
    sessions[id] = vars;
  } else {
    VLOG(1) << "Found session variables for " << std::hex << id;
  }
  return sessions[id];
}

JNIEXPORT void JNICALL TENSORFLOW_METHOD(testLoaded)(JNIEnv* env,
                                                     jobject thiz) {
  LOG(INFO) << "Native TF methods loaded.";
}

JNIEXPORT jint JNICALL TENSORFLOW_METHOD(initializeTensorFlow)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager, jstring model) {
  SessionVariables* vars = GetSessionVars(env, thiz);

  if (vars->session.get() != nullptr) {
    LOG(INFO) << "Compute graph already loaded. skipping.";
    return 0;
  }

  const int64 start_time = CurrentWallTimeUs();

  const std::string model_str = GetString(env, model);

  LOG(INFO) << "Loading Tensorflow.";

  LOG(INFO) << "Making new SessionOptions.";
  tensorflow::SessionOptions options;
  tensorflow::ConfigProto& config = options.config;
  LOG(INFO) << "Got config, " << config.device_count_size() << " devices";

  tensorflow::Session* session = tensorflow::NewSession(options);
  vars->session.reset(session);
  LOG(INFO) << "Session created.";

  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";

  AAssetManager* const asset_manager =
      AAssetManager_fromJava(env, java_asset_manager);
  LOG(INFO) << "Acquired AssetManager.";

  LOG(INFO) << "Reading file to proto: " << model_str;
  ReadFileToProtoOrDie(asset_manager, model_str.c_str(), &tensorflow_graph);

  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);

  // Clear the proto to save memory space.
  tensorflow_graph.Clear();

  if (!s.ok()) {
    LOG(ERROR) << "Could not create Tensorflow Graph: " << s;
    return s.code();
  }

  LOG(INFO) << "Tensorflow graph loaded from: " << model_str;

  const int64 end_time = CurrentWallTimeUs();
  LOG(INFO) << "Initialization done in " << (end_time - start_time) / 1000.0
            << "ms";

  return s.code();
}

static tensorflow::Tensor* GetTensor(JNIEnv* env, jobject thiz,
                                     jstring node_name_jstring) {
  SessionVariables* vars = GetSessionVars(env, thiz);
  std::string node_name = GetString(env, node_name_jstring);

  int output_index = -1;
  for (int i = 0; i < vars->output_tensors.size(); ++i) {
    if (vars->output_tensor_names[i] == node_name) {
      output_index = i;
      break;
    }
  }
  if (output_index == -1) {
    LOG(ERROR) << "Output [" << node_name << "] not found, aborting!";
    return nullptr;
  }

  tensorflow::Tensor* output = &vars->output_tensors[output_index];
  return output;
}

JNIEXPORT jint JNICALL TENSORFLOW_METHOD(runInference)(
    JNIEnv* env, jobject thiz, jobjectArray output_name_strings) {
  SessionVariables* vars = GetSessionVars(env, thiz);

  // Add the requested outputs to the output list.
  vars->output_tensor_names.clear();
  for (int i = 0; i < env->GetArrayLength(output_name_strings); i++) {
    jstring java_string =
        (jstring)(env->GetObjectArrayElement(output_name_strings, i));
    std::string output_name = GetString(env, java_string);
    vars->output_tensor_names.push_back(output_name);
  }

  ++(vars->num_runs);
  tensorflow::Status s;
  int64 start_time, end_time;

  start_time = CurrentWallTimeUs();

  std::vector<std::pair<std::string, tensorflow::Tensor> > input_tensors;
  for (const auto& entry : vars->input_tensors) {
    input_tensors.push_back(entry.second);
  }

  vars->output_tensors.clear();
  s = vars->session->Run(input_tensors, vars->output_tensor_names, {},
                         &(vars->output_tensors));
  end_time = CurrentWallTimeUs();
  const int64 elapsed_time_inf = end_time - start_time;
  vars->timing_total_us += elapsed_time_inf;
  VLOG(0) << "End computing. Ran in " << elapsed_time_inf / 1000 << "ms ("
          << (vars->timing_total_us / vars->num_runs / 1000) << "ms avg over "
          << vars->num_runs << " runs)";

  if (!s.ok()) {
    LOG(ERROR) << "Error during inference: " << s;
  }
  return s.code();
}

JNIEXPORT jint JNICALL TENSORFLOW_METHOD(close)(JNIEnv* env, jobject thiz) {
  SessionVariables* vars = GetSessionVars(env, thiz);

  tensorflow::Status s = vars->session->Close();
  if (!s.ok()) {
    LOG(ERROR) << "Error closing session: " << s;
  }

  mutex_lock l(mutex_);
  std::map<int64, SessionVariables*>& sessions = *GetSessionsSingleton();
  sessions.erase(vars->id);
  delete vars;

  return s.code();
}

// TODO(andrewharp): Use memcpy to fill/read nodes.
#define FILL_NODE_METHOD(DTYPE, JAVA_DTYPE, TENSOR_DTYPE)                  \
  FILL_NODE_SIGNATURE(DTYPE, JAVA_DTYPE) {                                 \
    SessionVariables* vars = GetSessionVars(env, thiz);                    \
    jboolean iCopied = JNI_FALSE;                                          \
    tensorflow::TensorShape shape;                                         \
    jint* dim_vals = env->GetIntArrayElements(dims, &iCopied);             \
    const int num_dims = env->GetArrayLength(dims);                        \
    for (int i = 0; i < num_dims; ++i) {                                   \
      shape.AddDim(dim_vals[i]);                                           \
    }                                                                      \
    env->ReleaseIntArrayElements(dims, dim_vals, JNI_ABORT);               \
    tensorflow::Tensor input_tensor(TENSOR_DTYPE, shape);                  \
    auto tensor_mapped = input_tensor.flat<JAVA_DTYPE>();                  \
    j##JAVA_DTYPE* values = env->Get##DTYPE##ArrayElements(arr, &iCopied); \
    j##JAVA_DTYPE* value_ptr = values;                                     \
    const int array_size = env->GetArrayLength(arr);                       \
    for (int i = 0;                                                        \
         i < std::min(static_cast<int>(tensor_mapped.size()), array_size); \
         ++i) {                                                            \
      tensor_mapped(i) = *value_ptr++;                                     \
    }                                                                      \
    env->Release##DTYPE##ArrayElements(arr, values, JNI_ABORT);            \
    std::string input_name = GetString(env, node_name);                    \
    std::pair<std::string, tensorflow::Tensor> input_pair(input_name,      \
                                                          input_tensor);   \
    vars->input_tensors[input_name] = input_pair;                          \
  }

#define READ_NODE_METHOD(DTYPE, JAVA_DTYPE)                                \
  READ_NODE_SIGNATURE(DTYPE, JAVA_DTYPE) {                                 \
    SessionVariables* vars = GetSessionVars(env, thiz);                    \
    Tensor* t = GetTensor(env, thiz, node_name_jstring);                   \
    if (t == nullptr) {                                                    \
      return -1;                                                           \
    }                                                                      \
    auto tensor_mapped = t->flat<JAVA_DTYPE>();                            \
    jboolean iCopied = JNI_FALSE;                                          \
    j##JAVA_DTYPE* values = env->Get##DTYPE##ArrayElements(arr, &iCopied); \
    j##JAVA_DTYPE* value_ptr = values;                                     \
    const int num_items = std::min(static_cast<int>(tensor_mapped.size()), \
                                   env->GetArrayLength(arr));              \
    for (int i = 0; i < num_items; ++i) {                                  \
      *value_ptr++ = tensor_mapped(i);                                     \
    }                                                                      \
    env->Release##DTYPE##ArrayElements(arr, values, 0);                    \
    return 0;                                                              \
  }

FILL_NODE_METHOD(Float, float, tensorflow::DT_FLOAT)
FILL_NODE_METHOD(Int, int, tensorflow::DT_INT32)
FILL_NODE_METHOD(Double, double, tensorflow::DT_DOUBLE)

READ_NODE_METHOD(Float, float)
READ_NODE_METHOD(Int, int)
READ_NODE_METHOD(Double, double)
