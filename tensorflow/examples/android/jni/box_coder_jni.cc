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

// This file loads the box coder mappings.

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
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/examples/android/proto/box_coder.pb.h"

#define TENSORFLOW_METHOD(METHOD_NAME) \
  Java_org_tensorflow_demo_TensorFlowMultiBoxDetector_##METHOD_NAME  // NOLINT

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT void JNICALL TENSORFLOW_METHOD(loadCoderOptions)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager, jstring location,
    jfloatArray priors);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

JNIEXPORT void JNICALL TENSORFLOW_METHOD(loadCoderOptions)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager, jstring location,
    jfloatArray priors) {
  AAssetManager* const asset_manager =
      AAssetManager_fromJava(env, java_asset_manager);
  LOG(INFO) << "Acquired AssetManager.";

  const std::string location_str = GetString(env, location);

  org_tensorflow_demo::MultiBoxCoderOptions multi_options;

  LOG(INFO) << "Reading file to proto: " << location_str;
  ReadFileToProtoOrDie(asset_manager, location_str.c_str(), &multi_options);

  LOG(INFO) << "Read file. " << multi_options.box_coder_size() << " entries.";

  jboolean iCopied = JNI_FALSE;
  jfloat* values = env->GetFloatArrayElements(priors, &iCopied);

  const int array_length = env->GetArrayLength(priors);
  LOG(INFO) << "Array length: " << array_length
            << " (/8 = " << (array_length / 8) << ")";
  CHECK_EQ(array_length % 8, 0);

  const int num_items =
      std::min(array_length / 8, multi_options.box_coder_size());

  for (int i = 0; i < num_items; ++i) {
    const org_tensorflow_demo::BoxCoderOptions& options =
        multi_options.box_coder(i);

    for (int j = 0; j < 4; ++j) {
      const org_tensorflow_demo::BoxCoderPrior& prior = options.priors(j);
      values[i * 8 + j * 2] = prior.mean();
      values[i * 8 + j * 2 + 1] = prior.stddev();
    }
  }
  env->ReleaseFloatArrayElements(priors, values, 0);

  LOG(INFO) << "Read " << num_items << " options";
}
