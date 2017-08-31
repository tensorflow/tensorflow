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

#ifndef THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_JNI_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_JNI_UTILS_H_

#include <stdint.h>

#include "tensorflow/examples/android/jni/object_tracking/utils.h"

// The JniLongField class is used to access Java fields from native code. This
// technique of hiding pointers to native objects in opaque Java fields is how
// the Android hardware libraries work. This reduces the amount of static
// native methods and makes it easier to manage the lifetime of native objects.
class JniLongField {
 public:
  JniLongField(const char* field_name)
      : field_name_(field_name), field_ID_(0) {}

  int64_t get(JNIEnv* env, jobject thiz) {
    if (field_ID_ == 0) {
      jclass cls = env->GetObjectClass(thiz);
      CHECK_ALWAYS(cls != 0, "Unable to find class");
      field_ID_ = env->GetFieldID(cls, field_name_, "J");
      CHECK_ALWAYS(field_ID_ != 0,
          "Unable to find field %s. (Check proguard cfg)", field_name_);
    }

    return env->GetLongField(thiz, field_ID_);
  }

  void set(JNIEnv* env, jobject thiz, int64_t value) {
    if (field_ID_ == 0) {
      jclass cls = env->GetObjectClass(thiz);
      CHECK_ALWAYS(cls != 0, "Unable to find class");
      field_ID_ = env->GetFieldID(cls, field_name_, "J");
      CHECK_ALWAYS(field_ID_ != 0,
          "Unable to find field %s (Check proguard cfg)", field_name_);
    }

    env->SetLongField(thiz, field_ID_, value);
  }

 private:
  const char* const field_name_;

  // This is just a cache
  jfieldID field_ID_;
};

#endif
