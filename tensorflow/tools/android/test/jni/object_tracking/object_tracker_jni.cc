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

#include <android/log.h>
#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cstdint>

#include "tensorflow/tools/android/test/jni/object_tracking/config.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/jni_utils.h"
#include "tensorflow/tools/android/test/jni/object_tracking/object_tracker.h"
#include "tensorflow/tools/android/test/jni/object_tracking/time_log.h"

namespace tf_tracking {

#define OBJECT_TRACKER_METHOD(METHOD_NAME) \
  Java_org_tensorflow_demo_tracking_ObjectTracker_##METHOD_NAME  // NOLINT

JniLongField object_tracker_field("nativeObjectTracker");

ObjectTracker* get_object_tracker(JNIEnv* env, jobject thiz) {
  ObjectTracker* const object_tracker =
      reinterpret_cast<ObjectTracker*>(object_tracker_field.get(env, thiz));
  CHECK_ALWAYS(object_tracker != NULL, "null object tracker!");
  return object_tracker;
}

void set_object_tracker(JNIEnv* env, jobject thiz,
                        const ObjectTracker* object_tracker) {
  object_tracker_field.set(env, thiz,
                           reinterpret_cast<intptr_t>(object_tracker));
}

#ifdef __cplusplus
extern "C" {
#endif
JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(initNative)(JNIEnv* env, jobject thiz,
                                               jint width, jint height,
                                               jboolean always_track);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(releaseMemoryNative)(JNIEnv* env,
                                                        jobject thiz);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(registerNewObjectWithAppearanceNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2, jbyteArray frame_data);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(setPreviousPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2, jlong timestamp);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(setCurrentPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2);

JNIEXPORT
jboolean JNICALL OBJECT_TRACKER_METHOD(haveObject)(JNIEnv* env, jobject thiz,
                                                   jstring object_id);

JNIEXPORT
jboolean JNICALL OBJECT_TRACKER_METHOD(isObjectVisible)(JNIEnv* env,
                                                        jobject thiz,
                                                        jstring object_id);

JNIEXPORT
jstring JNICALL OBJECT_TRACKER_METHOD(getModelIdNative)(JNIEnv* env,
                                                        jobject thiz,
                                                        jstring object_id);

JNIEXPORT
jfloat JNICALL OBJECT_TRACKER_METHOD(getCurrentCorrelation)(JNIEnv* env,
                                                            jobject thiz,
                                                            jstring object_id);

JNIEXPORT
jfloat JNICALL OBJECT_TRACKER_METHOD(getMatchScore)(JNIEnv* env, jobject thiz,
                                                    jstring object_id);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(getTrackedPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloatArray rect_array);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(nextFrameNative)(JNIEnv* env, jobject thiz,
                                                    jbyteArray y_data,
                                                    jbyteArray uv_data,
                                                    jlong timestamp,
                                                    jfloatArray vg_matrix_2x3);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(forgetNative)(JNIEnv* env, jobject thiz,
                                                 jstring object_id);

JNIEXPORT
jbyteArray JNICALL OBJECT_TRACKER_METHOD(getKeypointsPacked)(
    JNIEnv* env, jobject thiz, jfloat scale_factor);

JNIEXPORT
jfloatArray JNICALL OBJECT_TRACKER_METHOD(getKeypointsNative)(
    JNIEnv* env, jobject thiz, jboolean only_found_);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(getCurrentPositionNative)(
    JNIEnv* env, jobject thiz, jlong timestamp, jfloat position_x1,
    jfloat position_y1, jfloat position_x2, jfloat position_y2,
    jfloatArray delta);

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(drawNative)(JNIEnv* env, jobject obj,
                                               jint view_width,
                                               jint view_height,
                                               jfloatArray delta);

JNIEXPORT void JNICALL OBJECT_TRACKER_METHOD(downsampleImageNative)(
    JNIEnv* env, jobject thiz, jint width, jint height, jint row_stride,
    jbyteArray input, jint factor, jbyteArray output);

#ifdef __cplusplus
}
#endif

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(initNative)(JNIEnv* env, jobject thiz,
                                               jint width, jint height,
                                               jboolean always_track) {
  LOGI("Initializing object tracker. %dx%d @%p", width, height, thiz);
  const Size image_size(width, height);
  TrackerConfig* const tracker_config = new TrackerConfig(image_size);
  tracker_config->always_track = always_track;

  // XXX detector
  ObjectTracker* const tracker = new ObjectTracker(tracker_config, NULL);
  set_object_tracker(env, thiz, tracker);
  LOGI("Initialized!");

  CHECK_ALWAYS(get_object_tracker(env, thiz) == tracker,
               "Failure to set hand tracker!");
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(releaseMemoryNative)(JNIEnv* env,
                                                        jobject thiz) {
  delete get_object_tracker(env, thiz);
  set_object_tracker(env, thiz, NULL);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(registerNewObjectWithAppearanceNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2, jbyteArray frame_data) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  LOGI("Registering the position of %s at %.2f,%.2f,%.2f,%.2f", id_str, x1, y1,
       x2, y2);

  jboolean iCopied = JNI_FALSE;

  // Copy image into currFrame.
  jbyte* pixels = env->GetByteArrayElements(frame_data, &iCopied);

  BoundingBox bounding_box(x1, y1, x2, y2);
  get_object_tracker(env, thiz)->RegisterNewObjectWithAppearance(
      id_str, reinterpret_cast<const uint8_t*>(pixels), bounding_box);

  env->ReleaseByteArrayElements(frame_data, pixels, JNI_ABORT);

  env->ReleaseStringUTFChars(object_id, id_str);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(setPreviousPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2, jlong timestamp) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  LOGI(
      "Registering the position of %s at %.2f,%.2f,%.2f,%.2f"
      " at time %lld",
      id_str, x1, y1, x2, y2, static_cast<long long>(timestamp));

  get_object_tracker(env, thiz)->SetPreviousPositionOfObject(
      id_str, BoundingBox(x1, y1, x2, y2), timestamp);

  env->ReleaseStringUTFChars(object_id, id_str);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(setCurrentPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloat x1, jfloat y1,
    jfloat x2, jfloat y2) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  LOGI("Registering the position of %s at %.2f,%.2f,%.2f,%.2f", id_str, x1, y1,
       x2, y2);

  get_object_tracker(env, thiz)->SetCurrentPositionOfObject(
      id_str, BoundingBox(x1, y1, x2, y2));

  env->ReleaseStringUTFChars(object_id, id_str);
}

JNIEXPORT
jboolean JNICALL OBJECT_TRACKER_METHOD(haveObject)(JNIEnv* env, jobject thiz,
                                                   jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const bool haveObject = get_object_tracker(env, thiz)->HaveObject(id_str);
  env->ReleaseStringUTFChars(object_id, id_str);
  return haveObject;
}

JNIEXPORT
jboolean JNICALL OBJECT_TRACKER_METHOD(isObjectVisible)(JNIEnv* env,
                                                        jobject thiz,
                                                        jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const bool visible = get_object_tracker(env, thiz)->IsObjectVisible(id_str);
  env->ReleaseStringUTFChars(object_id, id_str);
  return visible;
}

JNIEXPORT
jstring JNICALL OBJECT_TRACKER_METHOD(getModelIdNative)(JNIEnv* env,
                                                        jobject thiz,
                                                        jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);
  const TrackedObject* const object =
      get_object_tracker(env, thiz)->GetObject(id_str);
  env->ReleaseStringUTFChars(object_id, id_str);
  jstring model_name = env->NewStringUTF(object->GetModel()->GetName().c_str());
  return model_name;
}

JNIEXPORT
jfloat JNICALL OBJECT_TRACKER_METHOD(getCurrentCorrelation)(JNIEnv* env,
                                                            jobject thiz,
                                                            jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const float correlation =
      get_object_tracker(env, thiz)->GetObject(id_str)->GetCorrelation();
  env->ReleaseStringUTFChars(object_id, id_str);
  return correlation;
}

JNIEXPORT
jfloat JNICALL OBJECT_TRACKER_METHOD(getMatchScore)(JNIEnv* env, jobject thiz,
                                                    jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const float match_score =
      get_object_tracker(env, thiz)->GetObject(id_str)->GetMatchScore().value;
  env->ReleaseStringUTFChars(object_id, id_str);
  return match_score;
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(getTrackedPositionNative)(
    JNIEnv* env, jobject thiz, jstring object_id, jfloatArray rect_array) {
  jboolean iCopied = JNI_FALSE;
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  const BoundingBox bounding_box =
      get_object_tracker(env, thiz)->GetObject(id_str)->GetPosition();
  env->ReleaseStringUTFChars(object_id, id_str);

  jfloat* rect = env->GetFloatArrayElements(rect_array, &iCopied);
  bounding_box.CopyToArray(reinterpret_cast<float*>(rect));
  env->ReleaseFloatArrayElements(rect_array, rect, 0);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(nextFrameNative)(JNIEnv* env, jobject thiz,
                                                    jbyteArray y_data,
                                                    jbyteArray uv_data,
                                                    jlong timestamp,
                                                    jfloatArray vg_matrix_2x3) {
  TimeLog("Starting object tracker");

  jboolean iCopied = JNI_FALSE;

  float vision_gyro_matrix_array[6];
  jfloat* jmat = NULL;

  if (vg_matrix_2x3 != NULL) {
    // Copy the alignment matrix into a float array.
    jmat = env->GetFloatArrayElements(vg_matrix_2x3, &iCopied);
    for (int i = 0; i < 6; ++i) {
      vision_gyro_matrix_array[i] = static_cast<float>(jmat[i]);
    }
  }
  // Copy image into currFrame.
  jbyte* pixels = env->GetByteArrayElements(y_data, &iCopied);
  jbyte* uv_pixels =
      uv_data != NULL ? env->GetByteArrayElements(uv_data, &iCopied) : NULL;

  TimeLog("Got elements");

  // Add the frame to the object tracker object.
  get_object_tracker(env, thiz)->NextFrame(
      reinterpret_cast<uint8_t*>(pixels), reinterpret_cast<uint8_t*>(uv_pixels),
      timestamp, vg_matrix_2x3 != NULL ? vision_gyro_matrix_array : NULL);

  env->ReleaseByteArrayElements(y_data, pixels, JNI_ABORT);

  if (uv_data != NULL) {
    env->ReleaseByteArrayElements(uv_data, uv_pixels, JNI_ABORT);
  }

  if (vg_matrix_2x3 != NULL) {
    env->ReleaseFloatArrayElements(vg_matrix_2x3, jmat, JNI_ABORT);
  }

  TimeLog("Released elements");

  PrintTimeLog();
  ResetTimeLog();
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(forgetNative)(JNIEnv* env, jobject thiz,
                                                 jstring object_id) {
  const char* const id_str = env->GetStringUTFChars(object_id, 0);

  get_object_tracker(env, thiz)->ForgetTarget(id_str);

  env->ReleaseStringUTFChars(object_id, id_str);
}

JNIEXPORT
jfloatArray JNICALL OBJECT_TRACKER_METHOD(getKeypointsNative)(
    JNIEnv* env, jobject thiz, jboolean only_found) {
  jfloat keypoint_arr[kMaxKeypoints * kKeypointStep];

  const int number_of_keypoints =
      get_object_tracker(env, thiz)->GetKeypoints(only_found, keypoint_arr);

  // Create and return the array that will be passed back to Java.
  jfloatArray keypoints =
      env->NewFloatArray(number_of_keypoints * kKeypointStep);
  if (keypoints == NULL) {
    LOGE("null array!");
    return NULL;
  }
  env->SetFloatArrayRegion(keypoints, 0, number_of_keypoints * kKeypointStep,
                           keypoint_arr);

  return keypoints;
}

JNIEXPORT
jbyteArray JNICALL OBJECT_TRACKER_METHOD(getKeypointsPacked)(
    JNIEnv* env, jobject thiz, jfloat scale_factor) {
  // 2 bytes to a uint16_t and two pairs of xy coordinates per keypoint.
  const int bytes_per_keypoint = sizeof(uint16_t) * 2 * 2;
  jbyte keypoint_arr[kMaxKeypoints * bytes_per_keypoint];

  const int number_of_keypoints =
      get_object_tracker(env, thiz)->GetKeypointsPacked(
          reinterpret_cast<uint16_t*>(keypoint_arr), scale_factor);

  // Create and return the array that will be passed back to Java.
  jbyteArray keypoints =
      env->NewByteArray(number_of_keypoints * bytes_per_keypoint);

  if (keypoints == NULL) {
    LOGE("null array!");
    return NULL;
  }

  env->SetByteArrayRegion(
      keypoints, 0, number_of_keypoints * bytes_per_keypoint, keypoint_arr);

  return keypoints;
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(getCurrentPositionNative)(
    JNIEnv* env, jobject thiz, jlong timestamp, jfloat position_x1,
    jfloat position_y1, jfloat position_x2, jfloat position_y2,
    jfloatArray delta) {
  jfloat point_arr[4];

  const BoundingBox new_position = get_object_tracker(env, thiz)->TrackBox(
      BoundingBox(position_x1, position_y1, position_x2, position_y2),
      timestamp);

  new_position.CopyToArray(point_arr);
  env->SetFloatArrayRegion(delta, 0, 4, point_arr);
}

JNIEXPORT
void JNICALL OBJECT_TRACKER_METHOD(drawNative)(
    JNIEnv* env, jobject thiz, jint view_width, jint view_height,
    jfloatArray frame_to_canvas_arr) {
  ObjectTracker* object_tracker = get_object_tracker(env, thiz);
  if (object_tracker != NULL) {
    jfloat* frame_to_canvas =
        env->GetFloatArrayElements(frame_to_canvas_arr, NULL);

    object_tracker->Draw(view_width, view_height, frame_to_canvas);
    env->ReleaseFloatArrayElements(frame_to_canvas_arr, frame_to_canvas,
                                   JNI_ABORT);
  }
}

JNIEXPORT void JNICALL OBJECT_TRACKER_METHOD(downsampleImageNative)(
    JNIEnv* env, jobject thiz, jint width, jint height, jint row_stride,
    jbyteArray input, jint factor, jbyteArray output) {
  if (input == NULL || output == NULL) {
    LOGW("Received null arrays, hopefully this is a test!");
    return;
  }

  jbyte* const input_array = env->GetByteArrayElements(input, 0);
  jbyte* const output_array = env->GetByteArrayElements(output, 0);

  {
    tf_tracking::Image<uint8_t> full_image(
        width, height, reinterpret_cast<uint8_t*>(input_array), false);

    const int new_width = (width + factor - 1) / factor;
    const int new_height = (height + factor - 1) / factor;

    tf_tracking::Image<uint8_t> downsampled_image(
        new_width, new_height, reinterpret_cast<uint8_t*>(output_array), false);

    downsampled_image.DownsampleAveraged(
        reinterpret_cast<uint8_t*>(input_array), row_stride, factor);
  }

  env->ReleaseByteArrayElements(input, input_array, JNI_ABORT);
  env->ReleaseByteArrayElements(output, output_array, 0);
}

}  // namespace tf_tracking
