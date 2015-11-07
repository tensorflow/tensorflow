// The methods are exposed to Java to allow for interaction with the native
// Tensorflow code. See
// tensorflow/examples/android/src/org/tensorflow/TensorflowClassifier.java
// for the Java counterparts.

#ifndef ORG_TENSORFLOW_JNI_TENSORFLOW_JNI_H_  // NOLINT
#define ORG_TENSORFLOW_JNI_TENSORFLOW_JNI_H_  // NOLINT

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define TENSORFLOW_METHOD(METHOD_NAME) \
  Java_org_tensorflow_demo_TensorflowClassifier_##METHOD_NAME  // NOLINT

JNIEXPORT jint JNICALL
TENSORFLOW_METHOD(initializeTensorflow)(
    JNIEnv* env, jobject thiz, jobject java_asset_manager,
    jstring model, jstring labels,
    jint num_classes, jint mognet_input_size, jint image_mean);

JNIEXPORT jstring JNICALL
TENSORFLOW_METHOD(classifyImageBmp)(
    JNIEnv* env, jobject thiz, jobject bitmap);

JNIEXPORT jstring JNICALL
TENSORFLOW_METHOD(classifyImageRgb)(
    JNIEnv* env, jobject thiz, jintArray image, jint width, jint height);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ORG_TENSORFLOW_JNI_TENSORFLOW_JNI_H_  // NOLINT
