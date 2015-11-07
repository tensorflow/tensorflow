// This file binds the native image utility code to the Java class
// which exposes them.

#include <jni.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensorflow/core/platform/port.h"
#include "tensorflow/examples/android/jni/rgb2yuv.h"
#include "tensorflow/examples/android/jni/yuv2rgb.h"

#define IMAGEUTILS_METHOD(METHOD_NAME) \
  Java_org_tensorflow_demo_env_ImageUtils_##METHOD_NAME  // NOLINT

using namespace tensorflow;

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertYUV420SPToARGB8888)(
    JNIEnv* env, jclass clazz, jbyteArray input, jintArray output,
    jint width, jint height, jboolean halfSize);

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertYUV420SPToRGB565)(
    JNIEnv* env, jclass clazz, jbyteArray input, jbyteArray output,
    jint width, jint height);

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertARGB8888ToYUV420SP)(
    JNIEnv* env, jclass clazz, jintArray input, jbyteArray output,
    jint width, jint height);

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertRGB565ToYUV420SP)(
    JNIEnv* env, jclass clazz, jbyteArray input, jbyteArray output,
    jint width, jint height);

#ifdef __cplusplus
}
#endif

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertYUV420SPToARGB8888)(
    JNIEnv* env, jclass clazz, jbyteArray input, jintArray output,
    jint width, jint height, jboolean halfSize) {
  jboolean inputCopy = JNI_FALSE;
  jbyte* const i = env->GetByteArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jint* const o = env->GetIntArrayElements(output, &outputCopy);

  if (halfSize) {
    ConvertYUV420SPToARGB8888HalfSize(reinterpret_cast<uint8*>(i),
                                      reinterpret_cast<uint32*>(o),
                                      width, height);
  } else {
    ConvertYUV420SPToARGB8888(reinterpret_cast<uint8*>(i),
                              reinterpret_cast<uint8*>(i) + width * height,
                              reinterpret_cast<uint32*>(o),
                              width, height);
  }

  env->ReleaseByteArrayElements(input, i, JNI_ABORT);
  env->ReleaseIntArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertYUV420SPToRGB565)(
    JNIEnv* env, jclass clazz, jbyteArray input, jbyteArray output,
    jint width, jint height) {
  jboolean inputCopy = JNI_FALSE;
  jbyte* const i = env->GetByteArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jbyte* const o = env->GetByteArrayElements(output, &outputCopy);

  ConvertYUV420SPToRGB565(reinterpret_cast<uint8*>(i),
                          reinterpret_cast<uint16*>(o),
                          width, height);

  env->ReleaseByteArrayElements(input, i, JNI_ABORT);
  env->ReleaseByteArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertARGB8888ToYUV420SP)(
    JNIEnv* env, jclass clazz, jintArray input, jbyteArray output,
    jint width, jint height) {
  jboolean inputCopy = JNI_FALSE;
  jint* const i = env->GetIntArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jbyte* const o = env->GetByteArrayElements(output, &outputCopy);

  ConvertARGB8888ToYUV420SP(reinterpret_cast<uint32*>(i),
                            reinterpret_cast<uint8*>(o),
                            width, height);

  env->ReleaseIntArrayElements(input, i, JNI_ABORT);
  env->ReleaseByteArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertRGB565ToYUV420SP)(
    JNIEnv* env, jclass clazz, jbyteArray input, jbyteArray output,
    jint width, jint height) {
  jboolean inputCopy = JNI_FALSE;
  jbyte* const i = env->GetByteArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jbyte* const o = env->GetByteArrayElements(output, &outputCopy);

  ConvertRGB565ToYUV420SP(reinterpret_cast<uint16*>(i),
                          reinterpret_cast<uint8*>(o),
                          width, height);

  env->ReleaseByteArrayElements(input, i, JNI_ABORT);
  env->ReleaseByteArrayElements(output, o, 0);
}
