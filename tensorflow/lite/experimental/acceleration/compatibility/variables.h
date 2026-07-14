/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_VARIABLES_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_VARIABLES_H_

// This file lists generally useful compatibility properties.
// Properties starting with "tflite." are reserved.
// Users of the compatibility library can use arbitrary other property names.

namespace tflite {
namespace acceleration {
// System properties, not specific to any single delegate.

// Android properties.
//
// Android SDK version number. Android system property ro.build.version.sdk.
// E.g., "28".
inline constexpr char kAndroidSdkVersion[] = "tflite.android_sdk_version";
// SoC model. Looked up from database or possibly returned from Android system
// property ro.board.platform, normalized. E.g., "sdm450".
inline constexpr char kSoCModel[] = "tflite.soc_model";
// SoC vendor. Looked up from database. E.g., "qualcomm".
inline constexpr char kSoCVendor[] = "tflite.soc_vendor";
// Device manufacturer. Android API android.os.Build.MANUFACTURER, normalized.
// E.g., "google".
inline constexpr char kManufacturer[] = "tflite.manufacturer";
// Device model. Android API android.os.Build.MODEL, normalized.
// E.g., "pixel_2".
inline constexpr char kDeviceModel[] = "tflite.device_model";
// Device name. Android API android.os.Build.DEVICE, normalized.
// E.g., "walleye".
inline constexpr char kDeviceName[] = "tflite.device_name";

// GPU-related properties.
//
// OpenGL ES version. E.g., 3.2.
inline constexpr char kOpenGLESVersion[] = "tflite.opengl_es_version";
// GPU model, result of querying GL_RENDERER, normalized. E.g.,
// "adreno_(tm)_505".
inline constexpr char kGPUModel[] = "tflite.gpu_model";
// GPU vendor, normalized. E.g., "adreno_(tm)_505".
inline constexpr char kGPUVendor[] = "tflite.gpu_vendor";
// OpenGL driver version, result of querying GL_VERSION. E.g.,
// "opengl_es_3.2_v@328.0_(git@6fb5a5b,_ife855c4895)_(date:08/21/18)"
inline constexpr char kOpenGLDriverVersion[] = "tflite.opengl_driver_version";

// Allowlist use case. This property is used to allow joining multiple lists
// into a single decision tree.
inline constexpr char kUseCase[] = "tflite.use_case";

// NNAPI-related properties.
//
// NNAPI accelerator name, returned by ANeuralNetworksDevice_getName. E.g.,
// "qti-dsp".
inline constexpr char kNNAPIAccelerator[] = "tflite.nnapi_accelerator";
// NNAPI accelerator feature level, returned by
// ANeuralNetworksDevice_getFeatureLevel. E.g., 29. Actual variables are named
// "tflite.nnapi_feature_level.<accelerator name>", e.g.,
// "tflite.nnapi_feature_level.qti-dsp".
inline constexpr char kNNAPIFeatureLevelPrefix[] = "tflite.nnapi_feature_level";

namespace gpu {
// GPU-delegate derived properties.

// Whether the GPU delegate works in general.
// Possible values are ("", "SUPPORTED", "UNSUPPORTED"). An empty value for
// this field means that the device is unsupported.
inline constexpr char kStatus[] = "tflite.gpu.status";

inline constexpr char kStatusSupported[] = "SUPPORTED";
inline constexpr char kStatusUnknown[] = "UNKNOWN";
inline constexpr char kStatusUnsupported[] = "UNSUPPORTED";

enum class CompatibilityStatus {
  kUnknown = 0,
  kSupported,
  kUnsupported,
};

}  // namespace gpu

namespace metadata {
// Latency for model to run.
inline constexpr char kLatency[] = "tflite.latency";

// Frame time for one frame through model.
inline constexpr char kFrameTime[] = "tflite.frame_time";

}  // namespace metadata

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_VARIABLES_H_
