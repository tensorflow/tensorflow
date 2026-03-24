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
#ifndef TENSORFLOW_LITE_VERSION_H_
#define TENSORFLOW_LITE_VERSION_H_

// TODO (b/446006189): Generate the following file from a template that aligns
// LiteRT versioning.
// Update the following version number with every LiteRT release following the
// semantic versioning https://semver.org/
#ifndef TF_VERSION_STRING
#ifndef TF_MAJOR_VERSION
#define TF_MAJOR_VERSION 2
#endif
#ifndef TF_MINOR_VERSION
#define TF_MINOR_VERSION 19
#endif
#ifndef TF_PATCH_VERSION
#define TF_PATCH_VERSION 0
#endif
#ifndef TF_VERSION_SUFFIX
#define TF_VERSION_SUFFIX ""
#endif
#ifndef TF_STR_HELPER
#define TF_STR_HELPER(x) #x
#define TF_STR(x) TF_STR_HELPER(x)
#endif
#define TF_VERSION_STRING                                            \
  (TF_STR(TF_MAJOR_VERSION) "." TF_STR(TF_MINOR_VERSION) "." TF_STR( \
      TF_PATCH_VERSION) TF_VERSION_SUFFIX)
#endif  // TF_VERSION_STRING

// The version number of the Schema. Ideally all changes will be backward
// compatible. If that ever changes, we must ensure that version is the first
// entry in the new tflite root so that we can see that version is not 1.
#define TFLITE_SCHEMA_VERSION (3)

// TensorFlow Lite Runtime version.
// This value is currently shared with that of TensorFlow.
#define TFLITE_VERSION_STRING TF_VERSION_STRING

// TensorFlow Lite Extension APIs version.
// This is the semantic version number for the custom op and delegate APIs.
// This value is currently shared with that of TensorFlow Lite.
#define TFLITE_EXTENSION_APIS_VERSION_STRING TFLITE_VERSION_STRING

#endif  // TENSORFLOW_LITE_VERSION_H_
