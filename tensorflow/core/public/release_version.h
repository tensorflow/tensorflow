/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PUBLIC_RELEASE_VERSION_H_
#define TENSORFLOW_CORE_PUBLIC_RELEASE_VERSION_H_

// A cc_library //third_party/tensorflow/core/public:release_version provides
// defines with the version data from //third_party/tensorflow/tf_version.bzl.
// The version suffix can be set by passing the build parameters
// --repo_env=ML_WHEEL_BUILD_DATE=<date> and
// --repo_env=ML_WHEEL_VERSION_SUFFIX=<suffix>.
// To update the project version, update tf_version.bzl.

#define _TF_STR_HELPER(x) #x
#define _TF_STR(x) _TF_STR_HELPER(x)

// e.g. "0.5.0" or "0.6.0-alpha".
#define TF_VERSION_STRING                                            \
  (_TF_STR(TF_MAJOR_VERSION) "." _TF_STR(TF_MINOR_VERSION) "." _TF_STR( \
      TF_PATCH_VERSION) TF_VERSION_SUFFIX)

#endif  // TENSORFLOW_CORE_PUBLIC_RELEASE_VERSION_H_
