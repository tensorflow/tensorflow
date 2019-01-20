/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOCO_TFLITE_WHITELISTED_FLEX_OPS_H_
#define TENSORFLOW_LITE_TOCO_TFLITE_WHITELISTED_FLEX_OPS_H_

#include <string>

namespace toco {
namespace tflite {

// Whether the given op has been statically whitelisted for flex export.
//
// This static whitelist is formed by the intersection of ops supported by
// TensorFlowMobile on both iOS and Android. As the converter is likely running
// on a host that has the full suite of TensorFlow ops available, we use this
// static whitelist to ensure compatibility when deploying to a mobile device.
// TODO(b/118389105): Automate generation of the whitelisted flex ops.
bool IsWhitelistedFlexOp(const std::string& tensorflow_op_name);

}  // namespace tflite
}  // namespace toco

#endif  // TENSORFLOW_LITE_TOCO_TFLITE_WHITELISTED_FLEX_OPS_H_
