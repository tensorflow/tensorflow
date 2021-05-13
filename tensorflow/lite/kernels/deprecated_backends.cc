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

namespace tflite {

// Include this target as a dependency in order to define this function for
// CpuBackendContext. Its use is to control execution of deprecated paths
// by providing a symbol definition for otherwise "weak" symbol
// declarations in CpuBackendContext.
extern bool UseGemmlowpOnX86() { return true; }

}  // namespace tflite
