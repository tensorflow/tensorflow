/* Copyright 2019 Google LLC. All Rights Reserved.

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

// Temporary dotprod-detection code until we can rely on getauxval.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_DETECT_ARM_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_DETECT_ARM_H_

namespace ruy {

// On A64, returns true if the dotprod extension is present.
// On other architectures, returns false unconditionally.
bool DetectDotprod();

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_DETECT_ARM_H_
