/* 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_ROCM
#ifndef TENSORFLOW_RTGLIB_ATTR_TO_RTG_
#define TENSORFLOW_RTGLIB_ATTR_TO_RTG_

#ifndef TENSORFLOW_RTGLIB_COMMON_HEADER_
#include "common_headers.h"
#endif  // TENSORFLOW_RTGLIB_COMMON_HEADER_

namespace tensorflow {
namespace rtglib {
namespace convert {
void GetProgram(const NameAttrList&, void **, int&);
void EvalProgram(void*, Tensor*, std::vector<const Tensor*>&, bool, void*, int);
void GetOutputShape(void *, TensorShape&);

} // namspace convert
} // namespace rtglib
} // namespace tensorflow 

#endif // TENSORFLOW_RTGLIB_ATTR_TO_RTG_
#endif // TENSORFLOW_USE_ROCM
