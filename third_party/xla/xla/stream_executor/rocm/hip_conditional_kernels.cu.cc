/* Copyright 2023 The OpenXLA Authors.

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

#include <string_view>

namespace stream_executor::gpu {

std::string_view GetSetIfConditionKernel() { return "<unsupported>"; }
std::string_view GetSetIfElseConditionKernel() { return "<unsupported>"; }
std::string_view GetSetCaseConditionKernel() { return "<unsupported>"; }
std::string_view GetSetForConditionKernel() { return "<unsupported>"; }
std::string_view GetSetWhileConditionKernel() { return "<unsupported>"; }

}  // namespace stream_executor::gpu
