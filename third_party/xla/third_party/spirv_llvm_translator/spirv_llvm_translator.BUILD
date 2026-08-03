# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

cc_library(
    name = "spirv_llvm_translator",
    srcs = glob([
        "lib/SPIRV/libSPIRV/*.cpp",
        "lib/SPIRV/libSPIRV/*.hpp",
        "lib/SPIRV/libSPIRV/*.h",
        "lib/SPIRV/Mangler/*.cpp",
        "lib/SPIRV/Mangler/*.h",
        "lib/SPIRV/*.cpp",
        "lib/SPIRV/*.hpp",
        "lib/SPIRV/*.h",
    ]),
    hdrs = glob(["include/*"]),
    includes = [
        "include/",
        "lib/SPIRV/",
        "lib/SPIRV/Mangler/",
        "lib/SPIRV/libSPIRV/",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Demangle",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@spirv_headers//:spirv_cpp_headers",
    ],
)
