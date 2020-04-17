# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

exports_files(["coreml_delegate.h"])

objc_library(
    name = "coreml_executor",
    srcs = ["coreml_executor.mm"],
    hdrs = ["coreml_executor.h"],
    sdk_frameworks = [
        "Foundation",
        "UIKit",
        "CoreML",
    ],
    deps = [
        ":mlmodel_proto_cc",
    ],
)

cc_library(
    name = "mlmodel_proto_cc",
    deps = [
        "@coremltools//:mlmodel_cc_proto",
    ],
)

objc_library(
    name = "coreml_delegate",
    srcs = ["coreml_delegate.mm"],
    hdrs = ["coreml_delegate.h"],
    deps = [
        ":coreml_delegate_kernel",
        ":mlmodel_proto_cc",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/delegates/coreml/builders:op_builder",
    ],
)

objc_library(
    name = "coreml_delegate_kernel",
    srcs = [
        "coreml_delegate_kernel.mm",
    ],
    hdrs = [
        "coreml_delegate_kernel.h",
    ],
    deps = [
        ":coreml_executor",
        ":mlmodel_proto_cc",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/delegates/coreml/builders:op_builder",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/kernels/internal:types",
    ],
)
