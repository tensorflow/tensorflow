# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Definition of the C++ library for object tracking
cc_library(
    name = "object_tracking_lib",
    srcs = [
        "frame_pair.cc",
        "keypoint_detector.cc",
        "logging.cc",
        "object_detector.cc",
        "object_tracker.cc",
        "optical_flow.cc",
        "time_log.cc",
        "tracked_object.cc",
    ],
    hdrs = [
        "config.h",
        "flow_cache.h",
        "frame_pair.h",
        "geom.h",
        "image.h",
        "image-inl.h",
        "image_data.h",
        "image_utils.h",
        "integral_image.h",
        "keypoint.h",
        "keypoint_detector.h",
        "logging.h",
        "object_detector.h",
        "object_model.h",
        "object_tracker.h",
        "optical_flow.h",
        "time_log.h",
        "tracked_object.h",
        "utils.h",
    ],
    copts = [
        "-std=c++17",
    ],
    defines = [
        "STANDALONE_DEMO_LIB",
    ],
)

# Definition of the test for the object tracking library
cc_binary(
    name = "object_tracking_test",
    testonly = True,
    srcs = [
        "object_tracking_test.cc",
    ],
    copts = [
        "-std=c++17",
    ],
    deps = [
        ":object_tracking_lib",
        "@com_google_googletest//:gtest_main",
    ],
)
