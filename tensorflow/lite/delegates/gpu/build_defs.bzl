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

"""Additional build options needed for the GPU Delegate."""

def gpu_delegate_linkopts():
    """Additional link options needed when linking in the GPU Delegate."""
    return select({
        "//tensorflow:android": [
            "-lEGL",
            # We don't need to link libGLESv3, because if it exists,
            # it is a symlink to libGLESv2.
            # See Compatibility Definition Document:
            # https://source.android.com/compatibility/10/android-10-cdd#7_1_4_1_opengl_es
            "-lGLESv2",
        ],
        "//conditions:default": [],
    })

def tflite_angle_heapcheck_deps():
    # copybara:uncomment_begin(google-only)
    # return select({
    # "//tensorflow/lite/delegates/gpu:tflite_gpu_angle": [
    # "@com_google_googletest//:gtest_main_no_heapcheck",
    # ],
    # "//conditions:default": [
    # "@com_google_googletest//:gtest_main",
    # ],
    # })
    # copybara:uncomment_end
    # copybara:comment_begin(oss-only)
    return ["@com_google_googletest//:gtest_main"]
    # copybara:comment_end

def gtest_main_no_heapcheck_deps():
    # copybara:uncomment_begin(google-only)
    # return ["@com_google_googletest//:gtest_main_no_heapcheck"]
    # copybara:uncomment_end
    # copybara:comment_begin(oss-only)
    return ["@com_google_googletest//:gtest_main"]
    # copybara:comment_end
