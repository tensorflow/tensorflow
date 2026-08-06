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

load("//tensorflow/tools/toolchains/remote_config:containers.bzl", "container_digests")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# Remote build platforms.
# Each of the platform rules here provide a platform definition that is bound to a docker image.
# The result of the starlark configuration is checked into
# //third_party/toolchains/preconfig.

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cpu.
platform(
    name = "rbe_ubuntu16.04",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-ubuntu16.04@%s" % container_digests["ubuntu16.04"],
        "Pool": "default",
    },
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cpu-centos6.
platform(
    name = "rbe_centos6",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-centos6@%s" % container_digests["centos6"],
        "Pool": "default",
    },
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.0-cudnn7-ubuntu14.04.
platform(
    name = "rbe_cuda10.0-cudnn7-ubuntu14.04",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu14.04@%s" % container_digests["cuda10.0-cudnn7-ubuntu14.04"],
        "Pool": "default",
    },
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.0-cudnn7-centos6.
platform(
    name = "rbe_cuda10.0-cudnn7-centos6",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-centos6@%s" % container_digests["cuda10.0-cudnn7-centos6"],
        "Pool": "default",
    },
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.ubuntu16.04-manylinux2010.
platform(
    name = "rbe_ubuntu16.04-manylinux2010",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-ubuntu16.04-manylinux2010@%s" % container_digests["ubuntu16.04-manylinux2010"],
        "Pool": "default",
    },
)
