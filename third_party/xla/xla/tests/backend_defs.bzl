# Copyright 2026 The OpenXLA Authors.
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
"""GPU backends constants and helpers for XLA testing."""

load("//xla/tests:plugin.bzl", "plugins")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")
load("//xla/tsl:tsl.bzl", "if_google")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)

visibility(DEFAULT_LOAD_VISIBILITY)

# Possible backend values for the GPU family.
NVIDIA_GPU_BACKENDS = [
    "nvgpu_any",
    "p100",
    "v100",
    "a100",
    "h100",
    "b200",
    "gb200",
    "gb300",
] + if_google([], ["rtx6000pro"])

# The generic "gpu" backend includes the actual backends in this list.
NVIDIA_GPU_DEFAULT_BACKENDS = [
    "nvgpu_any",
    "a100",
    "h100",
    "b200",
    "gb200",
    "gb300",
] + if_google([], ["rtx6000pro"])

AMD_GPU_DEFAULT_BACKENDS = ["amdgpu_any"]

INTEL_GPU_DEFAULT_BACKENDS = ["intelgpu_any"]

DEFAULT_BACKENDS = ["cpu"] + NVIDIA_GPU_DEFAULT_BACKENDS + AMD_GPU_DEFAULT_BACKENDS + INTEL_GPU_DEFAULT_BACKENDS

GPU_BACKENDS = NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS + INTEL_GPU_DEFAULT_BACKENDS

GPU_DEFAULT_BACKENDS = NVIDIA_GPU_DEFAULT_BACKENDS

DEFAULT_DISABLED_BACKENDS = []

ALL_HARDWARE_BACKENDS = ["cpu"] + GPU_BACKENDS + list(plugins.keys())

ALL_BACKENDS = ["interpreter"] + ALL_HARDWARE_BACKENDS

# buildifier: disable=function-docstring
def prepare_nvidia_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args, common_tags):
    # Expand "gpu" backend name into device specific backend names unless it's tagged rocm-only or oneapi-only
    new_backends = [name for name in backends if name != "gpu"]
    if len(new_backends) < len(backends) and "rocm-only" not in common_tags and "oneapi-only" not in common_tags:
        new_backends.extend(NVIDIA_GPU_DEFAULT_BACKENDS)

    new_disabled_backends = [name for name in disabled_backends if name != "gpu"]
    if len(new_disabled_backends) < len(disabled_backends):
        new_disabled_backends.extend(NVIDIA_GPU_BACKENDS)

    new_backend_tags = {key: value for key, value in backend_tags.items() if key != "gpu"}
    gpu_backend_tags = backend_tags.get("gpu", tf_gpu_tests_tags())
    for key in NVIDIA_GPU_BACKENDS:
        new_backend_tags.setdefault(key, gpu_backend_tags[:])

    new_backend_args = {key: value for key, value in backend_args.items() if key != "gpu"}
    if "gpu" in backend_args:
        for key in NVIDIA_GPU_BACKENDS:
            new_backend_args.setdefault(key, backend_args["gpu"])

    # Disable backends that don't meet the device requirements.
    sm_requirements = {
        "nvgpu_any": (0, 0),
        "p100": (6, 0),
        "v100": (7, 0),
        "a100": (8, 0),
        "h100": (9, 0),
        "b200": (10, 0),
        "gb200": (10, 0),
        "gb300": (10, 3),
        "rtx6000pro": (12, 0),
    }
    for gpu_backend in NVIDIA_GPU_BACKENDS:
        all_tags = new_backend_tags[gpu_backend]
        requires_gpu = [t for t in all_tags if t.startswith("requires-gpu-")]

        # full tag is meaningless in OSS, when run at google, this routes tests to full GPUs
        # instead of mig partition(s).
        requires_full_gpu = if_google("full" in new_backend_tags[gpu_backend], False)
        requires_sm, only = None, False
        num_gpus = None
        for tag in requires_gpu:
            if ":" in tag:  # Multi-GPU tests are suffixed with colon and number of GPUs.
                tag, suffix = tag.split(":")  # Remove the suffix from the tag for further parsing.
                parsed_num_gpus = int(suffix)
                if num_gpus and num_gpus != parsed_num_gpus:
                    fail("Inconsistent number of GPUs: %d vs %d" % (num_gpus, parsed_num_gpus))
                num_gpus = parsed_num_gpus
            if tag.startswith("requires-gpu-sm"):
                version = tag.split("-")[2][2:]
                sm = (int(version[:-1]), int(version[-1]))
                if not requires_sm or sm < requires_sm:
                    requires_sm = sm
                if tag.endswith("-only"):
                    only = True
        if only:
            disable = requires_sm != sm_requirements[gpu_backend]
        else:
            disable = requires_sm and requires_sm > sm_requirements[gpu_backend]

        if disable:
            new_disabled_backends.append(gpu_backend)
        else:
            sm_major, sm_minor = sm_requirements[gpu_backend]
            full = "-full" if requires_full_gpu else ""
            sm_tag = "requires-gpu-nvidia" if sm_major == 0 else "requires-gpu-sm%s%s%s-only" % (sm_major, sm_minor, full)
            if num_gpus:
                sm_tag += ":%d" % num_gpus
            new_backend_tags[gpu_backend] = [t for t in all_tags if t not in requires_gpu]
            new_backend_tags[gpu_backend].append(sm_tag)
            new_backend_tags[gpu_backend].append("cuda-only")

    return new_backends, new_disabled_backends, new_backend_tags, new_backend_args

# buildifier: disable=function-docstring
def prepare_amd_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args, common_tags):
    new_backends = [name for name in backends if name != "gpu"]

    # Expand "gpu" backend name into device specific backend names unless it's tagged cuda-only or oneapi-only
    if len(new_backends) < len(backends) and "cuda-only" not in common_tags and "oneapi-only" not in common_tags:
        new_backends.extend(AMD_GPU_DEFAULT_BACKENDS)

    new_disabled_backends = [name for name in disabled_backends if name != "gpu"]
    if len(new_disabled_backends) < len(disabled_backends):
        new_disabled_backends.extend(AMD_GPU_DEFAULT_BACKENDS)

    new_backend_tags = {
        key: value
        for key, value in backend_tags.items()
        if key not in ["gpu"] + NVIDIA_GPU_BACKENDS
    }

    gpu_backend_tags = backend_tags.get("gpu", [])
    nvidia_tags = []
    for key in gpu_backend_tags:
        if key.startswith("requires-"):
            nvidia_tags.append(key)

    for key in nvidia_tags:
        gpu_backend_tags.remove(key)

    for key in AMD_GPU_DEFAULT_BACKENDS:
        new_backend_tags.setdefault(key, gpu_backend_tags[:])

    for backend in AMD_GPU_DEFAULT_BACKENDS:
        new_backend_tags[backend].append("requires-gpu-amd")
        new_backend_tags[backend].append("rocm-only")

    return new_backends, new_disabled_backends, new_backend_tags, backend_args

# buildifier: disable=function-docstring
def prepare_intel_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args, common_tags):
    new_backends = [name for name in backends if name != "gpu"]

    # Expand "gpu" backend name into device specific backend names unless it's tagged cuda-only or rocm-only
    if len(new_backends) < len(backends) and "cuda-only" not in common_tags and "rocm-only" not in common_tags:
        new_backends.extend(INTEL_GPU_DEFAULT_BACKENDS)

    new_disabled_backends = [name for name in disabled_backends if name != "gpu"]
    if len(new_disabled_backends) < len(disabled_backends):
        new_disabled_backends.extend(INTEL_GPU_DEFAULT_BACKENDS)

    new_backend_tags = {
        key: value
        for key, value in backend_tags.items()
        if key not in ["gpu"] + NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS
    }

    gpu_backend_tags = backend_tags.get("gpu", [])
    nvidia_tags = []
    for key in gpu_backend_tags:
        if key.startswith("requires-"):
            nvidia_tags.append(key)

    for key in nvidia_tags:
        gpu_backend_tags.remove(key)

    for key in INTEL_GPU_DEFAULT_BACKENDS:
        new_backend_tags.setdefault(key, gpu_backend_tags[:])

    for backend in INTEL_GPU_DEFAULT_BACKENDS:
        new_backend_tags[backend].append("requires-gpu-intel")
        new_backend_tags[backend].append("oneapi-only")

    return new_backends, new_disabled_backends, new_backend_tags, backend_args

# buildifier: disable=function-docstring
def prepare_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args, common_tags):
    nvidia_backends = [
        backend
        for backend in backends
        if backend in ["gpu"] + NVIDIA_GPU_BACKENDS
    ]
    amd_backends = [
        backend
        for backend in backends
        if backend in ["gpu"] + AMD_GPU_DEFAULT_BACKENDS
    ]
    intel_backends = [
        backend
        for backend in backends
        if backend in ["gpu"] + INTEL_GPU_DEFAULT_BACKENDS
    ]
    other_backends = [
        backend
        for backend in backends
        if backend not in ["gpu"] + NVIDIA_GPU_BACKENDS + AMD_GPU_DEFAULT_BACKENDS + INTEL_GPU_DEFAULT_BACKENDS
    ]

    nvidia_backends, nvidia_disabled_backends, nvidia_backend_tags, nvidia_backend_args = \
        prepare_nvidia_gpu_backend_data(nvidia_backends, disabled_backends, backend_tags, backend_args, common_tags)
    amd_backends, amd_disabled_backends, amd_backend_tags, amd_backend_args = \
        prepare_amd_gpu_backend_data(amd_backends, disabled_backends, backend_tags, {}, common_tags)
    intel_backends, intel_disabled_backends, intel_backend_tags, intel_backend_args = \
        prepare_intel_gpu_backend_data(intel_backends, disabled_backends, backend_tags, {}, common_tags)

    new_backends = [
        backend
        for backend in nvidia_backends + amd_backends + intel_backends + other_backends
    ]

    disabled_backends = nvidia_disabled_backends + amd_disabled_backends + intel_disabled_backends

    backend_tags = nvidia_backend_tags | amd_backend_tags | intel_backend_tags

    backend_args = nvidia_backend_args | amd_backend_args | intel_backend_args

    return new_backends, disabled_backends, backend_tags, backend_args
