import os
import subprocess

import triton


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def test_address_sanitizer():
    if not is_hip():
        return  #not supported on NV backend

    # It is recommended to disable various memory caching strategies both within the ROCm stack and PyTorch
    # This will give the address sanitizer the best chance at finding the memory fault where it originates,
    # otherwise it could be masked by writing past the end of a cached block within a larger allocation.
    os.environ["HSA_DISABLE_FRAGMENT_ALLOCATOR"] = "1"
    os.environ["AMD_PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"
    os.environ["TRITON_ENABLE_ASAN"] = "1"

    # HSA_XNACK here is required to set the xnack+ setting for the GPU at runtime.
    # If it is not set and the default xnack setting of the system is xnack-
    # a runtime error something like "No kernel image found" will occur. The system
    # xnack setting can be found through rocminfo. xnack+ is required for ASAN.
    # More information about xnack in general can be found here:
    # https://llvm.org/docs/AMDGPUUsage.html#target-features
    # https://rocm.docs.amd.com/en/docs-6.1.0/conceptual/gpu-memory.html
    os.environ["HSA_XNACK"] = "1"

    # Disable buffer ops given it has builtin support for out of bound access.
    os.environ["AMDGCN_USE_BUFFER_OPS"] = "0"

    out = subprocess.Popen(["python", "address_sanitizer_helper.py"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    assert "Begin function __asan_report" in out.stdout.read().decode()
    assert "heap-buffer-overflow" in out.stderr.read().decode()
