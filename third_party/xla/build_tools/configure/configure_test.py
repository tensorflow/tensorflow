# Copyright 2024 The OpenXLA Authors.
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
from absl.testing import absltest

from xla.build_tools import test_utils
from xla.build_tools.configure import configure


XLAConfigOptions = configure.XLAConfigOptions
DiscoverablePathsAndVersions = configure.DiscoverablePathsAndVersions
Backend = configure.Backend
HostCompiler = configure.HostCompiler
CudaCompiler = configure.CudaCompiler
OS = configure.OS

_PYTHON_BIN_PATH = "/usr/bin/python3"
_CLANG_PATH = "/usr/lib/llvm-17/bin/clang"
_GCC_PATH = "/usr/bin/gcc"
_COMPILER_OPTIONS = ("-Wno-sign-compare",)

# CUDA specific paths and versions
_CUDA_SPECIFIC_PATHS_AND_VERSIONS = {
    "cublas_version": "12.3",
    "cuda_toolkit_path": "/usr/local/cuda-12.2",
    "cuda_compute_capabilities": ["7.5"],
    "cudnn_version": "8",
    "ld_library_path": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
    "nccl_version": "2",
}


class ConfigureTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    testdata = (
        test_utils.xla_src_root() / "build_tools" / "configure" / "testdata"
    )

    with (testdata / "clang.bazelrc").open() as f:
      cls.clang_bazelrc_lines = [line.strip() for line in f.readlines()]

    with (testdata / "gcc.bazelrc").open() as f:
      cls.gcc_bazelrc_lines = [line.strip() for line in f.readlines()]

    with (testdata / "cuda_clang.bazelrc").open() as f:
      cls.cuda_clang_bazelrc_lines = [line.strip() for line in f.readlines()]

    with (testdata / "nvcc_clang.bazelrc").open() as f:
      cls.nvcc_clang_bazelrc_lines = [line.strip() for line in f.readlines()]

    with (testdata / "nvcc_gcc.bazelrc").open() as f:
      cls.nvcc_gcc_bazelrc_lines = [line.strip() for line in f.readlines()]

  def test_clang_bazelrc(self):
    config = XLAConfigOptions(
        backend=Backend.CPU,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.CLANG,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.NVCC,
        using_nccl=False,
        using_tensorrt=False,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            clang_path=_CLANG_PATH,
            ld_library_path="",
            clang_major_version=17,
        )
    )

    self.assertEqual(bazelrc_lines, self.clang_bazelrc_lines)

  def test_gcc_bazelrc(self):
    config = XLAConfigOptions(
        backend=Backend.CPU,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.GCC,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.NVCC,
        using_nccl=False,
        using_tensorrt=False,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            gcc_path=_GCC_PATH,
            ld_library_path="",
        )
    )

    self.assertEqual(bazelrc_lines, self.gcc_bazelrc_lines)

  def test_cuda_clang_bazelrc(self):
    config = XLAConfigOptions(
        backend=Backend.CUDA,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.CLANG,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.CLANG,
        using_nccl=False,
        using_tensorrt=False,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            clang_path=_CLANG_PATH,
            clang_major_version=17,
            **_CUDA_SPECIFIC_PATHS_AND_VERSIONS,
        )
    )

    self.assertEqual(bazelrc_lines, self.cuda_clang_bazelrc_lines)

  def test_nvcc_clang_bazelrc(self):
    config = XLAConfigOptions(
        backend=Backend.CUDA,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.CLANG,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.NVCC,
        using_nccl=False,
        using_tensorrt=False,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            clang_path=_CLANG_PATH,
            clang_major_version=17,
            **_CUDA_SPECIFIC_PATHS_AND_VERSIONS,
        )
    )

    self.assertEqual(bazelrc_lines, self.nvcc_clang_bazelrc_lines)

  def test_nvcc_gcc_bazelrc(self):
    config = XLAConfigOptions(
        backend=Backend.CUDA,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.GCC,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.NVCC,
        using_nccl=False,
        using_tensorrt=False,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            gcc_path=_GCC_PATH,
            **_CUDA_SPECIFIC_PATHS_AND_VERSIONS,
        )
    )

    self.assertEqual(bazelrc_lines, self.nvcc_gcc_bazelrc_lines)


if __name__ == "__main__":
  absltest.main()
