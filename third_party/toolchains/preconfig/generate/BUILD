licenses(["restricted"])

load(":generate.bzl", "tensorflow_rbe_config")

tensorflow_rbe_config(
    name = "ubuntu16.04-py3-clang",
    compiler = "clang",
    os = "ubuntu16.04",
    python_version = "3",
)

tensorflow_rbe_config(
    name = "centos6-py-gcc7",
    build_bazel_src = True,
    compiler = "gcc",
    compiler_prefix = "/opt/rh/devtoolset-7/root/usr/bin",
    os = "centos6",
    python_version = "2",
)

tensorflow_rbe_config(
    name = "centos6-py3-gcc7",
    build_bazel_src = True,
    compiler = "gcc",
    compiler_prefix = "/opt/rh/devtoolset-7/root/usr/bin",
    os = "centos6",
    python_version = "3",
)

tensorflow_rbe_config(
    name = "centos6-py3-gcc7-cuda10.0-cudnn7-tensorrt5",
    build_bazel_src = True,
    compiler = "gcc",
    compiler_prefix = "/opt/rh/devtoolset-7/root/usr/bin",
    cuda_version = "10.0",
    cudnn_install_path = "/usr/local/cuda-10.0",
    cudnn_version = "7",
    os = "centos6",
    python_version = "3",
    tensorrt_install_path = "/usr",
    tensorrt_version = "5",
)

tensorflow_rbe_config(
    name = "centos6-py3-gcc7-cuda10.1-cudnn7-tensorrt5",
    build_bazel_src = True,
    compiler = "gcc",
    compiler_prefix = "/opt/rh/devtoolset-7/root/usr/bin",
    cuda_version = "10.1",
    cudnn_install_path = "/usr/local/cuda-10.1",
    cudnn_version = "7",
    os = "centos6",
    python_version = "3",
    tensorrt_install_path = "/usr",
    tensorrt_version = "5",
)

tensorflow_rbe_config(
    name = "ubuntu14.04-py3-gcc7-cuda10.0-cudnn7-tensorrt5",
    compiler = "gcc-7",
    cuda_version = "10.0",
    cudnn_version = "7",
    os = "ubuntu14.04",
    python_version = "3",
    tensorrt_version = "5",
)

tensorflow_rbe_config(
    name = "ubuntu14.04-py3-gcc-cuda10.0-cudnn7-tensorrt5",
    compiler = "gcc",
    cuda_version = "10.0",
    cudnn_version = "7",
    os = "ubuntu14.04",
    python_version = "3",
    tensorrt_version = "5",
)

tensorflow_rbe_config(
    name = "ubuntu16.04-py-gcc7_manylinux2010",
    compiler = "/dt7/usr/bin/gcc",
    compiler_prefix = "/usr/bin",
    os = "ubuntu16.04-manylinux2010",
    python_version = "2",
)

tensorflow_rbe_config(
    name = "ubuntu16.04-py3-gcc7_manylinux2010",
    compiler = "/dt7/usr/bin/gcc",
    compiler_prefix = "/usr/bin",
    os = "ubuntu16.04-manylinux2010",
    python_version = "3.6",
)

tensorflow_rbe_config(
    name = "ubuntu16.04-py3-clang_manylinux2010-cuda10.0-cudnn7-tensorrt5.1",
    compiler = "/clang_r373795/bin/clang",
    cuda_version = "10.0",
    cudnn_version = "7",
    os = "ubuntu16.04-manylinux2010",
    python_version = "3",
    sysroot = "/dt7",
    tensorrt_install_path = "/usr",
    tensorrt_version = "5.1",
)

tensorflow_rbe_config(
    name = "ubuntu16.04-py3-gcc7_manylinux2010-cuda10.1-cudnn7-tensorrt6.0",
    compiler = "/dt7/usr/bin/gcc",
    compiler_prefix = "/usr/bin",
    cuda_version = "10.1",
    cudnn_version = "7",
    os = "ubuntu16.04-manylinux2010",
    python_version = "3",
    tensorrt_install_path = "/usr",
    tensorrt_version = "6.0",
)

tensorflow_rbe_config(
    name = "ubuntu16.04-py3-clang_manylinux2010-cuda10.1-cudnn7-tensorrt6.0",
    compiler = "/clang_r373795/bin/clang",
    cuda_version = "10.1",
    cudnn_version = "7",
    os = "ubuntu16.04-manylinux2010",
    python_version = "3",
    sysroot = "/dt7",
    tensorrt_install_path = "/usr",
    tensorrt_version = "6.0",
)
