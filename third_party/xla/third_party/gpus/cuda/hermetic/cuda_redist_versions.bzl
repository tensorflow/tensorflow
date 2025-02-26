# Copyright 2024 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hermetic CUDA redistribution versions."""

CUDA_REDIST_PATH_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
CUDNN_REDIST_PATH_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"

CUDA_REDIST_JSON_DICT = {
    "11.8": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_11.8.0.json",
        "941a950a4ab3b95311c50df7b3c8bca973e0cdda76fc2f4b456d2d5e4dac0281",
    ],
    "12.1.1": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.1.1.json",
        "bafea3cb83a4cf5c764eeedcaac0040d0d3c5db3f9a74550da0e7b6ac24d378c",
    ],
    "12.2.0": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.2.0.json",
        "d883762c6339c8ebb3ffb072facc8f7265cd257d2db16a475fff9a9306ecea89",
    ],
    "12.3.1": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.3.1.json",
        "b3cc4181d711cf9b6e3718f323b23813c24f9478119911d7b4bceec9b437dbc3",
    ],
    "12.3.2": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.3.2.json",
        "1b6eacf335dd49803633fed53ef261d62c193e5a56eee5019e7d2f634e39e7ef",
    ],
    "12.4.0": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.4.0.json",
        "a4f496b8d5299939b34c9ef88dc4274821f8c9451b2d7c9bcee53166932da067",
    ],
    "12.4.1": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.4.1.json",
        "9cd815f3b71c2e3686ef2219b7794b81044f9dcefaa8e21dacfcb5bc4d931892",
    ],
    "12.5.0": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.5.0.json",
        "166664b520bfe51f27abcc8c7a934f4cb6ea287f8c399b5f8255f6f4d214569a",
    ],
    "12.5.1": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.5.1.json",
        "7ab9c76014ae4907fa1b51738af599607a5fd8ca3a5c4bb4c3b31338cc642a93",
    ],
    "12.6.0": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.0.json",
        "87740b01676b3d18982982ab96ec7fa1a626d03a96df070a6b0f258d01ff5fab",
    ],
    "12.6.1": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.1.json",
        "22ddfeb81a6f9cee4a708a2e3b4db1c36c7db0a1daa1f33f9c7f2f12a1e790de",
    ],
    "12.6.2": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.2.json",
        "8056da1f5acca8e613da1349d9b8782b774ad0254e3eddcc95734ded4d33f2df",
    ],
    "12.6.3": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3.json",
        "9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf",
    ],
    "12.8.0": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.0.json",
        "daa0d766b36feaa933592162c27be5fb63b68fc547ca6886c160a35d96ee8891",
    ],
}

CUDNN_REDIST_JSON_DICT = {
    "8.6": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_8.6.0.json",
        "7f6f50bed4fd8216dc10d6ef505771dc0ecc99cce813993ab405cb507a21d51d",
    ],
    "8.9.4.25": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_8.9.4.25.json",
        "02258dba8384860c9230fe3c78522e7bd8e350e461ccd37a8d932cb64127ba57",
    ],
    "8.9.6": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_8.9.6.json",
        "6069ef92a2b9bb18cebfbc944964bd2b024b76f2c2c35a43812982e0bc45cf0c",
    ],
    "8.9.7.29": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_8.9.7.29.json",
        "a0734f26f068522464fa09b2f2c186dfbe6ad7407a88ea0c50dd331f0c3389ec",
    ],
    "9.1.1": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.1.1.json",
        "d22d569405e5683ff8e563d00d6e8c27e5e6a902c564c23d752b22a8b8b3fe20",
    ],
    "9.2.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.2.0.json",
        "6852eb279b95d2b5775f7a7737ec133bed059107f863cdd8588f3ae6f13eadd7",
    ],
    "9.2.1": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.2.1.json",
        "9a4198c59b2e66b2b115a736ebe4dc8f3dc6d78161bb494702f824da8fc77b99",
    ],
    "9.3.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.3.0.json",
        "d17d9a7878365736758550294f03e633a0b023bec879bf173349bfb34781972e",
    ],
    "9.4.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.4.0.json",
        "6eeaafc5cc3d4bb2f283e6298e4c55d4c59d7c83c5d9fd8721a2c0e55aee4e54",
    ],
    "9.5.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.5.0.json",
        "3939f0533fdd0d3aa7edd1ac358d43da18e438e5d8f39c3c15bb72519bad7fb5",
    ],
    "9.5.1": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.5.1.json",
        "a5484eef575bbb1fd4f96136cf12244ebc194b661f5ae9ed3b8aaa07e06434b1",
    ],
    "9.6.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.6.0.json",
        "6dd9a931d981fe5afc7e7ed0c422a4035b1411db4e28a39cf2429e62e3efcd3e",
    ],
    "9.7.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.7.0.json",
        "e715c1d028585d228c4678c2cdc5ad9a34fde54515a1c52aa60e36021a90dd90",
    ],
    "9.7.1": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.7.1.json",
        "f9bc411a4908f0931e7323f89049e3a38453632c4ac5f4aa3220af69ddded9dc",
    ],
}

CUDA_12_NCCL_WHEEL_DICT = {
    "x86_64-unknown-linux-gnu": {
        "version": "2.25.1",
        "url": "https://files.pythonhosted.org/packages/11/0c/8c78b7603f4e685624a3ea944940f1e75f36d71bd6504330511f4a0e1557/nvidia_nccl_cu12-2.25.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "sha256": "362aed5963fb9ea2ed2f264409baae30143498fd0e5c503aeaa1badd88cdc54a",
    },
    "aarch64-unknown-linux-gnu": {
        "version": "2.25.1",
        "url": "https://files.pythonhosted.org/packages/4b/28/f62adab24f2d4b2165b22145af56a7598ab535feb6ccd172f76b9106ebaa/nvidia_nccl_cu12-2.25.1-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        "sha256": "4ab428bc915785cc66e8c57cb34c7a64cf739c46702b8db748b6ad6cc7180cf8",
    },
}

CUDA_11_NCCL_WHEEL_DICT = {
    "x86_64-unknown-linux-gnu": {
        "version": "2.21.5",
        "url": "https://files.pythonhosted.org/packages/ac/9a/8b6a28b3b87d5fddab0e92cd835339eb8fbddaa71ae67518c8c1b3d05bae/nvidia_nccl_cu11-2.21.5-py3-none-manylinux2014_x86_64.whl",
        "sha256": "49d8350629c7888701d1fd200934942671cb5c728f49acc5a0b3a768820bed29",
    },
}

CUDA_NCCL_WHEELS = {
    "11.8": CUDA_11_NCCL_WHEEL_DICT,
    "12.1.0": CUDA_12_NCCL_WHEEL_DICT,
    "12.1.1": CUDA_12_NCCL_WHEEL_DICT,
    "12.2.0": CUDA_12_NCCL_WHEEL_DICT,
    "12.3.1": CUDA_12_NCCL_WHEEL_DICT,
    "12.3.2": CUDA_12_NCCL_WHEEL_DICT,
    "12.4.0": CUDA_12_NCCL_WHEEL_DICT,
    "12.4.1": CUDA_12_NCCL_WHEEL_DICT,
    "12.5.0": CUDA_12_NCCL_WHEEL_DICT,
    "12.5.1": CUDA_12_NCCL_WHEEL_DICT,
    "12.6.0": CUDA_12_NCCL_WHEEL_DICT,
    "12.6.1": CUDA_12_NCCL_WHEEL_DICT,
    "12.6.2": CUDA_12_NCCL_WHEEL_DICT,
    "12.6.3": CUDA_12_NCCL_WHEEL_DICT,
    "12.8.0": CUDA_12_NCCL_WHEEL_DICT,
}

REDIST_VERSIONS_TO_BUILD_TEMPLATES = {
    "nvidia_driver": {
        "repo_name": "cuda_driver",
        "version_to_template": {
            "570": "//third_party/gpus/cuda/hermetic:cuda_driver.BUILD.tpl",
            "560": "//third_party/gpus/cuda/hermetic:cuda_driver.BUILD.tpl",
            "555": "//third_party/gpus/cuda/hermetic:cuda_driver.BUILD.tpl",
            "550": "//third_party/gpus/cuda/hermetic:cuda_driver.BUILD.tpl",
            "545": "//third_party/gpus/cuda/hermetic:cuda_driver.BUILD.tpl",
            "530": "//third_party/gpus/cuda/hermetic:cuda_driver.BUILD.tpl",
            "520": "//third_party/gpus/cuda/hermetic:cuda_driver.BUILD.tpl",
        },
    },
    "cuda_nccl": {
        "repo_name": "cuda_nccl",
        "version_to_template": {
            "2": "//third_party/nccl/hermetic:cuda_nccl.BUILD.tpl",
        },
    },
    "cudnn": {
        "repo_name": "cuda_cudnn",
        "version_to_template": {
            "9": "//third_party/gpus/cuda/hermetic:cuda_cudnn9.BUILD.tpl",
            "8": "//third_party/gpus/cuda/hermetic:cuda_cudnn.BUILD.tpl",
        },
    },
    "libcublas": {
        "repo_name": "cuda_cublas",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_cublas.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_cublas.BUILD.tpl",
        },
    },
    "cuda_cudart": {
        "repo_name": "cuda_cudart",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_cudart.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_cudart.BUILD.tpl",
        },
    },
    "libcufft": {
        "repo_name": "cuda_cufft",
        "version_to_template": {
            "11": "//third_party/gpus/cuda/hermetic:cuda_cufft.BUILD.tpl",
            "10": "//third_party/gpus/cuda/hermetic:cuda_cufft.BUILD.tpl",
        },
    },
    "cuda_cupti": {
        "repo_name": "cuda_cupti",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_cupti.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_cupti.BUILD.tpl",
        },
    },
    "libcurand": {
        "repo_name": "cuda_curand",
        "version_to_template": {
            "10": "//third_party/gpus/cuda/hermetic:cuda_curand.BUILD.tpl",
        },
    },
    "libcusolver": {
        "repo_name": "cuda_cusolver",
        "version_to_template": {
            "11": "//third_party/gpus/cuda/hermetic:cuda_cusolver.BUILD.tpl",
        },
    },
    "libcusparse": {
        "repo_name": "cuda_cusparse",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_cusparse.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_cusparse.BUILD.tpl",
        },
    },
    "libnvjitlink": {
        "repo_name": "cuda_nvjitlink",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvjitlink.BUILD.tpl",
        },
    },
    "cuda_nvrtc": {
        "repo_name": "cuda_nvrtc",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvrtc.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvrtc.BUILD.tpl",
        },
    },
    "cuda_cccl": {
        "repo_name": "cuda_cccl",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_cccl.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_cccl.BUILD.tpl",
        },
    },
    "cuda_nvcc": {
        "repo_name": "cuda_nvcc",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvcc.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvcc.BUILD.tpl",
        },
    },
    "cuda_nvml_dev": {
        "repo_name": "cuda_nvml",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvml.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvml.BUILD.tpl",
        },
    },
    "cuda_nvprune": {
        "repo_name": "cuda_nvprune",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvprune.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvprune.BUILD.tpl",
        },
    },
    "cuda_nvtx": {
        "repo_name": "cuda_nvtx",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvtx.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvtx.BUILD.tpl",
        },
    },
}
