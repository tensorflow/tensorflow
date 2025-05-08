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
MIRRORED_TAR_CUDA_REDIST_PATH_PREFIX = "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/"
MIRRORED_TAR_CUDNN_REDIST_PATH_PREFIX = "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/"

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
    "12.8.1": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json",
        "249e28a83008d711d5f72880541c8be6253f6d61608461de4fcb715554a6cf17",
    ],
}

MIRRORED_TARS_CUDA_REDIST_JSON_DICT = {
    "11.8": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_11.8.0_tar.json",
        "a325b9dfba60c88f71b681e2f58b790b09afd9cb476fe620fabcb50be6f30add",
    ],
    "12.1.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.1.1_tar.json",
        "f4c6679ebf3dedbeff329d5ee0c8bfec3f32c4976f5d9cdc238ac9faa0109502",
    ],
    "12.2.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.2.0_tar.json",
        "69db566d620fbc5ecb8ee367d60b7e1d23f0ee64a11eca4cad97b037d9850819",
    ],
    "12.3.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.3.1_tar.json",
        "d2d6331166117ca6889899245071903b1b01127713e934f8a91850f52862644c",
    ],
    "12.3.2": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.3.2_tar.json",
        "796b019c6d707a656544ef007ad180d2e57dbf5c018683464166e2c512c1ec68",
    ],
    "12.4.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.4.0_tar.json",
        "3b5066efdfe8072997ca8f3bbb9bf8c4bb869f25461d22887247be4d16101ba7",
    ],
    "12.4.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.4.1_tar.json",
        "ff6cf5d43fd65e65bf1380f295adcc77b1c7598feff5b782912885ee5ac242e8",
    ],
    "12.5.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.5.0_tar.json",
        "32a8d4ce1b31d15f02ac6a9cc7c5b060bd329a2a754906b1485752d9c9da59b5",
    ],
    "12.5.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.5.1_tar.json",
        "b1d50589900b5b50d01d1f741448802020835b5135fcbb969c6bf7b831372a7f",
    ],
    "12.6.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.0_tar.json",
        "a5de3ae3f01ab25dec442fa133ca1d3eb0001fab6de14490b2f314b03dd3c0e4",
    ],
    "12.6.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.1_tar.json",
        "8da05eb613d2d71b4814fde25de0a418b1dc04c0a409209dfce82b5ca8b15dec",
    ],
    "12.6.2": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.2_tar.json",
        "cb18f8464212e71c364f6d8c9bf6b70c0908e2e069d75c90fc65e0b07981bb53",
    ],
    "12.6.3": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.6.3_tar.json",
        "e1b558de79fe2da21cac80c498e4175a48087677627eacb915dd78f42833b5b3",
    ],
    "12.8.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.0_tar.json",
        "c9790b289d654844d9dd2ec07f30383220dac1320f7d7d686722e946f9a55e44",
    ],
    "12.8.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1_tar.json",
        "30a1b8ace0d38237f4ab3ab28d89dbc77ae2c4ebabe27ba08b3c0961cc6cc7fa",
    ],
}

CUDNN_REDIST_JSON_DICT = {
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
    "9.8.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.8.0.json",
        "a1599fa1f8dcb81235157be5de5ab7d3936e75dfc4e1e442d07970afad3c4843",
    ],
}

MIRRORED_TARS_CUDNN_REDIST_JSON_DICT = {
    "8.9.4.25": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_8.9.4.25_tar.json",
        "cf2642a1db2b564065232277f061e89f1b20435f88164fa783855ac69f33d3c2",
    ],
    "8.9.6": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_8.9.6_tar.json",
        "dab3ead7f79bf0378e2e9037a9f6a87f249c581aa75d1e2f352ffa3df56d5356",
    ],
    "8.9.7.29": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_8.9.7.29_tar.json",
        "7e305dc19b8a273645078bb3a37faaa54256a59ac9137934979983d9ce481717",
    ],
    "9.1.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.1.1_tar.json",
        "6960bc9e472b21c4ffae0a75309f41f48eb3d943a553ad70273927fb170fa99f",
    ],
    "9.2.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.2.0_tar.json",
        "35469a1494c8f95d81774fd7750c6cd2def3919e83b0fa8e0285edd42bcead20",
    ],
    "9.2.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.2.1_tar.json",
        "de77cb78dd620f1c1f8d1a07e167ba6d6cfa1be5769172a09c5315a1463811c1",
    ],
    "9.3.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.3.0_tar.json",
        "50aadf1e10b0988bb74497331953f1afbd9c596c27c6014f4d3f370cec2713aa",
    ],
    "9.4.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.4.0_tar.json",
        "114a6ad4152ea014cc07fec1fa63a029c6eec6a5dc4463c8dc83ad6d5f809795",
    ],
    "9.5.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.5.0_tar.json",
        "f224f5a875129eeb5b3c7e18d8a5f2e7bb5498f0e3095a8ae5fb863ebc450c52",
    ],
    "9.5.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.5.1_tar.json",
        "28ce996b3f4171f6a3873152470e14753788cddd089261513c18c773fe2a2b73",
    ],
    "9.6.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.6.0_tar.json",
        "084cc250593cfbc962f7942a4871aa13a179ce5beb1aea236b74080cc23e29f0",
    ],
    "9.7.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.7.0_tar.json",
        "402906b09b7b2624e6a5c6937a41cc3330d6e588f2f211504ad3fb8a5823fa01",
    ],
    "9.7.1": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.7.1_tar.json",
        "2eaa4594c1ab188c939026d90245d3ffca2a83d41aba1be903f644cc1215c23d",
    ],
    "9.8.0": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.8.0_tar.json",
        "030378782b94597855cdf7d3068968f88460cd9c4ce9d73c77cfad64dfdea070",
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
    "12.8.1": CUDA_12_NCCL_WHEEL_DICT,
}

# Ensures PTX version compatibility w/ Clang & ptxas in cuda_configure.bzl
PTX_VERSION_DICT = {
    # To find, invoke `llc -march=nvptx64 -mcpu=help 2>&1 | grep ptx | sort -V | tail -n 1`
    "clang": {
        "14": "7.5",
        "15": "7.5",
        "16": "7.8",
        "17": "8.1",
        "18": "8.3",
        "19": "8.5",
        "20": "8.7",
    },
    # To find, look at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes
    "cuda": {
        "11.8": "7.8",
        "12.1": "8.1",
        "12.2": "8.2",
        "12.3": "8.3",
        "12.4": "8.4",
        "12.5": "8.5",
        "12.6": "8.5",
        "12.8": "8.7",
        "12.9": "8.8",
    },
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
            "13": "//third_party/gpus/cuda/hermetic:cuda_cudart.BUILD.tpl",
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
            "13": "//third_party/gpus/cuda/hermetic:cuda_nvjitlink.BUILD.tpl",
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvjitlink.BUILD.tpl",
        },
    },
    "cuda_nvrtc": {
        "repo_name": "cuda_nvrtc",
        "version_to_template": {
            "13": "//third_party/gpus/cuda/hermetic:cuda_nvrtc.BUILD.tpl",
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvrtc.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvrtc.BUILD.tpl",
        },
    },
    "cuda_cccl": {
        "repo_name": "cuda_cccl",
        "version_to_template": {
            "13": "//third_party/gpus/cuda/hermetic:cuda_cccl.BUILD.tpl",
            "12": "//third_party/gpus/cuda/hermetic:cuda_cccl.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_cccl.BUILD.tpl",
        },
    },
    "cuda_nvcc": {
        "repo_name": "cuda_nvcc",
        "version_to_template": {
            "13": "//third_party/gpus/cuda/hermetic:cuda_nvcc.BUILD.tpl",
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvcc.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvcc.BUILD.tpl",
        },
    },
    "cuda_nvml_dev": {
        "repo_name": "cuda_nvml",
        "version_to_template": {
            "13": "//third_party/gpus/cuda/hermetic:cuda_nvml.BUILD.tpl",
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvml.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvml.BUILD.tpl",
        },
    },
    "cuda_nvprune": {
        "repo_name": "cuda_nvprune",
        "version_to_template": {
            "13": "//third_party/gpus/cuda/hermetic:cuda_nvprune.BUILD.tpl",
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvprune.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvprune.BUILD.tpl",
        },
    },
    "cuda_nvtx": {
        "repo_name": "cuda_nvtx",
        "version_to_template": {
            "13": "//third_party/gpus/cuda/hermetic:cuda_nvtx.BUILD.tpl",
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvtx.BUILD.tpl",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvtx.BUILD.tpl",
        },
    },
}
