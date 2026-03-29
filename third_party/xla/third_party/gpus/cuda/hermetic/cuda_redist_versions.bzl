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
NVSHMEM_REDIST_PATH_PREFIX = "https://developer.download.nvidia.com/compute/nvshmem/redist/"
CUDNN_REDIST_PATH_PREFIX = "https://developer.download.nvidia.com/compute/cudnn/redist/"
MIRRORED_TAR_CUDA_REDIST_PATH_PREFIX = "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cuda/redist/"
MIRRORED_TAR_CUDNN_REDIST_PATH_PREFIX = "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/cudnn/redist/"
MIRRORED_TAR_NVSHMEM_REDIST_PATH_PREFIX = "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/nvshmem/redist/"

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
    "12.9.0": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.0.json",
        "4e4e17a12adcf8cac40b990e1618406cd7ad52da1817819166af28a9dfe21d4a",
    ],
    "12.9.1": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json",
        "8335301010b0023ee1ff61eb11e2600ca62002d76780de4089011ad77e0c7630",
    ],
    "13.0.0": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.0.json",
        "fe6a86b54450d03ae709123a52717870c49046d65d45303ce585c7aa8a83a217",
    ],
    "13.0.1": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.1.json",
        "9c494bc13b34e8fbcad083a6486d185b0906068b821722502edf9d0e3bd14096",
    ],
    "13.1.0": [
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.1.0.json",
        "55304d9d831bb095d9594aab276f96d2f0e30919f4cc1b3f6ca78cdb5f643e11",
    ],
    "13.1.1":[
        "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.1.1.json",
        "97cf605ccc4751825b1865f4af571c9b50dd29ffd13e9a38b296a9ecb1f0d422",
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
    "9.9.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.9.0.json",
        "614d3c5ceb02e1eb1508f0bc9231c3c03c113bb514b950a1108adb9fde801c77",
    ],
    "9.10.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.10.0.json",
        "d06b8df4d305dd7021838ffb2a26c2a861d522f2a129c6a372fad72ca009b1f1",
    ],
    "9.10.1": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.10.1.json",
        "2ac8d48d3ab4de1acdce65fa3e8ecfb14750d4e101b05fe3307d2f95f2740563",
    ],
    "9.10.2": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.10.2.json",
        "73a33a12bbb8eb12b105a515b5921db2e328b3ca679f92b6184c7f32fe94a8b0",
    ],
    "9.11.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.11.0.json",
        "7a16458ea21573e18d190df0c8d68ea1e8c82faf1bcfad4a39ceb600c26639cc",
    ],
    "9.11.1": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.11.1.json",
        "ace81583a37b8fe238324b73087e32f290099cbd6d012772b9f14ec4efac1f21",
    ],
    "9.12.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.12.0.json",
        "39bb68f0ca6abdbf9bab3ecb1cb18f458d635f72d72ede98a308216fd22efab3",
    ],
    "9.13.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.13.0.json",
        "55e3eb3ccb1ca543a7811312466f44841d630d3b2252f5763ad53509d2c09fbf",
    ],
    "9.13.1": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.13.1.json",
        "13d6a68bf4069a51fe653a769c41f9b0e3003e7f93ccf0f6cb89f642d10e2ccf",
    ],
    "9.14.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.14.0.json",
        "fe58e8e9559ef5c61ab7a9954472d16acdcbad3b099004296ae410d25982830d",
    ],
    "9.15.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.15.0.json",
        "2396ed88435a0f6b400db53ac229f49aa2425282994a186e867ea367c20fd352",
    ],
    "9.15.1": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.15.1.json",
        "8c9897222c644528a25e0bd4d04d5ee9b9cb57995307c176d4dce28c25e415ef",
    ],
    "9.16.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.16.0.json",
        "c95167877ac0ded30a29accc9d337a5e60cd70d1a01a3492de56624b39eab868",
    ],
    "9.17.0": [
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.17.0.json",
        "5c2fe21bd5626f1078caf030c569d894df44a844bfe3b0475ae5f55a4b64c395",
    ],
    "9.17.1":[
        "https://developer.download.nvidia.com/compute/cudnn/redist/redistrib_9.17.1.json",
        "f7583aa8652b5434ecd85bdc735ee2b5f1171e6841bf626b70e4937d4b2a2c88",
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

NVSHMEM_REDIST_JSON_DICT = {
    "3.2.5": [
        "https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.2.5.json",
        "6945425d3bfd24de23c045996f93ec720c010379bfd6f0860ac5f2716659442d",
    ],
    "3.3.9": [
        "https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.3.9.json",
        "fecaaab763c23d53f747c299491b4f4e32e0fc2e059b676772b886ada2ba711e",
    ],
    "3.3.20": [
        "https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.3.20.json",
        "0da2b7f4553e4debef4dbbe899fe7c3bb6324a7cba181e3da6666479c7d4038e",
    ],
    "3.3.24": [
        "https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.3.24.json",
        "60ef5424c1632bb1fa1fb41aea9d75b1777f62faeebb1eeaa818ed92068403b8",
    ],
    "3.4.5": [
        "https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.4.5.json",
        "a656614a6ec638d85922bc816e5e26063308c3905273a72a863cf0f24e188f38",
    ],
    "3.5.19": [
        "https://developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.5.19.json",
        "6dced4193eb728542504b346cfb768da6e3de2abca0cded95fda3a69729994d2",
    ],
}

MIRRORED_TARS_NVSHMEM_REDIST_JSON_DICT = {
    "3.2.5": [
        "https://storage.googleapis.com/mirror.tensorflow.org/developer.download.nvidia.com/compute/nvshmem/redist/redistrib_3.2.5_tar.json",
        "641f7ca7048e4acfb466ce8be722f4828b2fa6b8671c28f6e8c230344484fd1c",
    ],
}

CUDA_13_NCCL_WHEEL_DICT = {
    "x86_64-unknown-linux-gnu": {
        "version": "2.29.3",
        "url": "https://files.pythonhosted.org/packages/7b/70/aae7806eeaed043b3e212da435880ad067b5f14052986a6b4c0a4c62f68a/nvidia_nccl_cu13-2.29.3-py3-none-manylinux_2_18_x86_64.whl",
        "sha256": "2a321629f49490e4e0122ecb578a4b4a6f89e72740dd988e04dfa4758fab7fc3",
    },
    "aarch64-unknown-linux-gnu": {
        "version": "2.29.3",
        "url": "https://files.pythonhosted.org/packages/27/59/ff243ebe6fa1767a9135719829347f609a90607cfbba9637ba3e9b3e36ce/nvidia_nccl_cu13-2.29.3-py3-none-manylinux_2_18_aarch64.whl",
        "sha256": "eab9f5c565ab3326906f1d1b5be5773a174c2a1b47002faed76f9e957392f713",
    },
}

CUDA_12_NCCL_WHEEL_DICT = {
    "x86_64-unknown-linux-gnu": {
        "version": "2.26.5",
        "url": "https://files.pythonhosted.org/packages/48/fb/ec4ac065d9b0d56f72eaf1d9b0df601e33da28197b32ca351dc05b342611/nvidia_nccl_cu12-2.26.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "sha256": "ea5ed3e053c735f16809bee7111deac62ac35b10128a8c102960a0462ce16cbe",
    },
    "aarch64-unknown-linux-gnu": {
        "version": "2.26.5",
        "url": "https://files.pythonhosted.org/packages/55/66/ed9d28946ead0fe1322df2f4fc6ea042340c0fe73b79a1419dc1fdbdd211/nvidia_nccl_cu12-2.26.5-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl",
        "sha256": "adb1bf4adcc5a47f597738a0700da6aef61f8ea4251b375540ae138c7d239588",
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
    "12.9.0": CUDA_12_NCCL_WHEEL_DICT,
    "12.9.1": CUDA_12_NCCL_WHEEL_DICT,
    "13.0.0": CUDA_13_NCCL_WHEEL_DICT,
    "13.0.1": CUDA_13_NCCL_WHEEL_DICT,
    "13.1.0": CUDA_13_NCCL_WHEEL_DICT,
    "13.1.1": CUDA_13_NCCL_WHEEL_DICT,
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
        "13.1": "9.1",
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
    "cuda_nvdisasm": {
        "repo_name": "cuda_nvdisasm",
        "version_to_template": {
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvdisasm.BUILD",
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
            "13": "//third_party/gpus/cuda/hermetic:cuda_nvprune.BUILD",
            "12": "//third_party/gpus/cuda/hermetic:cuda_nvprune.BUILD",
            "11": "//third_party/gpus/cuda/hermetic:cuda_nvprune.BUILD",
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

NVSHMEM_REDIST_VERSIONS_TO_BUILD_TEMPLATES = {
    "libnvshmem": {
        "repo_name": "nvidia_nvshmem",
        "version_to_template": {
            "3": "//third_party/nvshmem/hermetic:nvidia_nvshmem.BUILD.tpl",
        },
    },
}
