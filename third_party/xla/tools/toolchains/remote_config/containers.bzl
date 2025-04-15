"""Docker images used with remote config and RBE."""

# SHA 256 values for each image.
container_digests = {
    # TF now uses only this container
    "cuda11.2-cudnn8.1-ubuntu20.04-manylinux2014-multipython": "sha256:48612bd85709cd014711d0b0f87e0806f3567d06d2e81c6e860516b87498b821",
    # JAX manylinux2014 configs.
    "cuda11.8-cudnn8.6-ubuntu20.04-manylinux2014-multipython": "sha256:45619e91f14faabddd79fe0cb1526df4c4ad92fc2e6ebdc725ea4419225429c3",
    "cuda12.1-cudnn8.9-ubuntu20.04-manylinux2014-multipython": "sha256:8c266e5b0acd203aed5e8871b63f68a39d8d23f6d882e619797e58b973f7fe63",
    "cuda12.1-cudnn9.1-ubuntu20.04-manylinux2014-multipython": "sha256:7c14249d42f6fb4254a8085cea0683951979e9a325bda14522bf910802226394",
    "cuda12.2-cudnn8.9-ubuntu20.04-manylinux2014-multipython": "sha256:9fefda035b4a12b24cd5bae56c7dbb9527a5fd06a41ced0a22ac86fe5ed26428",
    "cuda12.2-cudnn9.1-ubuntu20.04-manylinux2014-multipython": "sha256:0c78f3428cde36f041b758fc2f01d23d2f0dd72dec248f78667fb0c9d1f74cef",
    "cuda12.3-cudnn8.9-ubuntu20.04-manylinux2014-multipython": "sha256:6f9524a2ed7f75255dc4be3a0c5e3bda581385a1c13e2fa890bc17fa62da95b2",
    "cuda12.3-cudnn8.9-ubuntu22.04-manylinux2014-multipython": "sha256:97b219abb22994cf0530771d536f26fe301bacd328f0485c38af3847c2ee6b14",
    "cuda12.3-cudnn9.1-ubuntu20.04-manylinux2014-multipython": "sha256:89aa860228092d78dc336545a5ad32e342d656a8258186327c989bd301bb3eb8",
    # ROCM, probably not all of them still in use
    "rocm-ubuntu18.04-manylinux2010-multipython": "sha256:6e953a09b145df338bcb03e9e36f99b291140c29b72d0a048fb6c5905ccad5eb",
    "rocm-ubuntu20.04-manylinux2014-multipython": "sha256:906faec7765fe5dd067f2b092b5d5f220c1fedde725fb42c83d031b4d6f32204",
    # Unused?
    "ubuntu16.04": "sha256:b90dcf2f35f3354909f4491bdf019c110b4b4d95ef0395ebf178bc5d523a4208",
    "centos6": "sha256:d09c12fb26fbbe8398b4973260c75172eb67d509dae9d6f4ad54279b7d6b0494",
    "ubuntu16.04-manylinux2010": "sha256:5b6a55a33046843a3c3b3b3d13b028cc6e61d8af5275998e95fa539cafe72c51",
    "cuda10.0-cudnn7-ubuntu14.04": "sha256:d433e1221f802dac393bc8652fabcc63aa46896cd920bb888ae0e2002fe6b756",
    "cuda10.0-cudnn7-centos7": "sha256:a453b7147a60928a8345689eae48916a746b3578b5e831bfa151f0529d469c88",
    "cuda10.0-cudnn7-centos6": "sha256:a1909ba09c703340ee0074ce63dd94fe8fea48035a25264677907a609e2375e0",
    "cuda10.1-cudnn7-centos6": "sha256:454b899657e87893ee5e68dc0f87df59b6a0a7418ae09cafcc3dd65ac71feca9",
    "cuda10.0-cudnn7-ubuntu16.04-manylinux2010": "sha256:5812d9d0ef0a3276fc5faaf4cd01f3d6e03d635893a6e2d2e04f6f01d626c432",
    "cuda10.1-cudnn7-ubuntu16.04-manylinux2010": "sha256:5e6d21c8ef226316eb6df5e2e6015244c16a8e5d936b52a09820442d2f8a919f",
    "cuda10.1-cudnn7-ubuntu16.04-manylinux2010-multipython": "sha256:3f890a951c81a201d60d0161a56ce628a90323be0c7f795550caa37f6f41a85c",
    "cuda10.1-cudnn7-ubuntu18.04-manylinux2010-multipython": "sha256:71e946b59a63e80d10178f9e2869af3b31583737b779edc27688c99f4998dd05",
    "cuda10.2-cudnn7-ubuntu18.04-manylinux2010-multipython": "sha256:9bb24f6f98e71b424dd885ec9fe39770831376840fa2231057807c89f814c803",
    "cuda11.0-cudnn8-ubuntu18.04-manylinux2010-multipython": "sha256:3e5c991f67e2cca610cb9f6b39927e3757ba1e7f2424d18cef8b871bfa4d75b3",
    "windows-1803": "sha256:f109576c7c0c8a1783ff22b666e8923b52dbbe7933f69a1c7a7275202c304a12",
}

containers = {
    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.ubuntu16.04-manylinux2010.
    "ubuntu16.04-manylinux2010": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-ubuntu16.04-manylinux2010",
        "digest": container_digests["ubuntu16.04-manylinux2010"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.0-cudnn7-ubuntu16.04-manylinux2010.
    "cuda10.0-cudnn7-ubuntu16.04-manylinux2010": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010",
        "digest": container_digests["cuda10.0-cudnn7-ubuntu16.04-manylinux2010"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.1-cudnn7-ubuntu16.04-manylinux2010.
    "cuda10.1-cudnn7-ubuntu16.04-manylinux2010": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda10.1-cudnn7-ubuntu16.04-manylinux2010",
        "digest": container_digests["cuda10.1-cudnn7-ubuntu16.04-manylinux2010"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.1-cudnn7-ubuntu16.04-manylinux2010-multipython.
    "cuda10.1-cudnn7-ubuntu16.04-manylinux2010-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda10.1-cudnn7-ubuntu16.04-manylinux2010-multipython",
        "digest": container_digests["cuda10.1-cudnn7-ubuntu16.04-manylinux2010-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.1-cudnn7-ubuntu18.04-manylinux2010-multipython.
    "cuda10.1-cudnn7-ubuntu18.04-manylinux2010-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda10.1-cudnn7-ubuntu18.04-manylinux2010-multipython",
        "digest": container_digests["cuda10.1-cudnn7-ubuntu18.04-manylinux2010-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.2-cudnn7-ubuntu18.04-manylinux2010-multipython.
    "cuda10.2-cudnn7-ubuntu18.04-manylinux2010-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda10.2-cudnn7-ubuntu18.04-manylinux2010-multipython",
        "digest": container_digests["cuda10.2-cudnn7-ubuntu18.04-manylinux2010-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda11.0-cudnn8-ubuntu18.04-manylinux2010-multipython.
    "cuda11.0-cudnn8-ubuntu18.04-manylinux2010-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda11.0-cudnn8-ubuntu18.04-manylinux2010-multipython",
        "digest": container_digests["cuda11.0-cudnn8-ubuntu18.04-manylinux2010-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda11.2-cudnn8.1-ubuntu20.04-manylinux2014-multipython.
    "cuda11.2-cudnn8.1-ubuntu20.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda11.2-cudnn8.1-ubuntu20.04-manylinux2014-multipython",
        "digest": container_digests["cuda11.2-cudnn8.1-ubuntu20.04-manylinux2014-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda11.8-cudnn8.6-ubuntu20.04-manylinux2014-multipython.
    "cuda11.8-cudnn8.6-ubuntu20.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda11.8-cudnn8.6-ubuntu20.04-manylinux2014-multipython",
        "digest": container_digests["cuda11.8-cudnn8.6-ubuntu20.04-manylinux2014-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda12.1-cudnn8.9-ubuntu20.04-manylinux2014-multipython.
    "cuda12.1-cudnn8.9-ubuntu20.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda12.1-cudnn8.9-ubuntu20.04-manylinux2014-multipython",
        "digest": container_digests["cuda12.1-cudnn8.9-ubuntu20.04-manylinux2014-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda12.1-cudnn9.1-ubuntu20.04-manylinux2014-multipython.
    "cuda12.1-cudnn9.1-ubuntu20.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda12.1-cudnn9.1-ubuntu20.04-manylinux2014-multipython",
        "digest": container_digests["cuda12.1-cudnn9.1-ubuntu20.04-manylinux2014-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda12.2-cudnn8.9-ubuntu20.04-manylinux2014-multipython.
    "cuda12.2-cudnn8.9-ubuntu20.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda12.2-cudnn8.9-ubuntu20.04-manylinux2014-multipython",
        "digest": container_digests["cuda12.2-cudnn8.9-ubuntu20.04-manylinux2014-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda12.2-cudnn9.1-ubuntu20.04-manylinux2014-multipython.
    "cuda12.2-cudnn9.1-ubuntu20.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda12.2-cudnn9.1-ubuntu20.04-manylinux2014-multipython",
        "digest": container_digests["cuda12.2-cudnn9.1-ubuntu20.04-manylinux2014-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda12.3-cudnn8.9-ubuntu20.04-manylinux2014-multipython.
    "cuda12.3-cudnn8.9-ubuntu20.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda12.3-cudnn8.9-ubuntu20.04-manylinux2014-multipython",
        "digest": container_digests["cuda12.3-cudnn8.9-ubuntu20.04-manylinux2014-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda12.3-cudnn9.1-ubuntu20.04-manylinux2014-multipython.
    "cuda12.3-cudnn9.1-ubuntu20.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda12.3-cudnn9.1-ubuntu20.04-manylinux2014-multipython",
        "digest": container_digests["cuda12.3-cudnn9.1-ubuntu20.04-manylinux2014-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda12.3-cudnn8.9-ubuntu20.04-manylinux2014-multipython.
    "cuda12.3-cudnn8.9-ubuntu22.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda12.3-cudnn8.9-ubuntu22.04-manylinux2014-multipython",
        "digest": container_digests["cuda12.3-cudnn8.9-ubuntu22.04-manylinux2014-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.rocm-ubuntu18.04-manylinux2010-multipython.
    "rocm-ubuntu18.04-manylinux2010-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-rocm-ubuntu18.04-manylinux2010-multipython",
        "digest": container_digests["rocm-ubuntu18.04-manylinux2010-multipython"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.rocm-ubuntu20.04-manylinux2014-multipython.
    "rocm-ubuntu20.04-manylinux2014-multipython": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-rocm-ubuntu20.04-manylinux2014-multipython",
        "digest": container_digests["rocm-ubuntu20.04-manylinux2014-multipython"],
    },

    # Built by mihaimaruseac@ from a private Dockerfile.
    "windows-1803": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/tf-win-rbe",
        "digest": container_digests["windows-1803"],
    },
}
