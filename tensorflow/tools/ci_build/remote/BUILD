# For generating docker images for TensorFlow remote build
licenses(["notice"])  # Apache 2.0

genrule(
    name = "remote-docker-build-cpu",
    srcs = [
        "@llvm-project//llvm:docker",  # llvm's docker file group
    ],
    outs = ["remote-docker-build-cpu.txt"],
    cmd = " ./$(location remote_docker_build.sh) -c -f $$(realpath $<) > $@",
    local = 1,
    tags = ["manual"],
    tools = ["remote_docker_build.sh"],
)

genrule(
    name = "remote-docker-build-gpu",
    srcs = [
        "@llvm-project//llvm:docker",  # llvm's docker file group
    ],
    outs = ["remote-docker-build-gpu.txt"],
    cmd = " ./$(location remote_docker_build.sh) -g -f $$(realpath $<) > $@",
    local = 1,
    tags = ["manual"],
    tools = ["remote_docker_build.sh"],
)
