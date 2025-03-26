load(
    "@local_xla//third_party/gpus/rocm:rocm_redist_ubuntu_20_04.bzl",
    "rocm_redist_ubuntu_20_04",
)
load(
    "@local_xla//third_party/gpus/rocm:rocm_redist_ubuntu_22_04.bzl",
    "rocm_redist_ubuntu_22_04",
)
load(
    "@local_xla//third_party/gpus/rocm:rocm_redist_ubuntu_24_04.bzl",
    "rocm_redist_ubuntu_24_04",
)

rocm_redist = {
    "ubuntu_20.04": rocm_redist_ubuntu_20_04,
    "ubuntu_22.04": rocm_redist_ubuntu_22_04,
    "ubuntu_24.04": rocm_redist_ubuntu_24_04,
}
