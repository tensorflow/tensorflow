rocm_redist = {
    "rocm_7.10.0_gfx90X": struct(
        url = "https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx90X-dcgpu-7.10.0a20251106.tar.gz",
        sha256 = "a9270cac210e02f60a7f180e6a4d2264436cdcce61167440e6e16effb729a8ea",
        rocm_device_lib_path = "llvm/amdgcn",
        required_softlinks = [
            struct(src = "llvm/amdgcn", dest = "amdgcn"),
        ],
    ),
    "rocm_7.10.0_gfx94X": struct(
        url = "https://therock-nightly-tarball.s3.amazonaws.com/therock-dist-linux-gfx94X-dcgpu-7.10.0a20251107.tar.gz",
        sha256 = "486dbf647bcf9b78f21d7477f43addc7b2075b1a322a119045db9cdc5eb98380",
        required_softlinks = [
            struct(src = "llvm/amdgcn", dest = "amdgcn"),
        ],
    ),
}
