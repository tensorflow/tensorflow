
exports_files(glob(["requirements*"]) + [
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "AUTHORS",
    "LICENSE"
])

exports_files(["LICENSE"])

# Config setting for AddressSanitizer (ASan)
config_setting(
    name = "asan",
    values = {"compilation_mode": "opt"},
)

copts = ["-fsanitize=address"]
linkopts = ["-fsanitize=address", "/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.a"]
