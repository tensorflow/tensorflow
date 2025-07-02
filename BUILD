load("@rules_cc//cc:defs.bzl", "cc_toolchain_suite")

exports_files(glob(["requirements*"]) + [
    "configure",
    "configure.py",
    "ACKNOWLEDGEMENTS",
    "AUTHORS",
    "LICENSE",
])

cc_toolchain_suite(
  name = "crosstool",
  toolchains = {
    "k8": "@llvm_toolchain//:cc-toolchain-x86_64-linux",
  },
)
