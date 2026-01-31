"""Tensorflow JIT package_group definitions"""

def legacy_jit_users_package_group(name):
    """Defines visibility group for //third_party/tensorflow/compiler/jit.

    Args:
      name: package group name
    """

    native.package_group(
        name = name,
        packages = ["//..."],
    )
