def if_enable_acl(if_true, if_false = []):
    return select({
        "@local_xla//third_party/compute_library:build_with_acl": if_true,
        "//conditions:default": if_false,
    })

def acl_deps():
    """Returns the correct set of ACL library dependencies.

      Shorthand for select() to pull in the correct set of ACL library deps
      for aarch64 platform

    Returns:
      a select evaluating to a list of library dependencies, suitable for
      inclusion in the deps attribute of rules.
    """
    return select({
        "@local_xla//third_party/compute_library:build_with_acl": ["@compute_library//:arm_compute"],
        "//conditions:default": [],
    })
