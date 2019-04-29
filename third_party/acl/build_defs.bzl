"""Build configurations for Arm Compute Library."""

def clean_dep(dep):
    return str(Label(dep))

def if_acl(if_true, if_false = []):
    """select()'ing on whether we're building with Arm Compute Library support."""
    return select({
        clean_dep("//tensorflow:with_acl_support"): if_true,
        "//conditions:default": if_false,
    })
