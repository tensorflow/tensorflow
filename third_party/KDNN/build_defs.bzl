def if_enable_kdnn(if_true, if_false = []):
    """Shorthand to select() if we are building with KDNN and KDNN is enabled.

    This is only effective when built with KDNN.

    Args:
        if_true: expression to evaluate if building with KDNN and KDNN is enabled
        if_false: expression to evaluate if building without KDNN or KDNN is not enabled.

    Returns:
        A select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        "@org_tensorflow//third_party/KDNN:enable_kdnn_arm64": if_true,
        "//conditions:default": if_false,
    })

def kdnn_deps():
    """Shorthand for select() to pull in the correct set of KDNN library deps.
    """
    return select({
        "@org_tensorflow//third_party/KDNN:enable_kdnn_arm64": ["//third_party/KDNN:kdnn_adapter"],
        "//conditions:default": [],
    })
