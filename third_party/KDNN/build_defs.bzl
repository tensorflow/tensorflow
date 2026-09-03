def if_enable_kdnn(if_true, if_false = []):
    """Selects KDNN sources only for an enabled Linux ARM64 build.

    This is only effective when built with KDNN.

    Args:
        if_true: expression to evaluate if building with KDNN and KDNN is enabled
        if_false: expression to evaluate if building without KDNN or KDNN is not enabled.

    Returns:
        A select evaluating to either if_true or if_false as appropriate.
    """
    return select({
        Label("//third_party/KDNN:enable_kdnn_arm64"): if_true,  # copybara:comment_replace Label("//tensorflow/third_party/KDNN:enable_kdnn_arm64"): if_true,
        "//conditions:default": if_false,
    })

def kdnn_deps():
    """Selects the KDNN binary dependency for an enabled Linux ARM64 build.
    """
    return select({
        Label("//third_party/KDNN:enable_kdnn_arm64"): ["//third_party/KDNN:kdnn_adapter"],  # copybara:comment_replace Label("//tensorflow/third_party/KDNN:enable_kdnn_arm64"): ["//tensorflow/third_party/KDNN:kdnn_adapter"],
        "//conditions:default": [],
    })
