"""Build configurations for nGraph."""

def clean_dep(dep):
    return str(Label(dep))

def if_ngraph(if_true, if_false = []):
    """select()'ing on whether we're building with nGraph support."""
    return select({
        clean_dep("//tensorflow:with_ngraph_support"): if_true,
        "//conditions:default": if_false,
    })
