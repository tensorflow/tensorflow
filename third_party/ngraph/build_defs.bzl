def clean_dep(dep):
  return str(Label(dep))

def if_ngraph(if_true, if_false = []):
    """Shorthand for select()'ing on whether we're building with nGraph support.

    Returns a select statement which evaluates to if_true if we're building
    with nGraph.  Otherwise, the select statement evaluates to default.

    """
    return select({
        clean_dep("//tensorflow:with_ngraph_support"): if_true,
        "//conditions:default": if_false
    })
