def clean_dep(dep):
  return str(Label(dep))

def if_ngraph(a):
    """Shorthand for select()'ing on whether we're building with nGraph support.

    Returns a select statement which evaluates to if_true if we're building
    with nGraph.  Otherwise, the select statement evaluates to default.

    """
    ret_val = select({
        clean_dep("//tensorflow:with_ngraph_support"): a,
        "//conditions:default": []
    })

    return ret_val
