"""TODO(jakeharmon): Write module docstring."""

# unused in TSL
def tf_additional_plugin_deps():
    return select({
        str(Label("@local_xla//xla/tsl:with_xla_support")): [
            str(Label("//tensorflow/compiler/jit")),
        ],
        "//conditions:default": [],
    })

def if_dynamic_kernels(extra_deps, otherwise = []):
    return select({
        str(Label("//tensorflow:dynamic_loaded_kernels")): extra_deps,
        "//conditions:default": otherwise,
    })
