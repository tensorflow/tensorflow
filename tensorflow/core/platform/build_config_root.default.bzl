"""TODO(jakeharmon): Write module docstring."""

load("@local_tsl//third_party/py/rules_pywrap:pywrap.bzl", "use_pywrap_rules")

# unused in TSL
def tf_additional_plugin_deps():
    return select({
        str(Label("@local_xla//xla/tsl:with_xla_support")): [
            str(Label("//tensorflow/compiler/jit")),
        ],
        "//conditions:default": [],
    })

def if_dynamic_kernels(extra_deps, otherwise = []):
    # TODO(b/356020232): remove after migration is done
    if use_pywrap_rules():
        return otherwise

    return select({
        str(Label("//tensorflow:dynamic_loaded_kernels")): extra_deps,
        "//conditions:default": otherwise,
    })
