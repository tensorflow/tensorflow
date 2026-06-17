"""Build macros for SHLO reference library."""

def shlo_ref_logging_linkopts():
    """Defines linker flags to enable logging"""
    return select({
        "//tensorflow:android": ["-llog"],
        "//conditions:default": [],
    })

def shlo_ref_linkopts():
    """Defines linker flags for linking SHLO binary"""
    return shlo_ref_logging_linkopts()
