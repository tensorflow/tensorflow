"""RISC-V Python C/C++ toolchain configuration for TensorFlow."""

# The key issue was that rules_python expects the py_cc_toolchain to provide
# a 'headers' field that itself provides CcInfo. The original implementation
# was passing the label directly, but it needs to be structured properly.

def _riscv64_py_cc_toolchain_impl(ctx):
    """Implementation of the RISC-V Python C/C++ toolchain rule.
    
    This toolchain provides the Python headers needed for building
    native Python extensions on RISC-V architecture.
    
    Args:
        ctx: The rule context
        
    Returns:
        A platform_common.ToolchainInfo provider with the correct structure
    """
    
    # Get the python_headers target which should be a cc_library
    python_headers_target = ctx.attr.python_headers[0]
    
    # Create a struct that has a 'providers_map' attribute
    # This is what rules_python expects when it accesses toolchain.headers.providers_map.values()
    headers_struct = struct(
        providers_map = {
            "headers": python_headers_target,
        },
    )
    
    return [
        platform_common.ToolchainInfo(
            headers = headers_struct,
            python_version = ctx.attr.python_version,
        ),
    ]

riscv64_py_cc_toolchain_rule = rule(
    implementation = _riscv64_py_cc_toolchain_impl,
    attrs = {
        "python_version": attr.string(
            default = "3.11",
            doc = "The Python version this toolchain is for",
        ),
        "python_headers": attr.label_list(
            mandatory = True,
            allow_files = False,
            doc = "The cc_library target providing Python headers",
        ),
    },
    doc = """
    Defines a Python C/C++ toolchain for RISC-V architecture.
    
    This toolchain is required for building TensorFlow with native
    extensions on RISC-V platforms. It provides the Python headers
    needed for compilation.
    """,
)
