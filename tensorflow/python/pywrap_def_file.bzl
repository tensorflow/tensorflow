# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Custom Starlark rule to generate the Windows DEF file by harvesting COFF symbols
from transitive C++ static libraries."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _pywrap_def_file_impl(ctx):
    if not ctx.executable.win_tool:
        ctx.actions.run_shell(
            inputs = [ctx.file.static_export_table],
            outputs = [ctx.outputs.out],
            command = "cp {} {}".format(ctx.file.static_export_table.path, ctx.outputs.out.path),
            mnemonic = "PywrapDefFileCopy",
        )
        return

    # We are on Windows. Collect all transitive library files from deps.
    transitive_lib_files = []
    for dep in ctx.attr.deps:
        if CcInfo in dep:
            for linker_input in dep[CcInfo].linking_context.linker_inputs.to_list():
                for lib in linker_input.libraries:
                    if lib.static_library:
                        transitive_lib_files.append(lib.static_library)
                    if lib.pic_static_library:
                        transitive_lib_files.append(lib.pic_static_library)
        else:
            # Fallback for non-CcInfo targets (e.g., pybind_extension rules in rules_pywrap).
            # Collect their direct output files (matching standard genrule srcs behavior).
            for f in dep.files.to_list():
                if f.path.endswith((".lib", ".obj", ".a", ".o")):
                    transitive_lib_files.append(f)

    params_file = ctx.actions.declare_file(ctx.label.name + ".obj_files.params")
    ctx.actions.write(
        output = params_file,
        content = "\n".join([f.path for f in transitive_lib_files]),
    )

    inputs = depset(
        direct = [ctx.file.unmangled_symbols_file, ctx.file.static_export_table, params_file] + transitive_lib_files,
    )

    cmd = "{} --workspace_root=$(pwd) --stage=mangling --unmangled_symbols_file={} --static_export_table={} --output_def_file={} --obj_files @{}".format(
        ctx.executable.win_tool.path,
        ctx.file.unmangled_symbols_file.path,
        ctx.file.static_export_table.path,
        ctx.outputs.out.path,
        params_file.path,
    )
    if ctx.attr.exit_1:
        cmd += " && exit 1"

    ctx.actions.run_shell(
        inputs = inputs,
        outputs = [ctx.outputs.out],
        tools = [ctx.attr.win_tool[DefaultInfo].files_to_run],
        command = cmd,
        mnemonic = "PywrapDefFileGen",
    )

pywrap_def_file = rule(
    implementation = _pywrap_def_file_impl,
    attrs = {
        "deps": attr.label_list(),
        "unmangled_symbols_file": attr.label(allow_single_file = True, mandatory = True),
        "static_export_table": attr.label(allow_single_file = True, mandatory = True),
        "out": attr.output(mandatory = True),
        "win_tool": attr.label(executable = True, cfg = "exec", allow_files = True),
        "exit_1": attr.bool(default = False),
    },
)
