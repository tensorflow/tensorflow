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

"""Repository rule for Debian 8 Jessie Clang-6.0 portable Linux builds."""

def _clang6_configure(ctx):
    # TODO(jart): It'd probably be better to use Bazel's struct.to_proto()
    #             method to generate a gigantic CROSSTOOL file that allows
    #             Clang to support everything.
    ctx.symlink(
        ctx.os.environ.get(
            "TF_LLVM_PATH",
            "/usr/lib/llvm-6.0",
        ),
        "clang6/llvm",
    )
    ctx.symlink(
        ctx.os.environ.get("STRIP", "/usr/bin/strip"),
        "clang6/sbin/strip",
    )
    ctx.symlink(
        ctx.os.environ.get("OBJDUMP", "/usr/bin/objdump"),
        "clang6/sbin/objdump",
    )
    ctx.symlink(ctx.attr._build, "clang6/BUILD")
    ctx.template("clang6/CROSSTOOL", ctx.attr._crosstool, {
        "%package(@local_config_clang6//clang6)%": str(ctx.path("clang6")),
    })

clang6_configure = repository_rule(
    implementation = _clang6_configure,
    attrs = {
        "_build": attr.label(
            default = str(Label("//tensorflow/tools/toolchains/clang6:clang.BUILD")),
        ),
        "_crosstool": attr.label(
            default = str(Label("//tensorflow/tools/toolchains/clang6:CROSSTOOL.tpl")),
        ),
    },
)
