# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

_DEFAULT_TYPINGS = [
    "@com_microsoft_typescript//:lib.es6.d.ts",
]

def tensorboard_typescript_genrule(name, srcs, typings=[], **kwargs):
  """Filegroup of compiled TypeScript sources.

  This is a very unsophisticated TypeScript rule where the user is responsible
  for passing all typings and sources via srcs. It's meant as a stopgap because
  TypeScript rules currently don't exist for Bazel. The definition of this rule
  will need to evolve as more ts_library rules are migrated.
  """
  for src in srcs:
    if (src.startswith("/") or
        src.endswith(".d.ts") or
        not src.endswith(".ts")):
      fail("srcs must be typescript sources in same package")
  typings_out = [src[:-3] + ".d.ts" for src in srcs]
  inputs = _DEFAULT_TYPINGS + typings + srcs
  # These inputs are meant to work around a sandbox bug in Bazel. If we list
  # @com_microsoft_typescript//:tsc.sh under tools, then its
  # data attribute won't be considered when --genrule_strategy=sandboxed. See
  # https://github.com/bazelbuild/bazel/issues/1147 and its linked issues.
  data = [
      "@org_nodejs",
      "@com_microsoft_typescript",
  ]
  native.genrule(
      name = name,
      srcs = inputs + data,
      outs = [src[:-3] + ".js" for src in srcs] + typings_out,
      cmd = "$(location @com_microsoft_typescript//:tsc.sh)" +
            " --inlineSourceMap" +
            " --inlineSources" +
            " --declaration" +
            " --outDir $(@D) " +
            " ".join(["$(locations %s)" % i for i in inputs]),
      tools = ["@com_microsoft_typescript//:tsc.sh"],
      **kwargs
  )
  native.filegroup(
      name = name + "_typings",
      srcs = typings_out,
      **kwargs
  )

def tensorboard_ts_library(**kwargs):
  """Rules referencing this will be deleted from the codebase soon."""
  pass

def tensorboard_webcomponent_library(**kwargs):
  """Rules referencing this will be deleted from the codebase soon."""
  pass

def tensorboard_wct_test_suite(**kwargs):
  """Rules referencing this will be deleted from the codebase soon."""
  pass
