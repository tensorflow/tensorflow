# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build definitions for TypeScript from Closure JavaScript libraries."""

load("@io_bazel_rules_closure//closure/private:defs.bzl",
     "JS_FILE_TYPE",
     "collect_js",
     "unfurl")

CLUTZ_ATTRIBUTES = {
    "_clutz": attr.label(
        default=Label("@io_angular_clutz//:clutz"),
        executable=True,
        cfg="host"),
    "_clutz_externs": attr.label(
        default=Label("@com_google_javascript_closure_compiler_externs"),
        allow_files=True),
}

def extract_dts_from_closure_libraries(ctx):
  """Extracts type definitions from closure dependencies.

  This just generates one big .d.ts file for all transitive Closure sources,
  and does not pass it down. That means each rule has to duplicate the effort,
  but on the other hand allows transitive dependencies on shared rules without
  causing duplicate definition errors.

  Args:
      ctx: A Skylark context.
  Returns:
      The generated Clutz typings file, or None if there were no JS deps.
  """
  deps = unfurl(ctx.attr.deps, provider="closure_js_library")
  js = collect_js(ctx, deps)
  if not js.srcs:
    return None
  js_typings = ctx.new_file(ctx.bin_dir, "%s-js-typings.d.ts" % ctx.label.name)
  srcs = depset(JS_FILE_TYPE.filter(ctx.files._clutz_externs)) + js.srcs
  args = ["-o", js_typings.path]
  for src in srcs:
    args.append(src.path)
  if getattr(ctx.attr, "clutz_entry_points", None):
    args.append("--closure_entry_points")
    args.extend(ctx.attr.clutz_entry_points)
  ctx.action(
      inputs=list(srcs),
      outputs=[js_typings],
      executable=ctx.executable._clutz,
      arguments=args,
      mnemonic="Clutz",
      progress_message="Running Clutz on %d JS files %s" % (
          len(srcs), ctx.label))
  return js_typings

################################################################################
# The following definitions are for API compatibility with internal clutz.bzl

CLUTZ_OUTPUTS = {}

def _clutz_aspect_impl(target, ctx):
  return struct()

clutz_aspect = aspect(
    implementation=_clutz_aspect_impl,
    attr_aspects=["exports"])
