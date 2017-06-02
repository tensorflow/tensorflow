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

"""Same as web_library but supports TypeScript."""

load("@io_bazel_rules_closure//closure/private:defs.bzl",
     "collect_runfiles",
     "convert_path_to_es6_module_name",
     "create_argfile",
     "difference",
     "long_path",
     "unfurl")

def _ts_web_library(ctx):
  if not ctx.attr.srcs:
    if ctx.attr.deps:
      fail("deps can not be set when srcs is not")
    if not ctx.attr.exports:
      fail("exports must be set if srcs is not")
  if ctx.attr.path:
    if not ctx.attr.path.startswith("/"):
      fail("webpath must start with /")
    if ctx.attr.path != "/" and ctx.attr.path.endswith("/"):
      fail("webpath must not end with / unless it is /")
    if "//" in ctx.attr.path:
      fail("webpath must not have //")
  elif ctx.attr.srcs:
    fail("path must be set when srcs is set")
  if "*" in ctx.attr.suppress and len(ctx.attr.suppress) != 1:
    fail("when \"*\" is suppressed no other items should be present")

  # process what came before
  deps = unfurl(ctx.attr.deps, provider="webfiles")
  webpaths = depset()
  manifests = depset(order="topological")
  jslibs = depset(order="postorder")
  ts_typings = depset(ctx.files._es6dts)
  ts_typings_paths = depset()
  ts_typings_execroots = depset()
  for dep in deps:
    webpaths += dep.webfiles.webpaths
    manifests += dep.webfiles.manifests
    if hasattr(dep.webfiles, "ts_typings"):
      ts_typings += dep.webfiles.ts_typings
    if hasattr(dep.webfiles, "ts_typings_paths"):
      ts_typings_paths += dep.webfiles.ts_typings_paths
    if hasattr(dep.webfiles, "ts_typings_execroots"):
      ts_typings_execroots += dep.webfiles.ts_typings_execroots
    if hasattr(dep.webfiles, "jslibs"):
      jslibs += dep.webfiles.jslibs
    if hasattr(dep, "closure_js_library"):
      jslibs += getattr(dep.closure_js_library, "srcs", [])

  # process what comes now
  manifest_srcs = []
  new_webpaths = []
  ts_inputs = depset()
  ts_outputs = []
  ts_files = ["lib.es6.d.ts"] + list(ts_typings_paths)
  new_typings = []
  new_typings_paths = []
  new_typings_execroot = struct(inputs=[])
  execroot = struct(
      inputs=[("lib.es6.d.ts", ctx.files._es6dts[0].path)],
      outputs=[],
      program=[ctx.executable._tsc.path, "-p"])
  web_srcs = []
  path = ctx.attr.path
  strip = _get_strip(ctx)
  for src in ctx.files.srcs:
    suffix = _get_path_relative_to_package(src)
    if strip:
      if not suffix.startswith(strip):
        fail("Relative src path not start with '%s': %s" % (strip, suffix))
      suffix = suffix[len(strip):]
    webpath = "%s/%s" % ("" if path == "/" else path, suffix)
    _add_webpath(ctx, src, webpath, webpaths, new_webpaths, manifest_srcs)
    if suffix.endswith(".d.ts"):
      web_srcs.append(src)
      entry = (webpath[1:], src.path)
      new_typings.append(src)
      new_typings_paths.append(entry[0])
      new_typings_execroot.inputs.append(entry)
      ts_inputs += [src]
      ts_files.append(entry[0])
      execroot.inputs.append(entry)
    elif suffix.endswith(".ts"):
      noext = suffix[:-3]
      js = ctx.new_file(ctx.bin_dir, "%s.js" % noext)
      dts = ctx.new_file(ctx.bin_dir, "%s.d.ts" % noext)
      webpath_js = webpath[:-3] + ".js"
      webpath_dts = webpath[:-3] + ".d.ts"
      _add_webpath(ctx, js, webpath_js, webpaths, new_webpaths, manifest_srcs)
      _add_webpath(ctx, dts, webpath_dts, webpaths, new_webpaths, manifest_srcs)
      ts_inputs += [src]
      ts_outputs.append(js)
      ts_outputs.append(dts)
      web_srcs.append(dts)
      web_srcs.append(js)
      ts_files.append(webpath[1:])
      execroot.inputs.append((webpath[1:], src.path))
      execroot.outputs.append((webpath_js[1:], js.path))
      execroot.outputs.append((webpath_dts[1:], dts.path))
      new_typings.append(dts)
      new_typings_paths.append(webpath_dts[1:])
      new_typings_execroot.inputs.append((webpath_dts[1:], dts.path))
    else:
      web_srcs.append(src)

  # create webfiles manifest
  manifest = ctx.new_file(ctx.configuration.bin_dir,
                          "%s.pbtxt" % ctx.label.name)
  ctx.file_action(
      output=manifest,
      content=struct(
          label=str(ctx.label),
          src=manifest_srcs).to_proto())
  manifests += [manifest]
  webpaths += new_webpaths

  # compile typescript
  workspace = ""
  if ctx.label.workspace_root:
    workspace = "/" + ctx.label.workspace_root
  if execroot.outputs:
    ts_config = ctx.new_file(ctx.bin_dir, "%s-tsc.json" % ctx.label.name)
    execroot.inputs.append(("tsconfig.json", ts_config.path))
    ctx.file_action(
        output=ts_config,
        content=struct(
            compilerOptions=struct(
                baseUrl=".",
                declaration=True,
                inlineSourceMap=True,
                inlineSources=True,
                module="es6",
                moduleResolution="node",
                noResolve=True,
                target="es5",
            ),
            files=list(ts_files),
        ).to_json())
    er_config = ctx.new_file(ctx.bin_dir,
                             "%s-tsc-execroot.json" % ctx.label.name)
    ctx.file_action(output=er_config, content=execroot.to_json())
    ts_inputs += collect_runfiles([ctx.attr._tsc])
    ts_inputs += ctx.files._tsc
    ts_inputs += ts_typings
    ts_inputs += ts_typings_execroots
    ts_inputs += [ts_config, er_config]
    ctx.action(
        inputs=list(ts_inputs),
        outputs=ts_outputs,
        executable=ctx.executable._execrooter,
        arguments=[er_config.path] + [f.path for f in ts_typings_execroots],
        progress_message="Compiling %d TypeScript files" % len(ts_files))

  # perform strict dependency checking
  inputs = [manifest]
  direct_manifests = depset([manifest])
  args = ["WebfilesValidator",
          "--dummy", ctx.outputs.dummy.path,
          "--target", manifest.path]
  for category in ctx.attr.suppress:
    args.append("--suppress")
    args.append(category)
  inputs.extend(web_srcs)
  for dep in deps:
    inputs.append(dep.webfiles.dummy)
    for f in dep.files:
      inputs.append(f)
    direct_manifests += [dep.webfiles.manifest]
    inputs.append(dep.webfiles.manifest)
    args.append("--direct_dep")
    args.append(dep.webfiles.manifest.path)
  for man in difference(manifests, direct_manifests):
    inputs.append(man)
    args.append("--transitive_dep")
    args.append(man.path)
  argfile = create_argfile(ctx, args)
  inputs.append(argfile)
  ctx.action(
      inputs=inputs,
      outputs=[ctx.outputs.dummy],
      executable=ctx.executable._ClosureWorker,
      arguments=["@@" + argfile.path],
      mnemonic="Closure",
      execution_requirements={"supports-workers": "1"},
      progress_message="Checking webfiles in %s" % ctx.label)
  web_srcs.append(ctx.outputs.dummy)

  # define development web server that only applies to this transitive closure
  params = struct(
      label=str(ctx.label),
      bind="[::]:6006",
      manifest=[long_path(ctx, man) for man in manifests],
      external_asset=[struct(webpath=k, path=v)
                      for k, v in ctx.attr.external_assets.items()])
  params_file = ctx.new_file(ctx.bin_dir, "%s_params.pbtxt" % ctx.label.name)
  ctx.file_action(output=params_file, content=params.to_proto())
  ctx.file_action(
      executable=True,
      output=ctx.outputs.executable,
      content="#!/bin/sh\nexec %s %s" % (
          ctx.executable._WebfilesServer.short_path,
          long_path(ctx, params_file)))

  if new_typings:
    er_config = ctx.new_file(ctx.bin_dir,
                             "%s-typings-execroot.json" % ctx.label.name)
    ctx.file_action(output=er_config, content=new_typings_execroot.to_json())
    ts_typings += new_typings
    ts_typings_paths += new_typings_paths
    ts_typings_execroots += [er_config]
  else:
    ts_typings = depset()
    ts_typings_paths = depset()
    ts_typings_execroots = depset()

  # export data to parent rules
  return struct(
      files=depset(web_srcs),
      exports=unfurl(ctx.attr.exports),
      webfiles=struct(
          manifest=manifest,
          manifests=manifests,
          webpaths=webpaths,
          dummy=ctx.outputs.dummy,
          jslibs=jslibs,
          ts_typings=ts_typings,
          ts_typings_paths=ts_typings_paths,
          ts_typings_execroots=ts_typings_execroots),
      runfiles=ctx.runfiles(
          files=ctx.files.srcs + ctx.files.data + ts_outputs + [
              manifest,
              params_file,
              ctx.outputs.executable,
              ctx.outputs.dummy],
          transitive_files=(collect_runfiles([ctx.attr._WebfilesServer]) |
                            collect_runfiles(deps) |
                            collect_runfiles(ctx.attr.data))))

def _web_aspect_impl(target, ctx):
  if ctx.rule.kind in ("js_library", "pinto_library"):
    return _web_aspect_js_library(target, ctx, [], depset())
  if hasattr(target, "js"):
    return _web_aspect_js_library(
        target,
        ctx,
        target.files,
        target.js.full_tc(True))
  return struct()

def _web_aspect_js_library(target, ctx, extra_srcs, extra_transitive):
  deps = unfurl((ctx.rule.attr.deps +
                 getattr(ctx.rule.attr, 'sticky_deps', [])),
                provider="webfiles")
  # process what came before
  webpaths = depset()
  manifests = depset(order="topological")
  jslibs = depset(order="postorder")
  for dep in deps:
    webpaths += dep.webfiles.webpaths
    manifests += dep.webfiles.manifests
    if hasattr(dep.webfiles, "jslibs"):
      jslibs += dep.webfiles.jslibs
  # process what comes now
  srcs = ctx.rule.files.srcs + extra_srcs
  jslibs += [src for src in srcs if src.path.endswith(".js")]
  manifest_srcs = []
  new_webpaths = []
  web_srcs = []
  for src in srcs:
    webpath = "/" + long_path(ctx, src)
    _add_webpath(ctx, src, webpath, webpaths, new_webpaths, manifest_srcs)
    web_srcs.append(src)
  # create webfiles manifest
  manifest = ctx.new_file(ctx.configuration.bin_dir,
                          "%s-webfiles.pbtxt" % ctx.label.name)
  ctx.file_action(
      output=manifest,
      content=struct(
          label=str(ctx.label),
          src=manifest_srcs).to_proto())
  manifests += [manifest]
  webpaths += new_webpaths
  return struct(
      exports=[] if srcs else deps,
      webfiles=struct(
          manifest=manifest,
          manifests=manifests,
          webpaths=webpaths,
          dummy=manifest,
          jslibs=jslibs),
      closure_legacy_js_runfiles=(depset(srcs + ctx.rule.files.data) |
                                  extra_transitive |
                                  collect_runfiles(deps) |
                                  collect_runfiles(ctx.rule.files.data)))

def _add_webpath(ctx, src, webpath, webpaths, new_webpaths, manifest_srcs):
  if webpath in new_webpaths:
    _fail(ctx, "multiple srcs within %s define the webpath %s " % (
        ctx.label, webpath))
  if webpath in webpaths:
    _fail(ctx, "webpath %s was defined by %s when already defined by deps" % (
        webpath, ctx.label))
  new_webpaths.append(webpath)
  manifest_srcs.append(struct(
      path=src.path,
      longpath=long_path(ctx, src),
      webpath=webpath))

def _fail(ctx, message):
  if ctx.attr.suppress == ["*"]:
    print(message)
  else:
    fail(message)

def _get_path_relative_to_package(artifact):
  """Returns file path relative to the package that declared it."""
  path = artifact.path
  for prefix in (artifact.root.path,
                 artifact.owner.workspace_root if artifact.owner else '',
                 artifact.owner.package if artifact.owner else ''):
    if prefix:
      prefix = prefix + "/"
      if not path.startswith(prefix):
        fail("Path %s doesn't start with %s" % (path, prefix))
      path = path[len(prefix):]
  return path

def _get_strip(ctx):
  strip = ctx.attr.strip_prefix
  if strip:
    if strip.startswith("/"):
      _fail(ctx, "strip_prefix should not end with /")
      strip = strip[1:]
    if strip.endswith("/"):
      _fail(ctx, "strip_prefix should not end with /")
    else:
      strip += "/"
  return strip

web_aspect = aspect(
    implementation=_web_aspect_impl,
    attr_aspects=["deps"])

ts_web_library = rule(
    implementation=_ts_web_library,
    executable=True,
    attrs={
        "path": attr.string(),
        "srcs": attr.label_list(allow_files=True),
        "deps": attr.label_list(aspects=[web_aspect], providers=["webfiles"]),
        "exports": attr.label_list(),
        "data": attr.label_list(cfg="data", allow_files=True),
        "suppress": attr.string_list(),
        "strip_prefix": attr.string(),
        "external_assets": attr.string_dict(default={"/_/runfiles": "."}),
        "_execrooter": attr.label(
            default=Label(
                "//tensorflow/tensorboard/scripts:execrooter"),
            executable=True,
            cfg="host"),
        "_tsc": attr.label(
            default=Label(
                "@com_microsoft_typescript//:tsc"),
            allow_files=True,
            executable=True,
            cfg="host"),
        "_es6dts": attr.label(
            default=Label(
                "@com_microsoft_typescript//:lib.es6.d.ts"),
            allow_files=True),
        "_ClosureWorker": attr.label(
            default=Label("@io_bazel_rules_closure//java/io/bazel/rules/closure:ClosureWorker"),
            executable=True,
            cfg="host"),
        "_WebfilesServer": attr.label(
            default=Label(
                "@io_bazel_rules_closure//java/io/bazel/rules/closure/webfiles/server:WebfilesServer"),
            executable=True,
            cfg="host"),
    },
    outputs={
        "dummy": "%{name}.ignoreme",
    })
