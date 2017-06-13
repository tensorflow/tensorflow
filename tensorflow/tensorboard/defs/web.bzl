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

load("//tensorflow/tensorboard/defs:defs.bzl", "legacy_js")

load("//third_party:clutz.bzl",
     "CLUTZ_ATTRIBUTES",
     "CLUTZ_OUTPUTS",
     "clutz_aspect",
     "extract_dts_from_closure_libraries")

load("@io_bazel_rules_closure//closure/private:defs.bzl",
     "CLOSURE_LIBRARY_BASE_ATTR",
     "CLOSURE_LIBRARY_DEPS_ATTR",
     "collect_js",
     "collect_runfiles",
     "convert_path_to_es6_module_name",
     "create_argfile",
     "difference",
     "long_path",
     "unfurl")

_ASPECT_SLURP_FILE_TYPE = FileType([
    ".html", ".js", ".css", ".gss", ".png", ".jpg", ".gif", ".ico", ".svg"])

_CLOSURE_WORKER = attr.label(
    default=Label("@io_bazel_rules_closure//java/io/bazel/rules/closure:ClosureWorker"),
    executable=True,
    cfg="host")

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
  ts_typings = depset(ctx.files._default_typings)
  ts_typings_paths = depset(
      [long_path(ctx, f) for f in ctx.files._default_typings])
  ts_typings_execroots = depset()
  aspect_runfiles = depset()
  for dep in deps:
    webpaths += dep.webfiles.webpaths
    if hasattr(dep.webfiles, "ts_typings"):
      ts_typings += dep.webfiles.ts_typings
    if hasattr(dep.webfiles, "ts_typings_paths"):
      ts_typings_paths += dep.webfiles.ts_typings_paths
    if hasattr(dep.webfiles, "ts_typings_execroots"):
      ts_typings_execroots += dep.webfiles.ts_typings_execroots
    if hasattr(dep.webfiles, "aspect_runfiles"):
      aspect_runfiles += dep.webfiles.aspect_runfiles

  # process what comes now
  manifest_srcs = []
  new_webpaths = []
  ts_inputs = depset()
  ts_outputs = []
  ts_files = list(ts_typings_paths)
  new_typings = []
  new_typings_paths = []
  new_typings_execroot = struct(inputs=[])
  execroot = struct(
      inputs=[(long_path(ctx, f), f.path) for f in ctx.files._default_typings],
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

  # get typings for closure code
  clutz_dts = extract_dts_from_closure_libraries(ctx)
  if clutz_dts:
    entry = (long_path(ctx, clutz_dts), clutz_dts.path)
    ts_inputs += [clutz_dts]
    ts_files.append(entry[0])
    execroot.inputs.append(entry)

  # compile typescript
  workspace = ""
  if ctx.label.workspace_root:
    workspace = "/" + ctx.label.workspace_root
  if execroot.outputs:
    ts_config = _new_file(ctx, "-tsc.json")
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
            files=ts_files,
        ).to_json())
    er_config = _new_file(ctx, "-tsc-execroot.json")
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
        progress_message="Compiling %d TypeScript files %s" % (
            len(ts_files), ctx.label))

  # perform strict dependency checking
  manifest = _make_manifest(ctx, manifest_srcs)
  webpaths += new_webpaths
  dummy, manifests = _run_webfiles_validator(ctx, web_srcs, deps, manifest)
  web_srcs.append(dummy)

  # define development web server that only applies to this transitive closure
  params = struct(
      label=str(ctx.label),
      bind="[::]:6006",
      manifest=[long_path(ctx, man) for man in manifests],
      external_asset=[struct(webpath=k, path=v)
                      for k, v in ctx.attr.external_assets.items()])
  params_file = _new_file(ctx, "-params.pbtxt")
  ctx.file_action(output=params_file, content=params.to_proto())
  ctx.file_action(
      executable=True,
      output=ctx.outputs.executable,
      content="#!/bin/sh\nexec %s %s" % (
          ctx.executable._WebfilesServer.short_path,
          long_path(ctx, params_file)))

  if new_typings:
    er_config = _new_file(ctx, "-typings-execroot.json")
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
      files=depset(web_srcs + [dummy]),
      exports=unfurl(ctx.attr.exports),
      webfiles=struct(
          manifest=manifest,
          manifests=manifests,
          webpaths=webpaths,
          dummy=dummy,
          ts_typings=ts_typings,
          ts_typings_paths=ts_typings_paths,
          ts_typings_execroots=ts_typings_execroots),
      closure_js_library=collect_js(
          ctx, unfurl(ctx.attr.deps, provider="closure_js_library")),
      runfiles=ctx.runfiles(
          files=ctx.files.srcs + ctx.files.data + ts_outputs + [
              manifest,
              params_file,
              ctx.outputs.executable,
              dummy],
          transitive_files=(collect_runfiles([ctx.attr._WebfilesServer]) |
                            collect_runfiles(deps) |
                            collect_runfiles(ctx.attr.data) |
                            aspect_runfiles)))

def _web_aspect_impl(target, ctx):
  if hasattr(target, "webfiles"):
    return struct()
  srcs = []
  deps = []
  if hasattr(ctx.rule.files, "srcs"):
    srcs.extend(_ASPECT_SLURP_FILE_TYPE.filter(ctx.rule.files.srcs))
  for attr in ("deps", "sticky_deps", "module_deps"):
    value = getattr(ctx.rule.attr, attr, None)
    if value:
      deps.extend(value)
  deps = unfurl(deps, provider="webfiles")
  webpaths = depset()
  aspect_runfiles = depset(srcs)
  for dep in deps:
    webpaths += dep.webfiles.webpaths
    if hasattr(dep.webfiles, "aspect_runfiles"):
      aspect_runfiles += dep.webfiles.aspect_runfiles
  manifest_srcs = []
  new_webpaths = []
  for src in srcs:
    webpath = "/" + long_path(ctx, src)
    _add_webpath(ctx, src, webpath, webpaths, new_webpaths, manifest_srcs)
  webpaths += new_webpaths
  manifest = _make_manifest(ctx, manifest_srcs)
  dummy, manifests = _run_webfiles_validator(ctx, srcs, deps, manifest)
  aspect_runfiles += [dummy, manifest]
  return struct(
      webfiles=struct(
          manifest=manifest,
          manifests=manifests,
          webpaths=webpaths,
          dummy=dummy,
          aspect_runfiles=aspect_runfiles))

def _make_manifest(ctx, src_list):
  manifest = _new_file(ctx, "-webfiles.pbtxt")
  ctx.file_action(
      output=manifest,
      content=struct(
          label=str(ctx.label),
          src=src_list).to_proto())
  return manifest

def _run_webfiles_validator(ctx, srcs, deps, manifest):
  dummy = _new_file(ctx, "-webfiles.ignoreme")
  manifests = depset(order="topological")
  for dep in deps:
    manifests += dep.webfiles.manifests
  if srcs:
    args = ["WebfilesValidator",
            "--dummy", dummy.path,
            "--target", manifest.path]
    if hasattr(ctx, "attr") and hasattr(ctx.attr, "suppress"):
      for category in ctx.attr.suppress:
        args.append("--suppress")
        args.append(category)
    inputs = [manifest]
    inputs.extend(srcs)
    direct_manifests = depset()
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
    argfile = _new_file(ctx, "-webfiles-checker-args.txt")
    ctx.file_action(output=argfile, content="\n".join(args))
    inputs.append(argfile)
    ctx.action(
        inputs=inputs,
        outputs=[dummy],
        executable=(getattr(ctx.executable, "_ClosureWorker", None) or
                    getattr(ctx.executable, "_ClosureWorkerAspect", None)),
        arguments=["@@" + argfile.path],
        mnemonic="Closure",
        execution_requirements={"supports-workers": "1"},
        progress_message="Checking webfiles %s" % ctx.label)
  else:
    ctx.file_action(output=dummy, content="BOO!")
  manifests += [manifest]
  return dummy, manifests

def _new_file(ctx, suffix):
  return ctx.new_file(ctx.bin_dir, "%s%s" % (ctx.label.name, suffix))

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
    attr_aspects=["deps", "sticky_deps", "module_deps"],
    attrs={"_ClosureWorkerAspect": _CLOSURE_WORKER})

ts_web_library = rule(
    implementation=_ts_web_library,
    executable=True,
    attrs=CLUTZ_ATTRIBUTES + {
        "path": attr.string(),
        "srcs": attr.label_list(allow_files=True),
        "deps": attr.label_list(
            aspects=[
                web_aspect,
                clutz_aspect,
                legacy_js,
            ]),
        "exports": attr.label_list(),
        "data": attr.label_list(cfg="data", allow_files=True),
        "suppress": attr.string_list(),
        "strip_prefix": attr.string(),
        "external_assets": attr.string_dict(default={"/_/runfiles": "."}),
        "clutz_entry_points": attr.string_list(),
        "_execrooter": attr.label(
            default=Label("//tensorflow/tensorboard/scripts:execrooter"),
            executable=True,
            cfg="host"),
        "_tsc": attr.label(
            default=Label("@com_microsoft_typescript//:tsc"),
            allow_files=True,
            executable=True,
            cfg="host"),
        "_default_typings": attr.label(
            default=Label("//tensorflow/tensorboard:ts_web_library_default_typings"),
            allow_files=True),
        "_WebfilesServer": attr.label(
            default=Label("@io_bazel_rules_closure//java/io/bazel/rules/closure/webfiles/server:WebfilesServer"),
            executable=True,
            cfg="host"),
        "_ClosureWorker": _CLOSURE_WORKER,
        "_closure_library_base": CLOSURE_LIBRARY_BASE_ATTR,
        "_closure_library_deps": CLOSURE_LIBRARY_DEPS_ATTR,
    },
    outputs=CLUTZ_OUTPUTS)
