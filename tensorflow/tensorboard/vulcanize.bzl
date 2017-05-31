# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

load("@io_bazel_rules_closure//closure/private:defs.bzl", "unfurl", "long_path")

def _tensorboard_html_binary(ctx):
  deps = unfurl(ctx.attr.deps, provider="webfiles")
  manifests = set()
  files = set()
  webpaths = set()
  for dep in deps:
    manifests += dep.webfiles.manifests
    webpaths += dep.webfiles.webpaths
    files += dep.data_runfiles.files
  webpaths += [ctx.attr.output_path]

  # vulcanize
  ctx.action(
      inputs=list(manifests + files),
      outputs=[ctx.outputs.html],
      executable=ctx.executable._Vulcanize,
      arguments=([ctx.attr.input_path,
                  ctx.attr.output_path,
                  ctx.outputs.html.path] +
                 [m.path for m in manifests]),
      progress_message="Vulcanizing %s" % ctx.attr.input_path)

  # webfiles manifest
  manifest_srcs = [struct(path=ctx.outputs.html.path,
                          longpath=long_path(ctx, ctx.outputs.html),
                          webpath=ctx.attr.output_path)]
  manifest = ctx.new_file(ctx.configuration.bin_dir,
                          "%s.pbtxt" % ctx.label.name)
  ctx.file_action(
      output=manifest,
      content=struct(
          label=str(ctx.label),
          src=manifest_srcs).to_proto())
  manifests += [manifest]

  # webfiles server
  params = struct(
      label=str(ctx.label),
      bind="[::]:6006",
      manifest=[long_path(ctx, man) for man in manifests],
      external_asset=[struct(webpath=k, path=v)
                      for k, v in ctx.attr.external_assets.items()])
  params_file = ctx.new_file(ctx.configuration.bin_dir,
                             "%s_server_params.pbtxt" % ctx.label.name)
  ctx.file_action(output=params_file, content=params.to_proto())
  ctx.file_action(
      executable=True,
      output=ctx.outputs.executable,
      content="#!/bin/sh\nexec %s %s" % (
          ctx.executable._WebfilesServer.short_path,
          long_path(ctx, params_file)))

  transitive_runfiles = set()
  transitive_runfiles += ctx.attr._WebfilesServer.data_runfiles.files
  for dep in deps:
    transitive_runfiles += dep.data_runfiles.files
  return struct(
      files=set([ctx.outputs.html]),
      webfiles=struct(
          manifest=manifest,
          manifests=manifests,
          webpaths=webpaths,
          dummy=ctx.outputs.html),
      runfiles=ctx.runfiles(
          files=ctx.files.data + [manifest,
                                  params_file,
                                  ctx.outputs.html,
                                  ctx.outputs.executable],
          transitive_files=transitive_runfiles))

tensorboard_html_binary = rule(
    implementation=_tensorboard_html_binary,
    executable=True,
    attrs={
        "input_path": attr.string(mandatory=True),
        "output_path": attr.string(mandatory=True),
        "data": attr.label_list(cfg="data", allow_files=True),
        "deps": attr.label_list(providers=["webfiles"], mandatory=True),
        "external_assets": attr.string_dict(default={"/_/runfiles": "."}),
        "_Vulcanize": attr.label(
            default=Label("//tensorflow/tensorboard/java/org/tensorflow/tensorboard/vulcanize:Vulcanize"),
            executable=True,
            cfg="host"),
        "_WebfilesServer": attr.label(
            default=Label(
                "@io_bazel_rules_closure//java/io/bazel/rules/closure/webfiles/server:WebfilesServer"),
            executable=True,
            cfg="host"),
    },
    outputs={
        "html": "%{name}.html",
    })
