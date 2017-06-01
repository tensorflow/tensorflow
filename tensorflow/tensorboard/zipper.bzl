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

def _tensorboard_zip_file(ctx):
  deps = unfurl(ctx.attr.deps, provider="webfiles")
  manifests = set(order="link")
  files = set()
  webpaths = set()
  for dep in deps:
    manifests += dep.webfiles.manifests
    webpaths += dep.webfiles.webpaths
    files += dep.data_runfiles.files
  ctx.action(
      inputs=list(manifests + files),
      outputs=[ctx.outputs.zip],
      executable=ctx.executable._Zipper,
      arguments=([ctx.outputs.zip.path] +
                 [m.path for m in manifests]),
      progress_message="Zipping %d files" % len(webpaths))
  transitive_runfiles = set()
  for dep in deps:
    transitive_runfiles += dep.data_runfiles.files
  return struct(
      files=set([ctx.outputs.zip]),
      runfiles=ctx.runfiles(
          files=ctx.files.data + [ctx.outputs.zip],
          transitive_files=transitive_runfiles))

tensorboard_zip_file = rule(
    implementation=_tensorboard_zip_file,
    attrs={
        "data": attr.label_list(cfg="data", allow_files=True),
        "deps": attr.label_list(providers=["webfiles"], mandatory=True),
        "_Zipper": attr.label(
            default=Label("//tensorflow/tensorboard/java/org/tensorflow/tensorboard/vulcanize:Zipper"),
            executable=True,
            cfg="host"),
    },
    outputs={
        "zip": "%{name}.zip",
    })
