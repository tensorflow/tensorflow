# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TensorBoard external JS dependencies (both infrastructure and frontend libs)
load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")
load("@io_bazel_rules_closure//closure:defs.bzl", "web_library_external")


  ##############################################################################
  # TensorBoard Build Tools
def tensorboard_js_workspace():
  filegroup_external(
      name = "org_nodejs",
      # MIT with portions licensed:
      # - MIT
      # - Old MIT
      # - 2-Clause-BSD
      # - 3-Clause-BSD
      # - ISC
      # - Unicode
      # - zlib
      # - Artistic 2.0
      licenses = ["notice"],
      sha256_urls_extract_macos = {
          "47109a00cac344d80296c195451bb5eee7c21727fcef1594384ddfe1f852957a": [
              "http://mirror.bazel.build/nodejs.org/dist/v4.3.2/node-v4.3.2-darwin-x64.tar.xz",
              "http://nodejs.org/dist/v4.3.2/node-v4.3.2-darwin-x64.tar.xz",
          ],
      },
      sha256_urls_windows = {
          "3d4cfca9dcec556a077a2324bf5bd165ea3e6e64a2bfd7fc6e7a1f0dc4eb552b": [
              "http://mirror.bazel.build/raw.githubusercontent.com/nodejs/node/v4.3.2/LICENSE",
              "https://raw.githubusercontent.com/nodejs/node/v4.3.2/LICENSE",
          ],
          "606c44c42d17866c017c50c0afadad411d9492ac4281d2431b937f881911614e": [
              "http://mirror.bazel.build/nodejs.org/dist/v4.3.2/win-x64/node.exe",
              "http://nodejs.org/dist/v4.3.2/win-x64/node.exe",
          ],
          "451a40570099a95488d6438f175813629e0430f87f23c8659bc18dc42494820a": [
              "http://mirror.bazel.build/nodejs.org/dist/v4.3.2/win-x64/node.lib",
              "http://nodejs.org/dist/v4.3.2/win-x64/node.lib",
          ],
      },
      sha256_urls_extract = {
          "4350d0431b49697517c6cca5d66adf5f74eb9101c52f52ae959fa94225822d44": [
              "http://mirror.bazel.build/nodejs.org/dist/v4.3.2/node-v4.3.2-linux-x64.tar.xz",
              "http://nodejs.org/dist/v4.3.2/node-v4.3.2-linux-x64.tar.xz",
          ],
      },
      strip_prefix = {
          "node-v4.3.2-darwin-x64.tar.xz": "node-v4.3.2-darwin-x64",
          "node-v4.3.2-linux-x64.tar.xz": "node-v4.3.2-linux-x64",
      },
      executable = [
          "node",
          "node.exe",
      ],
  )
  
  filegroup_external(
      name = "com_microsoft_typescript",
      licenses = ["notice"],  # Apache 2.0
      sha256_urls = {
          "a7d00bfd54525bc694b6e32f64c7ebcf5e6b7ae3657be5cc12767bce74654a47": [
              "http://mirror.bazel.build/raw.githubusercontent.com/Microsoft/TypeScript/v2.3.1/LICENSE.txt",
              "https://raw.githubusercontent.com/Microsoft/TypeScript/v2.3.1/LICENSE.txt",
          ],
          "8465342c318f9c4cf0a29b109fa63ee3742dd4dc7080d05d9fd8f604814d04cf": [
              "http://mirror.bazel.build/raw.githubusercontent.com/Microsoft/TypeScript/v2.3.1/lib/tsc.js",
              "https://raw.githubusercontent.com/Microsoft/TypeScript/v2.3.1/lib/tsc.js",
          ],
          "a67e36da3029d232e4e938e61a0a3302f516d71e7100d54dbf5362ad8618e994": [
              "http://mirror.bazel.build/raw.githubusercontent.com/Microsoft/TypeScript/v2.3.1/lib/lib.es6.d.ts",
              "https://raw.githubusercontent.com/Microsoft/TypeScript/v2.3.1/lib/lib.es6.d.ts",
          ],
      },
      extra_build_file_content = "\n".join([
          "sh_binary(",
          "    name = \"tsc\",",
          "    srcs = [\"tsc.sh\"],",
          "    data = [",
          "        \"tsc.js\",",
          "        \"@org_nodejs\",",
          "    ],",
          ")",
          "",
          "genrule(",
          "    name = \"tsc_sh\",",
          "    outs = [\"tsc.sh\"],",
          "    cmd = \"cat >$@ <<'EOF'\\n\" +",
          "          \"#!/bin/bash\\n\" +",
          "          \"NODE=external/org_nodejs/bin/node\\n\" +",
          "          \"if [[ -e external/org_nodejs/node.exe ]]; then\\n\" +",
          "          \"  NODE=external/org_nodejs/node.exe\\n\" +",
          "          \"fi\\n\" +",
          "          \"exec $${NODE} external/com_microsoft_typescript/tsc.js \\\"$$@\\\"\\n\" +",
          "          \"EOF\",",
          "    executable = True,",
          ")",
      ]),
  )


  native.new_http_archive(
      name = "io_angular_clutz",
      build_file = "//third_party:clutz.BUILD",
      sha256 = "2981de41d1ff4774b544423da9a2cd8beb3be649e95aef2ef2fd83957300b3fe",
      strip_prefix = "clutz-b0db5ade9bb535d387f05292316c422790c9848e",
      urls = [
          "http://mirror.bazel.build/github.com/angular/clutz/archive/b0db5ade9bb535d387f05292316c422790c9848e.tar.gz",  # 2017-05-22
          "https://github.com/angular/clutz/archive/b0db5ade9bb535d387f05292316c422790c9848e.tar.gz",
      ],
  )

  filegroup_external(
      name = "com_google_javascript_closure_compiler_externs",
      licenses = ["notice"],  # Apache 2.0
      sha256_urls_extract = {
          "0f515a6ebfa138490b3c5ea9f3591ea1a7e4a930d3074f18b3eca86084ad9b66": [
              "http://mirror.bazel.build/github.com/google/closure-compiler/archive/b37e6000001b0a6bf4c0be49024ebda14a8711d9.tar.gz",  # 2017-06-02
              "https://github.com/google/closure-compiler/archive/b37e6000001b0a6bf4c0be49024ebda14a8711d9.tar.gz",
          ],
      },
      strip_prefix = {"b37e6000001b0a6bf4c0be49024ebda14a8711d9.tar.gz": "closure-compiler-b37e6000001b0a6bf4c0be49024ebda14a8711d9/externs"},
  )

  filegroup_external(
      name = "com_google_javascript_closure_compiler_externs_polymer",
      licenses = ["notice"],  # Apache 2.0
      sha256_urls = {
          "23baad9a200a717a821c6df504c84d3a893d7ea9102b14876eb80097e3b94292": [
              "http://mirror.bazel.build/raw.githubusercontent.com/google/closure-compiler/0e8dc5597a295ee259e3fecd98d6535dc621232f/contrib/externs/polymer-1.0.js",  # 2017-05-27
              "https://raw.githubusercontent.com/google/closure-compiler/0e8dc5597a295ee259e3fecd98d6535dc621232f/contrib/externs/polymer-1.0.js",
          ],
      },
  )

  filegroup_external(
      name = "org_threejs",
      # no @license header
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "7aff264bd84c90bed3c72a4dc31db8c19151853c6df6980f52b01d3e9872c82d": [
              "http://mirror.bazel.build/raw.githubusercontent.com/mrdoob/three.js/ad419d40bdaab80abbb34b8f359b4ee840033a02/build/three.js",
              "https://raw.githubusercontent.com/mrdoob/three.js/ad419d40bdaab80abbb34b8f359b4ee840033a02/build/three.js",
          ],
          "0e98ded15bb7fe398a655667e76b39909d36c0973a8950d01c62f65f93161c27": [
              "http://mirror.bazel.build/raw.githubusercontent.com/mrdoob/three.js/ad419d40bdaab80abbb34b8f359b4ee840033a02/examples/js/controls/OrbitControls.js",
              "https://raw.githubusercontent.com/mrdoob/three.js/ad419d40bdaab80abbb34b8f359b4ee840033a02/examples/js/controls/OrbitControls.js",
          ],
      },
  )
  
  ##############################################################################
  # TensorBoard JavaScript Production Dependencies
  web_library_external(
      name = "com_lodash",
      licenses = ["notice"],  # MIT
      sha256 = "0e88207e5f90af4ce8790d6e1e7d09d2702d81bce0bafdc253d18c0a5bf7661e",
      urls = [
          "http://mirror.bazel.build/github.com/lodash/lodash/archive/3.10.1.tar.gz",
          "https://github.com/lodash/lodash/archive/3.10.1.tar.gz",
      ],
      strip_prefix = "lodash-3.10.1",
      path = "/lodash",
      srcs = ["lodash.js"],
  )

  filegroup_external(
      name = "com_numericjs",
      # no @license header
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "0e94aada97f12dee6118064add9170484c55022f5d53206ee4407143cd36ddcd": [
              "http://mirror.bazel.build/raw.githubusercontent.com/sloisel/numeric/v1.2.6/license.txt",
              "https://raw.githubusercontent.com/sloisel/numeric/v1.2.6/license.txt",
          ],
          "dfaca3b8485bee735788cc6eebca82ea25719adc1fb8911c7799c6bd5a95df3b": [
              "http://mirror.bazel.build/raw.githubusercontent.com/sloisel/numeric/v1.2.6/src/numeric.js",
              "https://raw.githubusercontent.com/sloisel/numeric/v1.2.6/src/numeric.js",
          ],
      },
  )

  filegroup_external(
      name = "com_palantir_plottable",
      # no @license header
      licenses = ["notice"],  # MIT
      sha256_urls_extract = {
          # Plottable doesn't have a release tarball on GitHub. Using the
          # sources directly from git also requires running Node tooling
          # beforehand to generate files. NPM is the only place to get it.
          "e3159beb279391c47433789f22b32bac88488cfcad6c0b6ec8605ce6b0081b0d": [
              "http://mirror.bazel.build/registry.npmjs.org/plottable/-/plottable-3.1.0.tgz",
              "https://registry.npmjs.org/plottable/-/plottable-3.1.0.tgz",
          ],
      },
  )

  filegroup_external(
      name = "io_github_cpettitt_dagre",
      # no @license header
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "6a349742a6cb219d5a2fc8d0844f6d89a6efc62e20c664450d884fc7ff2d6015": [
              "http://mirror.bazel.build/raw.githubusercontent.com/cpettitt/dagre/v0.7.4/LICENSE",
              "https://raw.githubusercontent.com/cpettitt/dagre/v0.7.4/LICENSE",
          ],
          "7323829ddd77924a69e2b1235ded3eac30acd990da0f037e0fbd3c8e9035b50d": [
              "http://mirror.bazel.build/raw.githubusercontent.com/cpettitt/dagre/v0.7.4/dist/dagre.core.js",
              "https://raw.githubusercontent.com/cpettitt/dagre/v0.7.4/dist/dagre.core.js",
          ],
      },
  )

  filegroup_external(
      name = "io_github_cpettitt_graphlib",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "6a349742a6cb219d5a2fc8d0844f6d89a6efc62e20c664450d884fc7ff2d6015": [
              "http://mirror.bazel.build/raw.githubusercontent.com/cpettitt/graphlib/v1.0.7/LICENSE",
              "https://raw.githubusercontent.com/cpettitt/graphlib/v1.0.7/LICENSE",
          ],
          "772045d412b1513b549be991c2e1846c38019429d43974efcae943fbe83489bf": [
              "http://mirror.bazel.build/raw.githubusercontent.com/cpettitt/graphlib/v1.0.7/dist/graphlib.core.js",
              "https://raw.githubusercontent.com/cpettitt/graphlib/v1.0.7/dist/graphlib.core.js",
          ],
      },
  )

  filegroup_external(
      name = "io_github_waylonflinn_weblas",
      # no @license header
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "633f2861a9a862b9cd7967e841e14dd3527912f209d6563595774fa31e3d84cb": [
              "http://mirror.bazel.build/raw.githubusercontent.com/waylonflinn/weblas/v0.9.0/LICENSES",
              "https://raw.githubusercontent.com/waylonflinn/weblas/v0.9.0/LICENSE",
          ],
          "f138fce57f673ca8a633f4aee5ae5b6fcb6ad0de59069a42a74e996fd04d8fcc": [
              "http://mirror.bazel.build/raw.githubusercontent.com/waylonflinn/weblas/v0.9.0/dist/weblas.js",
              "https://raw.githubusercontent.com/waylonflinn/weblas/v0.9.0/dist/weblas.js",
          ],
      },
  )

  filegroup_external(
      name = "org_d3js",
      # no @license header
      licenses = ["notice"],  # BSD-3-Clause
      sha256_urls_extract = {
          "b5fac5b296bc196e6aa7b59f9e33986fc44d23d59a0e211705187be9e35b943d": [
              "http://mirror.bazel.build/github.com/d3/d3/releases/download/v4.8.0/d3.zip",
              "https://github.com/d3/d3/releases/download/v4.8.0/d3.zip",
          ],
      },
      # TODO(jart): Use srcs=["d3.js"] instead of this once supported.
      generated_rule_name = "all_files",
      extra_build_file_content = "\n".join([
          "filegroup(",
          "    name = \"org_d3js\",",
          "    srcs = [\"d3.js\"],",
          ")",
      ]),
  )

  filegroup_external(
      name = "org_chromium_catapult_vulcanized_trace_viewer",
      licenses = ["notice"],  # BSD-3-Clause
      sha256_urls = {
          "f0df289ba9d03d857ad1c2f5918861376b1510b71588ffc60eff5c7a7bfedb09": [
              "http://mirror.bazel.build/raw.githubusercontent.com/catapult-project/catapult/2f7ee994984f3ebd3dd3dc3e05777bf180ec2ee8/LICENSE",
              "https://raw.githubusercontent.com/catapult-project/catapult/2f7ee994984f3ebd3dd3dc3e05777bf180ec2ee8/LICENSE",
          ],
          "9e99e79439ea5a1471bd4dd325bd6733e133bcb3da4df4b878ed6d2aec7c8d86": [
              "http://mirror.bazel.build/raw.githubusercontent.com/catapult-project/catapult/2f7ee994984f3ebd3dd3dc3e05777bf180ec2ee8/trace_viewer_full.html",
              "https://raw.githubusercontent.com/catapult-project/catapult/2f7ee994984f3ebd3dd3dc3e05777bf180ec2ee8/trace_viewer_full.html"
          ],
      },
  )

  ##############################################################################
  # TensorBoard Testing Dependencies
  web_library_external(
      name = "org_npmjs_registry_accessibility_developer_tools",
      licenses = ["notice"],  # Apache License 2.0
      sha256 = "1d6a72f401c9d53f68238c617dd43a05cd85ca5aa2e676a5b3c352711448e093",
      urls = [
          "http://mirror.bazel.build/registry.npmjs.org/accessibility-developer-tools/-/accessibility-developer-tools-2.10.0.tgz",
          "https://registry.npmjs.org/accessibility-developer-tools/-/accessibility-developer-tools-2.10.0.tgz",
      ],
      strip_prefix = "package",
      path = "/accessibility-developer-tools",
      suppress = ["strictDependencies"],
  )

  web_library_external(
      name = "org_npmjs_registry_async",
      licenses = ["notice"],  # MIT
      sha256 = "08655255ae810bf4d1cb1642df57658fcce823776d3ba8f4b46f4bbff6c87ece",
      urls = [
          "http://mirror.bazel.build/registry.npmjs.org/async/-/async-1.5.0.tgz",
          "https://registry.npmjs.org/async/-/async-1.5.0.tgz",
      ],
      strip_prefix = "package",
      path = "/async",
  )

  web_library_external(
      name = "org_npmjs_registry_chai",
      licenses = ["notice"],  # MIT
      sha256 = "aca8137bed5bb295bd7173325b7ad604cd2aeb341d739232b4f9f0b26745be90",
      urls = [
          "http://mirror.bazel.build/registry.npmjs.org/chai/-/chai-3.5.0.tgz",
          "https://registry.npmjs.org/chai/-/chai-3.5.0.tgz",
      ],
      strip_prefix = "package",
      path = "/chai",
  )

  web_library_external(
      name = "org_npmjs_registry_mocha",
      licenses = ["notice"],  # MIT
      sha256 = "13ef37a071196a2fba680799b906555d3f0ab61e80a7e8f73f93e77914590dd4",
      urls = [
          "http://mirror.bazel.build/registry.npmjs.org/mocha/-/mocha-2.5.3.tgz",
          "https://registry.npmjs.org/mocha/-/mocha-2.5.3.tgz",
      ],
      suppress = ["strictDependencies"],
      strip_prefix = "package",
      path = "/mocha",
  )

  web_library_external(
      name = "org_npmjs_registry_sinon",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "49edb057695fc9019aae992bf7e677a07de7c6ce2bf9f9facde4a245045d1532",
      urls = [
          "http://mirror.bazel.build/registry.npmjs.org/sinon/-/sinon-1.17.4.tgz",
          "https://registry.npmjs.org/sinon/-/sinon-1.17.4.tgz",
      ],
      strip_prefix = "package/lib",
      path = "/sinonjs",
  )

  web_library_external(
      name = "org_npmjs_registry_sinon_chai",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "b85fc56f713832960b56fe9269ee4bb2cd41edd2ceb130b0936e5bdbed5dea63",
      urls = [
          "http://mirror.bazel.build/registry.npmjs.org/sinon-chai/-/sinon-chai-2.8.0.tgz",
          "https://registry.npmjs.org/sinon-chai/-/sinon-chai-2.8.0.tgz",
      ],
      strip_prefix = "package",
      path = "/sinon-chai",
  )

  web_library_external(
      name = "org_npmjs_registry_stacky",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "c659e60f7957d9d80c23a7aacc4d71b19c6421a08f91174c0062de369595acae",
      urls = [
          "http://mirror.bazel.build/registry.npmjs.org/stacky/-/stacky-1.3.1.tgz",
          "https://registry.npmjs.org/stacky/-/stacky-1.3.1.tgz",
      ],
      strip_prefix = "package",
      path = "/stacky",
  )

  web_library_external(
      name = "org_npmjs_registry_web_component_tester",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "9d4ebd4945df8a936916d4d32b7f280f2a3afa35f79e7ca8ad3ed0a42770c537",
      urls = [
          "http://mirror.bazel.build/registry.npmjs.org/web-component-tester/-/web-component-tester-4.3.6.tgz",
          "https://registry.npmjs.org/web-component-tester/-/web-component-tester-4.3.6.tgz",
      ],
      strip_prefix = "package",
      path = "/web-component-tester",
      suppress = [
          "absolutePaths",
          "strictDependencies",
      ],
      deps = [
          "@com_lodash",
          "@org_npmjs_registry_accessibility_developer_tools",
          "@org_npmjs_registry_async",
          "@org_npmjs_registry_chai",
          "@org_npmjs_registry_mocha",
          "@org_npmjs_registry_sinon",
          "@org_npmjs_registry_sinon_chai",
          "@org_npmjs_registry_stacky",
          "@org_polymer_test_fixture",
      ],
  )

  web_library_external(
      name = "org_polymer_test_fixture",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "59d6cfb1187733b71275becfea181fe0aa1f734df5ff77f5850c806bbbf9a0d9",
      strip_prefix = "test-fixture-2.0.1",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/test-fixture/archive/v2.0.1.tar.gz",
          "https://github.com/PolymerElements/test-fixture/archive/v2.0.1.tar.gz",
      ],
      path = "/test-fixture",
      exclude = ["test/**"],
  )

