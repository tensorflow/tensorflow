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

# TensorBoard typing dependencies

load("@io_bazel_rules_closure//closure:defs.bzl", "filegroup_external")

def tensorboard_typings_workspace():
  filegroup_external(
      name = "org_definitelytyped",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "b7da645f6e5555feb7aeede73775da0023ce2257df9c8e76c9159266035a9c0d": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/ebc69904eb78f94030d5d517b42db20867f679c0/chai/chai.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/ebc69904eb78f94030d5d517b42db20867f679c0/chai/chai.d.ts",
          ],
          "e4cd3d5de0eb3bc7b1063b50d336764a0ac82a658b39b5cf90511f489ffdee60": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/efd40e67ff323f7147651bdbef03c03ead7b1675/lodash/lodash.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/efd40e67ff323f7147651bdbef03c03ead7b1675/lodash/lodash.d.ts",
          ],
          "695a03dd2ccb238161d97160b239ab841562710e5c4e42886aefd4ace2ce152e": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/ebc69904eb78f94030d5d517b42db20867f679c0/mocha/mocha.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/ebc69904eb78f94030d5d517b42db20867f679c0/mocha/mocha.d.ts",
          ],
          "513ccd9ee1c708881120eeacd56788fc3b3da8e5c6172b20324cebbe858803fe": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/708609e0764daeb5eb64104af7aca50c520c4e6e/sinon/sinon.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/708609e0764daeb5eb64104af7aca50c520c4e6e/sinon/sinon.d.ts",
          ],
          "44eba36339bd1c0792072b7b204ee926fe5ffe1e9e2da916e67ac55548e3668a": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/a872802c0c84ba98ff207d5e673a1fa867c67fd6/polymer/polymer.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/a872802c0c84ba98ff207d5e673a1fa867c67fd6/polymer/polymer.d.ts",  # 2016-09-22
          ],
          "7ce67447146eb2b9e9cdaaf8bf45b3209865378022cc8acf86616d3be84f6481": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/8cb9ee3fdfe352cfef672bdfdb5f9c428f915e9f/threejs/three.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/8cb9ee3fdfe352cfef672bdfdb5f9c428f915e9f/threejs/three.d.ts",  # r74 @ 2016-04-06
          ],
          "691756a6eb455f340c9e834de0d49fff269e7b8c1799c2454465dcd6a4435b80": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/46719185c564694c5583c4b7ad94dbb786ecad46/webcomponents.js/webcomponents.js.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/46719185c564694c5583c4b7ad94dbb786ecad46/webcomponents.js/webcomponents.js.d.ts",
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_array",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "61e7abb7b1f01fbcb0cab8cf39003392f422566209edd681fbd070eaa84ca000": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-array/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-array/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_axis",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "95f75c8dcc89850b2e72581d96a7b5f46ea4ac852f828893f141f14a597421f9": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-axis/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-axis/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_brush",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "a2738e693ce8a8640c2d29001e77582c9c361fd23bda44db471629866b60ada7": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-brush/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-brush/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_chord",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "c54d24756eb6d744b31e538ad9bab3a75f6d54e2288b29cc72338d4a057d3e83": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-chord/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-chord/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_collection",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "39e8599a768f45f80aa70ca3032f026111da50d409c7e39a2ef091667cc343d9": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-collection/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-collection/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_color",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "6dd19edd11276476c5d535279237d1a009c1a733611cc44621a88fda1ca04377": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-color/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-color/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_dispatch",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "af1474301e594fcb4bbdb134361fb6d26c7b333386c3213821532acde59e61a3": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-dispatch/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-dispatch/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_drag",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "2f8248ae2bf33fb1d61bb1ea4271cb4bacfd9a9939dc8d7bde7ec8b66d4441ed": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-drag/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-drag/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_dsv",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "62594d00cf9e4bb895339c8e56f64330e202a5eb2a0fa580a1f6e6336f2c93ce": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-dsv/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-dsv/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_ease",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "d5a9be5316b2d1823a3faa7f75de1e2c2efda5c75f0631b44a0f7b69e11f3a90": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-ease/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-ease/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_force",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "288421e2008668d2076a4684657dd3d29b992832ef02c552981eb94a91042553": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-force/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-force/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_format",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "b42cb17e580c1fd0b64d478f7bd80ca806efaefda24426a833cf1f30a7275bca": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-format/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-format/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_hierarchy",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "a5683f5835d8716c6b89c075235078438cfab5897023ed720bfa492e244e969e": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-hierarchy/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-hierarchy/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_interpolate",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "effeefea9ac02539def43d7b9aa2f39e8672c03aac9b407a61b09563ff141fad": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-interpolate/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-interpolate/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_path",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "deea4ab3654925d365dd1ffab69a2140808c6173e7f23c461ded2852c309eb9c": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-path/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-path/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_polygon",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "ec7a42affe79c87066f14173fcbc8d8b5747f54bfbe0e60111e2786ee4d227bf": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-polygon/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-polygon/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_quadtree",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "2908631a7da3bfb0096e3b89f464b45390bbb31ec798d1b6c0898ff82e344560": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-quadtree/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-quadtree/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_queue",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "4fc0503e3558d136b855335f36ea8984937ab63a2a28b8c7b293d35825388615": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-queue/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-queue/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_random",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "5130e803ba26d2dc931ddd0fa574b5abbb0fc4486e7975f97a83c01630763676": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-random/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-random/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_request",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "fc2b7c2c05498011eb039825aab76a7916698fb3e7133e278fc92ae529ae99f0": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-request/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-request/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_scale",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "ff3e2d2033a37d698c3bd2896ffd9dd4ceab1903d96aa90d388a6a2d14d8ee05": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-scale/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-scale/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_selection",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "47fae7c4bc425101490daae067727b74ee09e6c830331a4cf333cdb532a5d108": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-selection/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-selection/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_shape",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "7fec580ba54bc29417dc9030bb3731c9756a65c5e57dcce5a4f183fff7180cd8": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-shape/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-shape/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_time",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "4b68f2a4ee428f21f2e7d706c0a64f628f0ff5f130cd9f023ab23a04a8fe31de": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-time/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-time/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_timer",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "a196f42560be9fa1a77d473c0180f9f2f8d570ed0eee616aad0da94d90ef3661": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-timer/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-timer/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_transition",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "10c6cf259d6f965014e75a63925f302911c5afb8581d6d63b0597544fe104bd7": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-transition/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-transition/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_voronoi",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "411482515e2ccda4659f7b3d2fbd3a7ef5ea2c7053eec62c95a174b68ad60c3d": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-voronoi/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-voronoi/index.d.ts",  # 2017-06-08
          ],
      },
  )

  filegroup_external(
      name = "org_definitelytyped_types_d3_zoom",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "df0bedbb7711366a43418d6a3b47c4688ccb02a3d8ad0c2468cafcb6c2faa346": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-zoom/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/dc27c3788c00d279ae5ff61e8e2dfd568aae5e8e/types/d3-zoom/index.d.ts",  # 2017-06-08
          ],
      },
  )
