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
          "177293828c7a206bf2a7f725753d51396d38668311aa37c96445f91bbf8128a7": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/6e2f2280ef16ef277049d0ce8583af167d586c59/d3/d3.d.ts",  # v3
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/6e2f2280ef16ef277049d0ce8583af167d586c59/d3/d3.d.ts",  # v3
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
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/a872802c0c84ba98ff207d5e673a1fa867c67fd6/polymer/polymer.d.ts",
          ],
          "9453c3e6bae824e90758c3b38975c1ed77e6abd79bf513bcb08368fcdb14898e": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/f5407eba29c04fb8387c86df27512bd055b195d2/threejs/three.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/f5407eba29c04fb8387c86df27512bd055b195d2/threejs/three.d.ts",
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
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-array/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-array/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_axis",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "95f75c8dcc89850b2e72581d96a7b5f46ea4ac852f828893f141f14a597421f9": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-axis/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-axis/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_brush",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "a2738e693ce8a8640c2d29001e77582c9c361fd23bda44db471629866b60ada7": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-brush/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-brush/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_chord",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "c54d24756eb6d744b31e538ad9bab3a75f6d54e2288b29cc72338d4a057d3e83": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-chord/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-chord/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_collection",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "f987667167b1d2970911247e325eb1c37ca0823646f81ccec837ae59039822f7": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-collection/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-collection/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_color",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "9580c81f38ddcce7be0ac9bd3d0d083adebc34e17441709f90b9e4dcd1c19a56": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-color/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-color/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_dispatch",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "169f80b4cceca8e2e9ed384d81a5db0624cc01a26451dfb5a7e0cec6ea9cfb06": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-dispatch/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-dispatch/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_drag",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "08d35d139dde58c2722be98d718d01204fd6167d310f09b379e832f3c741489d": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-drag/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-drag/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_dsv",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "62594d00cf9e4bb895339c8e56f64330e202a5eb2a0fa580a1f6e6336f2c93ce": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-dsv/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-dsv/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_ease",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "d1cf8f99b7bf758c2ba3c0a4ce553e151d4d9b4cf45a6e8bd0edec7ce90f725b": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-ease/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-ease/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_force",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "288421e2008668d2076a4684657dd3d29b992832ef02c552981eb94a91042553": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-force/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-force/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_format",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "b42cb17e580c1fd0b64d478f7bd80ca806efaefda24426a833cf1f30a7275bca": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-format/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-format/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_hierarchy",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "a5683f5835d8716c6b89c075235078438cfab5897023ed720bfa492e244e969e": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-hierarchy/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-hierarchy/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_interpolate",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "590a71b741323ac3139b333ec8b743e24717fdd5b32bcff48ee521162a9dfe1c": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-interpolate/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-interpolate/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_path",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "96f35ba041bcaa265e2b373ee675177410d44d31c980e4f7fbeefd4bcba15b00": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-path/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-path/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_polygon",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "ce453451e8105cac6a4f4a4263ca2142ebb4bf442e342f470a81da691f220fcb": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-polygon/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-polygon/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_quadtree",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "238e278f1be5d6985a19800800cffee80f81199f71d848e3bbc288d1791a6f90": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-quadtree/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-quadtree/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_queue",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "e6ae19aad83495475653578de64fb9d6bf9764eda6c84d70f7935ec84bcc482e": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-queue/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-queue/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_random",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "d31b92ed86c23ec0a4776f99fa81ff033c95b96c8304d8aa9baf3b94af779aa8": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-random/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-random/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_request",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "44bb7b07d977028e6567540a3303b06fc9b33fb0960bc75c520e0733c840d89f": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-request/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-request/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_scale",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "02ce7c644ba34bd1abb84da2e832f248b048b6a23812be4365bd837f186c9f1f": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-scale/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-scale/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_selection",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "699043ddb28dfa5e46d87bc6a24cfc6d604237f298259d3fb3c7066e05e8c86e": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-selection/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-selection/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_shape",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "62668a7aaaf6232762b544f9f89c0f557ca7cfb0cd343a358dda7ecbe26f5739": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-shape/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-shape/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_time",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "0502490ce682fd9265fb1d5d693ce6cd82e3b05e5f5ee3433731266ecb03d5fc": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-time/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-time/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_timer",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "6f191f9aea704aa64b1defa40dfdff1447a6e6bb815feff1660f894500a9c94d": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-timer/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-timer/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_transition",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "a0a7c0c9bfb5c7d6d9d22a8d16b4484b66d13f2ed226954037546cb3da4098ba": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-transition/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-transition/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_voronoi",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "c6bd5f229f915151d0ef678fe50b1aa6a62334ea0a8c6fc0effbac9f7032efc7": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-voronoi/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-voronoi/index.d.ts",
          ],
      },
  )
  
  filegroup_external(
      name = "org_definitelytyped_types_d3_zoom",
      licenses = ["notice"],  # MIT
      sha256_urls = {
          "a25dc17fbd304cf7a0e5e7bbb8339c930d464eb40c4d6e5f839ce9c0191f4110": [
              "http://mirror.bazel.build/raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-zoom/index.d.ts",
              "https://raw.githubusercontent.com/DefinitelyTyped/DefinitelyTyped/1550dfd1b8e38d9bf104b3fd16ea9bf98a2b358e/types/d3-zoom/index.d.ts",
          ],
      },
  )
