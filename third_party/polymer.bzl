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

# TensorBoard Polymer Dependencies

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library_external")

def tensorboard_polymer_workspace():
  web_library_external(
      name = "org_polymer",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "2c38cf73bdd09e0f80a4b7210fd58f88a6bdd8624da73fb3e028e66a84c9e095",
      strip_prefix = "polymer-1.8.1",
      urls = [
          "http://mirror.bazel.build/github.com/polymer/polymer/archive/v1.8.1.tar.gz",
          "https://github.com/polymer/polymer/archive/v1.8.1.tar.gz",
      ],
      path = "/polymer",
      srcs = [
          "polymer.html",
          "polymer-micro.html",
          "polymer-mini.html",
      ],
  )

  web_library_external(
      name = "org_polymer_font_roboto",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "fae51429b56a4a4c15f1f0c23b733c7095940cc9c04c275fa7adb3bf055b23b3",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/font-roboto/archive/v1.0.1.tar.gz",
          "https://github.com/PolymerElements/font-roboto/archive/v1.0.1.tar.gz",
      ],
      strip_prefix = "font-roboto-1.0.1",
      path = "/font-roboto",
      srcs = ["roboto.html"],
  )

  web_library_external(
      name = "org_polymer_hydrolysis",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "703b50f6b00f9e0546b5a3451da57bb20f77a166e27e4967923b9e835bab9b80",
      urls = [
          "http://mirror.bazel.build/github.com/Polymer/polymer-analyzer/archive/v1.19.3.tar.gz",
          "https://github.com/Polymer/polymer-analyzer/archive/v1.19.3.tar.gz",
      ],
      strip_prefix = "polymer-analyzer-1.19.3",
      path = "/hydrolysis",
      srcs = [
          "hydrolysis-analyzer.html",
          "hydrolysis.html",
          "hydrolysis.js",
      ],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_a11y_announcer",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "53114ceb57d9f33a7a8058488cf06450e48502e5d033adf51c91330f61620353",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-a11y-announcer/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/iron-a11y-announcer/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "iron-a11y-announcer-2.0.0",
      path = "/iron-a11y-announcer",
      srcs = ["iron-a11y-announcer.html"],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_a11y_keys_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "09274155c8d537f8bb567b3be5e747253ef760995a59ee06cb0ab38e704212fb",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-a11y-keys-behavior/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/iron-a11y-keys-behavior/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "iron-a11y-keys-behavior-2.0.0",
      path = "/iron-a11y-keys-behavior",
      srcs = ["iron-a11y-keys-behavior.html"],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_ajax",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "9162d8af4611e911ac3ebbfc08bb7038ac04f6e79a9287b1476fe36ad6770bc5",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-ajax/archive/v1.2.0.tar.gz",
          "https://github.com/PolymerElements/iron-ajax/archive/v1.2.0.tar.gz",
      ],
      strip_prefix = "iron-ajax-1.2.0",
      path = "/iron-ajax",
      srcs = [
          "iron-ajax.html",
          "iron-request.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_promise_polyfill",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_autogrow_textarea",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "50bbb901d2c8f87462e3552e3d671a552faa12c37c485e548d7a234ebffbc427",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-autogrow-textarea/archive/v1.0.12.tar.gz",
          "https://github.com/PolymerElements/iron-autogrow-textarea/archive/v1.0.12.tar.gz",
      ],
      strip_prefix = "iron-autogrow-textarea-1.0.12",
      path = "/iron-autogrow-textarea",
      srcs = ["iron-autogrow-textarea.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_behaviors",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_form_element_behavior",
          "@org_polymer_iron_validatable_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_behaviors",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "a1e8d4b7a13f3d36beba9c2a6b186ed33a53e6af2e79f98c1fcc7e85e7b53f89",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-behaviors/archive/v1.0.17.tar.gz",
          "https://github.com/PolymerElements/iron-behaviors/archive/v1.0.17.tar.gz",
      ],
      strip_prefix = "iron-behaviors-1.0.17",
      path = "/iron-behaviors",
      srcs = [
          "iron-button-state.html",
          "iron-control-state.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_checked_element_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "539a0e1c4df0bc702d3bd342388e4e56c77ec4c2066cce69e41426a69f92e8bd",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-checked-element-behavior/archive/v1.0.4.tar.gz",
          "https://github.com/PolymerElements/iron-checked-element-behavior/archive/v1.0.4.tar.gz",
      ],
      strip_prefix = "iron-checked-element-behavior-1.0.4",
      path = "/iron-checked-element-behavior",
      srcs = ["iron-checked-element-behavior.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_form_element_behavior",
          "@org_polymer_iron_validatable_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_component_page",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "3636e8b9a1f229fc33b5aad3933bd02a9825f66e679a0be31855d7c8245c4b4b",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-component-page/archive/v1.1.4.tar.gz",
          "https://github.com/PolymerElements/iron-component-page/archive/v1.1.4.tar.gz",
      ],
      strip_prefix = "iron-component-page-1.1.4",
      path = "/iron-component-page",
      srcs = ["iron-component-page.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_hydrolysis",
          "@org_polymer_iron_ajax",
          "@org_polymer_iron_doc_viewer",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_icons",
          "@org_polymer_iron_selector",
          "@org_polymer_paper_header_panel",
          "@org_polymer_paper_styles",
          "@org_polymer_paper_toolbar",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_collapse",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "eb72f459a2a5adbcd922327eea02ed909e8056ad72fd8a32d04a14ce54b2e480",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-collapse/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/iron-collapse/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "iron-collapse-2.0.0",
      path = "/iron-collapse",
      srcs = ["iron-collapse.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_resizable_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_demo_helpers",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "e196985cb7e50108283c3fc189a3a018e697b4648107d597df71a5c17f5ee907",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-demo-helpers/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/iron-demo-helpers/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "iron-demo-helpers-2.0.0",
      path = "/iron-demo-helpers",
      srcs = [
          "demo-pages-shared-styles.html",
          "demo-snippet.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_font_roboto",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_icons",
          "@org_polymer_marked_element",
          "@org_polymer_paper_icon_button",
          "@org_polymer_paper_styles",
          "@org_polymer_prism_element",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_doc_viewer",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "5d487c99dd0cf626c800ae8667b0c8c88095f4482a68e837a1d3f58484ca8fb4",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-doc-viewer/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/iron-doc-viewer/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "iron-doc-viewer-2.0.0",
      path = "/iron-doc-viewer",
      srcs = [
          "iron-doc-property-styles.html",
          "iron-doc-property.html",
          "iron-doc-viewer-styles.html",
          "iron-doc-viewer.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_marked_element",
          "@org_polymer_paper_button",
          "@org_polymer_paper_styles",
          "@org_polymer_prism_element",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_dropdown",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "db9d6598157f8b114f1be1e6ed1c74917d4c37e660b2dda1e31f6873f1a33b80",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-dropdown/archive/v1.5.5.tar.gz",
          "https://github.com/PolymerElements/iron-dropdown/archive/v1.5.5.tar.gz",
      ],
      strip_prefix = "iron-dropdown-1.5.5",
      path = "/iron-dropdown",
      srcs = [
          "iron-dropdown.html",
          "iron-dropdown-scroll-manager.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
          "@org_polymer_iron_behaviors",
          "@org_polymer_iron_overlay_behavior",
          "@org_polymer_iron_resizable_behavior",
          "@org_polymer_neon_animation",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_fit_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "10132a2ea309a37c4c07b8fead71f64abc588ee6107931e34680f5f36dd8291e",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-fit-behavior/archive/v1.2.5.tar.gz",
          "https://github.com/PolymerElements/iron-fit-behavior/archive/v1.2.5.tar.gz",
      ],
      strip_prefix = "iron-fit-behavior-1.2.5",
      path = "/iron-fit-behavior",
      srcs = ["iron-fit-behavior.html"],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_flex_layout",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "79287f6ca1c2d4e003f68b88fe19d03a1b6a0011e2b4cae579fe4d1474163a2e",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-flex-layout/archive/v1.3.0.tar.gz",
          "https://github.com/PolymerElements/iron-flex-layout/archive/v1.3.0.tar.gz",
      ],
      strip_prefix = "iron-flex-layout-1.3.0",
      path = "/iron-flex-layout",
      srcs = [
          "classes/iron-flex-layout.html",
          "classes/iron-shadow-flex-layout.html",
          "iron-flex-layout.html",
          "iron-flex-layout-classes.html",
      ],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_form_element_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "1dd9371c638e5bc2ecba8a64074aa680dfb8712198e9612f9ed24d387efc8f26",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-form-element-behavior/archive/v1.0.6.tar.gz",
          "https://github.com/PolymerElements/iron-form-element-behavior/archive/v1.0.6.tar.gz",
      ],
      strip_prefix = "iron-form-element-behavior-1.0.6",
      path = "/iron-form-element-behavior",
      srcs = ["iron-form-element-behavior.html"],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_icon",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "9ed58a69159a02c07a6050d242e6d4e585a29f3245b8c8c390cfd52ddb786dc4",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-icon/archive/v1.0.11.tar.gz",
          "https://github.com/PolymerElements/iron-icon/archive/v1.0.11.tar.gz",
      ],
      strip_prefix = "iron-icon-1.0.11",
      path = "/iron-icon",
      srcs = ["iron-icon.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_meta",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_icons",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "3b18542c147c7923dc3a36b1a51984a73255d610f297d43c9aaccc52859bd0d0",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-icons/archive/v1.1.3.tar.gz",
          "https://github.com/PolymerElements/iron-icons/archive/v1.1.3.tar.gz",
      ],
      strip_prefix = "iron-icons-1.1.3",
      path = "/iron-icons",
      srcs = [
          "av-icons.html",
          "communication-icons.html",
          "device-icons.html",
          "editor-icons.html",
          "hardware-icons.html",
          "image-icons.html",
          "iron-icons.html",
          "maps-icons.html",
          "notification-icons.html",
          "places-icons.html",
          "social-icons.html",
      ],
      deps = [
          "@org_polymer_iron_icon",
          "@org_polymer_iron_iconset_svg",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_iconset_svg",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "75cfb41e78f86ef6cb5d201ad12021785ef9e192b490ad46dcc15a9c19bdf71a",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-iconset-svg/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/iron-iconset-svg/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "iron-iconset-svg-2.0.0",
      path = "/iron-iconset-svg",
      srcs = ["iron-iconset-svg.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_meta",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_input",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "c505101ead08ab25526b1f49baecc8c28b4221b92a65e7334c783bdc81553c36",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-input/archive/1.0.10.tar.gz",
          "https://github.com/PolymerElements/iron-input/archive/1.0.10.tar.gz",
      ],
      strip_prefix = "iron-input-1.0.10",
      path = "/iron-input",
      srcs = ["iron-input.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_announcer",
          "@org_polymer_iron_validatable_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_list",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "72a6530b9f0ad5557f5d287845792a0ada74d8b159198e27f940e226313dc116",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-list/archive/v1.3.9.tar.gz",
          "https://github.com/PolymerElements/iron-list/archive/v1.3.9.tar.gz",
      ],
      strip_prefix = "iron-list-1.3.9",
      path = "/iron-list",
      srcs = ["iron-list.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
          "@org_polymer_iron_resizable_behavior",
          "@org_polymer_iron_scroll_target_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_menu_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "35d33d1ae55c6efaa0c3744ebe8a06cc0a8b2af9286dd8d36e20726a8540a11a",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-menu-behavior/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/iron-menu-behavior/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "iron-menu-behavior-2.0.0",
      path = "/iron-menu-behavior",
      srcs = [
          "iron-menu-behavior.html",
          "iron-menubar-behavior.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
          "@org_polymer_iron_selector",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_meta",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "fb05e6031bae6b4effe5f15d44b3f548d5807f9e3b3aa2442ba17cf4b8b84361",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-meta/archive/v1.1.1.tar.gz",
          "https://github.com/PolymerElements/iron-meta/archive/v1.1.1.tar.gz",
      ],
      strip_prefix = "iron-meta-1.1.1",
      path = "/iron-meta",
      srcs = ["iron-meta.html"],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_overlay_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "3df5b54ff2e0510c87a2aff8c9d730d3fe83d3d11277cc1a49fa29b549acb46c",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-overlay-behavior/archive/v1.10.1.tar.gz",
          "https://github.com/PolymerElements/iron-overlay-behavior/archive/v1.10.1.tar.gz",
      ],
      strip_prefix = "iron-overlay-behavior-1.10.1",
      path = "/iron-overlay-behavior",
      srcs = [
          "iron-focusables-helper.html",
          "iron-overlay-backdrop.html",
          "iron-overlay-behavior.html",
          "iron-overlay-manager.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
          "@org_polymer_iron_fit_behavior",
          "@org_polymer_iron_resizable_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_iron_range_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "b2f2b6d52284542330bd30b586e217926eb0adec5e13934a3cef557717c22dc2",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-range-behavior/archive/v1.0.4.tar.gz",
          "https://github.com/PolymerElements/iron-range-behavior/archive/v1.0.4.tar.gz",
      ],
      strip_prefix = "iron-range-behavior-1.0.4",
      path = "/iron-range-behavior",
      srcs = ["iron-range-behavior.html"],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_resizable_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "a87a78ee9223c2f6afae7fc94a3ff91cbce6f7e2a7ed3f2979af7945c9281616",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-resizable-behavior/archive/v1.0.3.tar.gz",
          "https://github.com/PolymerElements/iron-resizable-behavior/archive/v1.0.3.tar.gz",
      ],
      strip_prefix = "iron-resizable-behavior-1.0.3",
      path = "/iron-resizable-behavior",
      srcs = ["iron-resizable-behavior.html"],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_scroll_target_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "7c6614c07d354375666ee96eea9f6d485dbf7898146a444ec26094a70d4e0afa",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-scroll-target-behavior/archive/v1.1.1.tar.gz",
          "https://github.com/PolymerElements/iron-scroll-target-behavior/archive/v1.1.1.tar.gz",
      ],
      strip_prefix = "iron-scroll-target-behavior-1.1.1",
      path = "/iron-scroll-target-behavior",
      srcs = ["iron-scroll-target-behavior.html"],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_selector",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "ba28a47443bad3b744611c9d7a79fb21dbdf2e35edc5ef8f812e2dcd72b16747",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-selector/archive/v1.5.2.tar.gz",
          "https://github.com/PolymerElements/iron-selector/archive/v1.5.2.tar.gz",
      ],
      strip_prefix = "iron-selector-1.5.2",
      path = "/iron-selector",
      srcs = [
          "iron-multi-selectable.html",
          "iron-selectable.html",
          "iron-selection.html",
          "iron-selector.html",
      ],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_iron_validatable_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "aef4901e68043824f36104799269573dd345ffaac494186e466fdc79c06fdb63",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/iron-validatable-behavior/archive/v1.1.1.tar.gz",
          "https://github.com/PolymerElements/iron-validatable-behavior/archive/v1.1.1.tar.gz",
      ],
      strip_prefix = "iron-validatable-behavior-1.1.1",
      path = "/iron-validatable-behavior",
      srcs = ["iron-validatable-behavior.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_meta",
      ],
  )

  web_library_external(
      name = "org_polymer_marked",
      licenses = ["notice"],  # MIT
      sha256 = "93d30bd593736ca440938d77808b7ef5972da0f3fcfe4ae63ae7b4ce117da2cb",
      urls = [
          "http://mirror.bazel.build/github.com/chjj/marked/archive/v0.3.2.zip",
          "https://github.com/chjj/marked/archive/v0.3.2.zip",
      ],
      strip_prefix = "marked-0.3.2",
      path = "/marked",
      srcs = ["lib/marked.js"],
  )

  web_library_external(
      name = "org_polymer_marked_element",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "2ac1f7fae0c1b656e671b2772492809714d836f27c9efa7f2c5fe077ee760f3c",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/marked-element/archive/v2.1.0.tar.gz",
          "https://github.com/PolymerElements/marked-element/archive/v2.1.0.tar.gz",
      ],
      strip_prefix = "marked-element-2.1.0",
      path = "/marked-element",
      srcs = [
          "marked-element.html",
          "marked-import.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_marked",
      ],
  )

  web_library_external(
      name = "org_polymer_neon_animation",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "8800c314a76b2da190a2b203259c1091f6d38e0057ed37c2a3d0b734980fa9a5",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/neon-animation/archive/v1.2.2.tar.gz",
          "https://github.com/PolymerElements/neon-animation/archive/v1.2.2.tar.gz",
      ],
      strip_prefix = "neon-animation-1.2.2",
      path = "/neon-animation",
      srcs = [
          "animations/cascaded-animation.html",
          "animations/fade-in-animation.html",
          "animations/fade-out-animation.html",
          "animations/hero-animation.html",
          "animations/opaque-animation.html",
          "animations/reverse-ripple-animation.html",
          "animations/ripple-animation.html",
          "animations/scale-down-animation.html",
          "animations/scale-up-animation.html",
          "animations/slide-down-animation.html",
          "animations/slide-from-bottom-animation.html",
          "animations/slide-from-left-animation.html",
          "animations/slide-from-right-animation.html",
          "animations/slide-from-top-animation.html",
          "animations/slide-left-animation.html",
          "animations/slide-right-animation.html",
          "animations/slide-up-animation.html",
          "animations/transform-animation.html",
          "neon-animatable.html",
          "neon-animatable-behavior.html",
          "neon-animated-pages.html",
          "neon-animation.html",
          "neon-animation-behavior.html",
          "neon-animation-runner-behavior.html",
          "neon-animations.html",
          "neon-shared-element-animatable-behavior.html",
          "neon-shared-element-animation-behavior.html",
          "web-animations.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_meta",
          "@org_polymer_iron_resizable_behavior",
          "@org_polymer_iron_selector",
          "@org_polymer_web_animations_js",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_behaviors",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "7cfcb9082ef9909da262df6b5c120bc62dbeaff278cb563e8fc60465ddd387e5",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-behaviors/archive/v1.0.12.tar.gz",
          "https://github.com/PolymerElements/paper-behaviors/archive/v1.0.12.tar.gz",
      ],
      strip_prefix = "paper-behaviors-1.0.12",
      path = "/paper-behaviors",
      srcs = [
          "paper-button-behavior.html",
          "paper-checked-element-behavior.html",
          "paper-inky-focus-behavior.html",
          "paper-ripple-behavior.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_behaviors",
          "@org_polymer_iron_checked_element_behavior",
          "@org_polymer_paper_ripple",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_button",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "896c0a7e34bfcce63fc23c63e105ed9c4d62fa3a6385b7161e1e5cd4058820a6",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-button/archive/v1.0.11.tar.gz",
          "https://github.com/PolymerElements/paper-button/archive/v1.0.11.tar.gz",
      ],
      strip_prefix = "paper-button-1.0.11",
      path = "/paper-button",
      srcs = ["paper-button.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_paper_behaviors",
          "@org_polymer_paper_material",
          "@org_polymer_paper_ripple",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_checkbox",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "0a291d0c64de1b6b807d66697bead9c66c0d7bc3c68b8037e6667f3d66a5904c",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-checkbox/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/paper-checkbox/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "paper-checkbox-2.0.0",
      path = "/paper-checkbox",
      srcs = ["paper-checkbox.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_paper_behaviors",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_dialog",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "cf60d3aa6ad57ba4eb8b1c16713e65057735eed94b009aeebdbcf3436c95a161",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-dialog/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/paper-dialog/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "paper-dialog-2.0.0",
      path = "/paper-dialog",
      srcs = ["paper-dialog.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_neon_animation",
          "@org_polymer_paper_dialog_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_dialog_behavior",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "d78e4f7d008c22537a9255ccda1e919fddae5cc125ef26a66eb2c47f648c20ab",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-dialog-behavior/archive/v1.2.7.tar.gz",
          "https://github.com/PolymerElements/paper-dialog-behavior/archive/v1.2.7.tar.gz",
      ],
      strip_prefix = "paper-dialog-behavior-1.2.7",
      path = "/paper-dialog-behavior",
      srcs = [
          "paper-dialog-behavior.html",
          "paper-dialog-common.css",
          "paper-dialog-shared-styles.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_overlay_behavior",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_dialog_scrollable",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "a2e69283e7674f782c44d811387a0f8da2d01fac0172743d1add65e253e6b5ff",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-dialog-scrollable/archive/1.1.5.tar.gz",
          "https://github.com/PolymerElements/paper-dialog-scrollable/archive/1.1.5.tar.gz",
      ],
      strip_prefix = "paper-dialog-scrollable-1.1.5",
      path = "/paper-dialog-scrollable",
      srcs = ["paper-dialog-scrollable.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_paper_dialog_behavior",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_dropdown_menu",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "9d88f654ec03ee9be211df9e69bede9e8a22b51bf1dbcc63b79762e4256d81ad",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-dropdown-menu/archive/v1.4.0.tar.gz",
          "https://github.com/PolymerElements/paper-dropdown-menu/archive/v1.4.0.tar.gz",
      ],
      strip_prefix = "paper-dropdown-menu-1.4.0",
      path = "/paper-dropdown-menu",
      srcs = [
          "paper-dropdown-menu.html",
          "paper-dropdown-menu-icons.html",
          "paper-dropdown-menu-light.html",
          "paper-dropdown-menu-shared-styles.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
          "@org_polymer_iron_behaviors",
          "@org_polymer_iron_form_element_behavior",
          "@org_polymer_iron_icon",
          "@org_polymer_iron_iconset_svg",
          "@org_polymer_iron_validatable_behavior",
          "@org_polymer_paper_behaviors",
          "@org_polymer_paper_input",
          "@org_polymer_paper_menu_button",
          "@org_polymer_paper_ripple",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_header_panel",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "0db4bd8a4bf6f20dcd0dffb4f907b31c93a8647c9c021344239cf30b40b87075",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-header-panel/archive/v1.1.4.tar.gz",
          "https://github.com/PolymerElements/paper-header-panel/archive/v1.1.4.tar.gz",
      ],
      strip_prefix = "paper-header-panel-1.1.4",
      path = "/paper-header-panel",
      srcs = ["paper-header-panel.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_flex_layout",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_icon_button",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "9cba5bcfd6aeb4c41581c1392c678cf2278d360e9d122f4d9db54a9ebb404496",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-icon-button/archive/v1.1.3.tar.gz",
          "https://github.com/PolymerElements/paper-icon-button/archive/v1.1.3.tar.gz",
      ],
      strip_prefix = "paper-icon-button-1.1.3",
      path = "/paper-icon-button",
      srcs = [
          "paper-icon-button.html",
          "paper-icon-button-light.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_icon",
          "@org_polymer_paper_behaviors",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_input",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "17c3dea9bb1c2026cc61324696c6c774214a0dc37686b91ca214a6af550994db",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-input/archive/v1.1.18.tar.gz",
          "https://github.com/PolymerElements/paper-input/archive/v1.1.18.tar.gz",
      ],
      strip_prefix = "paper-input-1.1.18",
      path = "/paper-input",
      srcs = [
          "paper-input.html",
          "paper-input-addon-behavior.html",
          "paper-input-behavior.html",
          "paper-input-char-counter.html",
          "paper-input-container.html",
          "paper-input-error.html",
          "paper-textarea.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
          "@org_polymer_iron_autogrow_textarea",
          "@org_polymer_iron_behaviors",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_form_element_behavior",
          "@org_polymer_iron_input",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_item",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "12ee0dcb61b0d5721c5988571f6974d7b2211e97724f4195893fbcc9058cdac8",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-item/archive/v1.1.4.tar.gz",
          "https://github.com/PolymerElements/paper-item/archive/v1.1.4.tar.gz",
      ],
      strip_prefix = "paper-item-1.1.4",
      path = "/paper-item",
      srcs = [
          "paper-icon-item.html",
          "paper-item.html",
          "paper-item-behavior.html",
          "paper-item-body.html",
          "paper-item-shared-styles.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_behaviors",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_listbox",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "674992d882b18a0618fa697180f196dbc052fb2f5d9ce4e19026a918b568ffd6",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-listbox/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/paper-listbox/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "paper-listbox-2.0.0",
      path = "/paper-listbox",
      srcs = ["paper-listbox.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_menu_behavior",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_material",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "913e9c63cf5c8286b0fab817079d7dc900a343d2c05809995d8d9ba0e41f8a29",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-material/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/paper-material/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "paper-material-2.0.0",
      path = "/paper-material",
      srcs = [
          "paper-material.html",
          "paper-material-shared-styles.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_menu",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "a3cee220926e315f7412236b3628288774694447c0da4428345f36d0f127ba3b",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-menu/archive/v1.2.2.tar.gz",
          "https://github.com/PolymerElements/paper-menu/archive/v1.2.2.tar.gz",
      ],
      strip_prefix = "paper-menu-1.2.2",
      path = "/paper-menu",
      srcs = [
          "paper-menu.html",
          "paper-menu-shared-styles.html",
          "paper-submenu.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_behaviors",
          "@org_polymer_iron_collapse",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_menu_behavior",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_menu_button",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "be3290c288a2bd4f9887213db22c75add99cc29ff4d088100c0bc4eb0e57997b",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-menu-button/archive/v1.5.1.tar.gz",
          "https://github.com/PolymerElements/paper-menu-button/archive/v1.5.1.tar.gz",
      ],
      strip_prefix = "paper-menu-button-1.5.1",
      path = "/paper-menu-button",
      srcs = [
          "paper-menu-button.html",
          "paper-menu-button-animations.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
          "@org_polymer_iron_behaviors",
          "@org_polymer_iron_dropdown",
          "@org_polymer_neon_animation",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_progress",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "2b6776b2f023c1f344feea17ba29b58d879e46f8ed43b7256495054b5183fff6",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-progress/archive/v1.0.9.tar.gz",
          "https://github.com/PolymerElements/paper-progress/archive/v1.0.9.tar.gz",
      ],
      strip_prefix = "paper-progress-1.0.9",
      path = "/paper-progress",
      srcs = ["paper-progress.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_range_behavior",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_radio_button",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "6e911d0c308aa388136b3af79d1bdcbe5a1f4159cbc79d71efb4ff3b6c0b4e91",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-radio-button/archive/v1.1.2.tar.gz",
          "https://github.com/PolymerElements/paper-radio-button/archive/v1.1.2.tar.gz",
      ],
      strip_prefix = "paper-radio-button-1.1.2",
      path = "/paper-radio-button",
      srcs = ["paper-radio-button.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_paper_behaviors",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_radio_group",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "7885ad1f81e9dcc03dcea4139b54a201ff55c18543770cd44f94530046c9e163",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-radio-group/archive/v1.0.9.tar.gz",
          "https://github.com/PolymerElements/paper-radio-group/archive/v1.0.9.tar.gz",
      ],
      strip_prefix = "paper-radio-group-1.0.9",
      path = "/paper-radio-group",
      srcs = ["paper-radio-group.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
          "@org_polymer_iron_selector",
          "@org_polymer_paper_radio_button",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_ripple",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "ba76bfb1c737260a8a103d3ca97faa1f7c3288c7db9b2519f401b7a782147c09",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-ripple/archive/v1.0.5.tar.gz",
          "https://github.com/PolymerElements/paper-ripple/archive/v1.0.5.tar.gz",
      ],
      strip_prefix = "paper-ripple-1.0.5",
      path = "/paper-ripple",
      srcs = ["paper-ripple.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_slider",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "8bb4db532e8b11b11f78006d9aefa217841392c1aeb449ee2a12e3b56748b774",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-slider/archive/v1.0.14.tar.gz",
          "https://github.com/PolymerElements/paper-slider/archive/v1.0.14.tar.gz",
      ],
      strip_prefix = "paper-slider-1.0.14",
      path = "/paper-slider",
      srcs = ["paper-slider.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_keys_behavior",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_form_element_behavior",
          "@org_polymer_iron_range_behavior",
          "@org_polymer_paper_behaviors",
          "@org_polymer_paper_input",
          "@org_polymer_paper_progress",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_spinner",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "6a752907fab7899cbeed15b478e7b9299047c15fbf9d1561d6eb4d204bdbd178",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-spinner/archive/v1.1.1.tar.gz",
          "https://github.com/PolymerElements/paper-spinner/archive/v1.1.1.tar.gz",
      ],
      strip_prefix = "paper-spinner-1.1.1",
      path = "/paper-spinner",
      srcs = [
          "paper-spinner.html", "paper-spinner-behavior.html",
          "paper-spinner-lite.html", "paper-spinner-styles.html"
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_styles",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "6d26b0a4c286402098853dc7388f6b22f30dfb7a74e47b34992ac03380144bb2",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-styles/archive/v1.1.4.tar.gz",
          "https://github.com/PolymerElements/paper-styles/archive/v1.1.4.tar.gz",
      ],
      strip_prefix = "paper-styles-1.1.4",
      path = "/paper-styles",
      srcs = [
          "classes/global.html",
          "classes/shadow.html",
          "classes/shadow-layout.html",
          "classes/typography.html",
          "color.html",
          "default-theme.html",
          "demo.css",
          "demo-pages.html",
          "paper-styles.html",
          "paper-styles-classes.html",
          "shadow.html",
          "typography.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_font_roboto",
          "@org_polymer_iron_flex_layout",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_tabs",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "c23b6a5221db35e5b1ed3eb8e8696b952572563e285adaec96aba1e3134db825",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-tabs/archive/v1.7.0.tar.gz",
          "https://github.com/PolymerElements/paper-tabs/archive/v1.7.0.tar.gz",
      ],
      strip_prefix = "paper-tabs-1.7.0",
      path = "/paper-tabs",
      srcs = [
          "paper-tab.html",
          "paper-tabs.html",
          "paper-tabs-icons.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_behaviors",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_iron_icon",
          "@org_polymer_iron_iconset_svg",
          "@org_polymer_iron_menu_behavior",
          "@org_polymer_iron_resizable_behavior",
          "@org_polymer_paper_behaviors",
          "@org_polymer_paper_icon_button",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_toast",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "b1c677e1681ef8d3f688a83da8f7b263902f757f395a9354a1c35f93b9125b60",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-toast/archive/v2.0.0.tar.gz",
          "https://github.com/PolymerElements/paper-toast/archive/v2.0.0.tar.gz",
      ],
      strip_prefix = "paper-toast-2.0.0",
      path = "/paper-toast",
      srcs = ["paper-toast.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_a11y_announcer",
          "@org_polymer_iron_overlay_behavior",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_toggle_button",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "4aa7cf0396fa2994a8bc2ac6e8428f48b07b945bb7c41bd52041ef5827b45de3",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-toggle-button/archive/v1.2.0.tar.gz",
          "https://github.com/PolymerElements/paper-toggle-button/archive/v1.2.0.tar.gz",
      ],
      strip_prefix = "paper-toggle-button-1.2.0",
      path = "/paper-toggle-button",
      srcs = ["paper-toggle-button.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_paper_behaviors",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_toolbar",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "dbddffc0654d9fb5fb48843087eebe16bf7a134902495a664c96c11bf8a2c63d",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-toolbar/archive/v1.1.4.tar.gz",
          "https://github.com/PolymerElements/paper-toolbar/archive/v1.1.4.tar.gz",
      ],
      strip_prefix = "paper-toolbar-1.1.4",
      path = "/paper-toolbar",
      srcs = ["paper-toolbar.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_iron_flex_layout",
          "@org_polymer_paper_styles",
      ],
  )

  web_library_external(
      name = "org_polymer_paper_tooltip",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "4c6667acf01f73da14c3cbc0aa574bf14280304567987ee0314534328377d2ad",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/paper-tooltip/archive/v1.1.2.tar.gz",
          "https://github.com/PolymerElements/paper-tooltip/archive/v1.1.2.tar.gz",
      ],
      strip_prefix = "paper-tooltip-1.1.2",
      path = "/paper-tooltip",
      srcs = ["paper-tooltip.html"],
      deps = [
          "@org_polymer",
          "@org_polymer_neon_animation",
      ],
  )

  web_library_external(
      name = "org_polymer_prism",
      licenses = ["notice"],  # MIT
      sha256 = "e06eb54f2a80e6b3cd0bd4d59f900423bcaee53fc03998a056df63740c684683",
      urls = [
          "http://mirror.bazel.build/github.com/PrismJS/prism/archive/abee2b7587f1925e57777044270e2a1860810994.tar.gz",
          "https://github.com/PrismJS/prism/archive/abee2b7587f1925e57777044270e2a1860810994.tar.gz",
      ],
      strip_prefix = "prism-abee2b7587f1925e57777044270e2a1860810994",
      path = "/prism",
      srcs = [
          "prism.js",
          "themes/prism.css",
      ],
  )

  web_library_external(
      name = "org_polymer_prism_element",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "c5c03c17520d4992a3576e397aa7a375e64c7f6794ec2af58031f47eef458945",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerElements/prism-element/archive/v1.2.0.tar.gz",
          "https://github.com/PolymerElements/prism-element/archive/v1.2.0.tar.gz",
      ],
      strip_prefix = "prism-element-1.2.0",
      path = "/prism-element",
      srcs = [
          "prism-highlighter.html",
          "prism-import.html",
          "prism-theme-default.html",
      ],
      deps = [
          "@org_polymer",
          "@org_polymer_prism",
      ],
  )

  web_library_external(
      name = "org_polymer_promise_polyfill",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "4495450e5d884c3e16b537b43afead7f84d17c7dc061bcfcbf440eac083e4ef5",
      strip_prefix = "promise-polyfill-1.0.0",
      urls = [
          "http://mirror.bazel.build/github.com/PolymerLabs/promise-polyfill/archive/v1.0.0.tar.gz",
          "https://github.com/PolymerLabs/promise-polyfill/archive/v1.0.0.tar.gz",
      ],
      path = "/promise-polyfill",
      srcs = [
          "Promise.js",
          "Promise-Statics.js",
          "promise-polyfill.html",
          "promise-polyfill-lite.html"
      ],
      deps = ["@org_polymer"],
  )

  web_library_external(
      name = "org_polymer_web_animations_js",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "f8bd760cbdeba131f6790bd5abe170bcbf7b1755ff58ed16d0b82fa8a7f34a7f",
      urls = [
          "http://mirror.bazel.build/github.com/web-animations/web-animations-js/archive/2.2.1.tar.gz",
          "https://github.com/web-animations/web-animations-js/archive/2.2.1.tar.gz",
      ],
      strip_prefix = "web-animations-js-2.2.1",
      path = "/web-animations-js",
      srcs = ["web-animations-next-lite.min.js"],
  )

  web_library_external(
      name = "org_polymer_webcomponentsjs",
      licenses = ["notice"],  # BSD-3-Clause
      sha256 = "1f58decac693deb926e6b62b5dbd459fb7c2e961f7241e6e646d1cd9a60281d2",
      urls = [
          "http://mirror.bazel.build/github.com/webcomponents/webcomponentsjs/archive/v0.7.23.tar.gz",
          "https://github.com/webcomponents/webcomponentsjs/archive/v0.7.23.tar.gz",
      ],
      strip_prefix = "webcomponentsjs-0.7.23",
      path = "/webcomponentsjs",
      srcs = [
          "CustomElements.js",
          "CustomElements.min.js",
          "HTMLImports.js",
          "HTMLImports.min.js",
          "MutationObserver.js",
          "MutationObserver.min.js",
          "ShadowDOM.js",
          "ShadowDOM.min.js",
          "webcomponents.js",
          "webcomponents.min.js",
          "webcomponents-lite.js",
          "webcomponents-lite.min.js",
      ],
  )
