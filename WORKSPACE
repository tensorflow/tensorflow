# Uncomment and update the paths in these entries to build the Android demo.
#android_sdk_repository(
#    name = "androidsdk",
#    api_level = 23,
#    build_tools_version = "23.0.1",
#    # Replace with path to Android SDK on your system
#    path = "<PATH_TO_SDK>",
#)
#
#android_ndk_repository(
#    name="androidndk",
#    path="<PATH_TO_NDK>",
#    api_level=21)

new_http_archive(
  name = "gmock_archive",
  url = "https://googlemock.googlecode.com/files/gmock-1.7.0.zip",
  sha256 = "26fcbb5925b74ad5fc8c26b0495dfc96353f4d553492eb97e85a8a6d2f43095b",
  build_file = "gmock.BUILD",
)

bind(
  name = "gtest",
  actual = "@gmock_archive//:gtest",
)

bind(
  name = "gtest_main",
  actual = "@gmock_archive//:gtest_main",
)

git_repository(
  name = "re2",
  remote = "https://github.com/google/re2.git",
  tag = "2015-07-01",
)

new_http_archive(
  name = "jpeg_archive",
  url = "http://www.ijg.org/files/jpegsrc.v9a.tar.gz",
  sha256 = "3a753ea48d917945dd54a2d97de388aa06ca2eb1066cbfdc6652036349fe05a7",
  build_file = "jpeg.BUILD",
)

git_repository(
  name = "gemmlowp",
  remote = "https://github.com/google/gemmlowp.git",
  commit = "cc5d3a0",
)

new_http_archive(
  name = "png_archive",
  url = "https://storage.googleapis.com/libpng-public-archive/libpng-1.2.53.tar.gz",
  sha256 = "e05c9056d7f323088fd7824d8c6acc03a4a758c4b4916715924edc5dd3223a72",
  build_file = "png.BUILD",
)

new_http_archive(
  name = "six_archive",
  url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
  sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
  build_file = "six.BUILD",
)

bind(
  name = "six",
  actual = "@six_archive//:six",
)

new_git_repository(
  name = "iron-ajax",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-ajax.git",
  tag = "v1.0.8",
)

new_git_repository(
  name = "iron-dropdown",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-dropdown.git",
  tag = "v1.0.6",
)

new_git_repository(
  name = "accessibility-developer-tools",
  build_file = "bower.BUILD",
  remote = "https://github.com/GoogleChrome/accessibility-developer-tools.git",
  tag = "v2.10.0",
)

new_git_repository(
  name = "iron-doc-viewer",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-doc-viewer.git",
  tag = "v1.0.6",
)

new_git_repository(
  name = "iron-icons",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/iron-icons.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "paper-icon-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-icon-button.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "sinonjs",
  build_file = "bower.BUILD",
  remote = "https://github.com/blittle/sinon.js.git",
  tag = "v1.17.1",
)

new_git_repository(
  name = "paper-dropdown-menu",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-dropdown-menu.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "iron-flex-layout",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/iron-flex-layout.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "iron-autogrow-textarea",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-autogrow-textarea.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "d3",
  build_file = "bower.BUILD",
  remote = "https://github.com/mbostock/d3.git",
  tag = "v3.5.6",
)

new_git_repository(
  name = "iron-component-page",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-component-page.git",
  tag = "v1.0.8",
)

new_git_repository(
  name = "stacky",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerLabs/stacky.git",
  tag = "v1.2.4",
)

new_git_repository(
  name = "paper-styles",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-styles.git",
  tag = "v1.0.12",
)

new_git_repository(
  name = "paper-input",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-input.git",
  tag = "v1.0.16",
)

new_git_repository(
  name = "paper-item",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-item.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "marked-element",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/marked-element.git",
  tag = "v1.1.1",
)

new_git_repository(
  name = "prism",
  build_file = "bower.BUILD",
  remote = "https://github.com/LeaVerou/prism.git",
  tag = "v1.3.0",
)

new_git_repository(
  name = "paper-progress",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-progress.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-checked-element-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-checked-element-behavior.git",
  tag = "v1.0.2",
)

new_git_repository(
  name = "paper-toolbar",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-toolbar.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "async",
  build_file = "bower.BUILD",
  remote = "https://github.com/caolan/async.git",
  tag = "0.9.2",
)

new_git_repository(
  name = "es6-promise",
  build_file = "bower.BUILD",
  remote = "https://github.com/components/es6-promise.git",
  tag = "v3.0.2",
)

new_git_repository(
  name = "promise-polyfill",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerlabs/promise-polyfill.git",
  tag = "v1.0.0",
)

new_git_repository(
  name = "font-roboto",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/font-roboto.git",
  tag = "v1.0.1",
)

new_git_repository(
  name = "paper-menu",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-menu.git",
  tag = "v1.1.1",
)

new_git_repository(
  name = "iron-icon",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/iron-icon.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-meta",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-meta.git",
  tag = "v1.1.0",
)

new_git_repository(
  name = "lodash",
  build_file = "bower.BUILD",
  remote = "https://github.com/lodash/lodash.git",
  tag = "3.10.1",
)

new_git_repository(
  name = "iron-resizable-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-resizable-behavior.git",
  tag = "v1.0.2",
)

new_git_repository(
  name = "iron-fit-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-fit-behavior.git",
  tag = "v1.0.3",
)

new_git_repository(
  name = "iron-overlay-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-overlay-behavior.git",
  tag = "v1.0.9",
)

new_git_repository(
  name = "neon-animation",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/neon-animation.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-a11y-keys-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/iron-a11y-keys-behavior.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "plottable",
  build_file = "bower.BUILD",
  remote = "https://github.com/palantir/plottable.git",
  tag = "v1.16.1",
)

new_git_repository(
  name = "webcomponentsjs",
  build_file = "bower.BUILD",
  remote = "https://github.com/Polymer/webcomponentsjs.git",
  tag = "v0.7.15",
)

new_git_repository(
  name = "iron-validatable-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-validatable-behavior.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "sinon-chai",
  build_file = "bower.BUILD",
  remote = "https://github.com/domenic/sinon-chai.git",
  tag = "2.8.0",
)

new_git_repository(
  name = "paper-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-button.git",
  tag = "v1.0.8",
)

new_git_repository(
  name = "iron-input",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-input.git",
  tag = "v1.0.6",
)

new_git_repository(
  name = "iron-menu-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-menu-behavior.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "paper-slider",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-slider.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-list",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-list.git",
  tag = "v1.1.5",
)

new_git_repository(
  name = "marked",
  build_file = "bower.BUILD",
  remote = "https://github.com/chjj/marked.git",
  tag = "v0.3.5",
)

new_git_repository(
  name = "paper-material",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/paper-material.git",
  tag = "v1.0.3",
)

new_git_repository(
  name = "iron-range-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-range-behavior.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "svg-typewriter",
  build_file = "bower.BUILD",
  remote = "https://github.com/palantir/svg-typewriter.git",
  tag = "v0.3.0",
)

new_git_repository(
  name = "web-animations-js",
  build_file = "bower.BUILD",
  remote = "https://github.com/web-animations/web-animations-js.git",
  tag = "2.1.2",
)

new_git_repository(
  name = "hydrolysis",
  build_file = "bower.BUILD",
  remote = "https://github.com/Polymer/hydrolysis.git",
  tag = "v1.19.3",
)

new_git_repository(
  name = "web-component-tester",
  build_file = "bower.BUILD",
  remote = "https://github.com/Polymer/web-component-tester.git",
  tag = "v3.3.29",
)

new_git_repository(
  name = "paper-toggle-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-toggle-button.git",
  tag = "v1.0.11",
)

new_git_repository(
  name = "paper-behaviors",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/paper-behaviors.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "paper-radio-group",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-radio-group.git",
  tag = "v1.0.6",
)

new_git_repository(
  name = "iron-selector",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-selector.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-form-element-behavior",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-form-element-behavior.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "mocha",
  build_file = "bower.BUILD",
  remote = "https://github.com/mochajs/mocha.git",
  tag = "v2.3.3",
)

new_git_repository(
  name = "dagre",
  build_file = "bower.BUILD",
  remote = "https://github.com/cpettitt/dagre.git",
  tag = "v0.7.4",
)

new_git_repository(
  name = "iron-behaviors",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-behaviors.git",
  tag = "v1.0.9",
)

new_git_repository(
  name = "graphlib",
  build_file = "bower.BUILD",
  remote = "https://github.com/cpettitt/graphlib.git",
  tag = "v1.0.7",
)

new_git_repository(
  name = "iron-collapse",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-collapse.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "paper-checkbox",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-checkbox.git",
  tag = "v1.0.13",
)

new_git_repository(
  name = "paper-radio-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-radio-button.git",
  tag = "v1.0.10",
)

new_git_repository(
  name = "paper-header-panel",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/paper-header-panel.git",
  tag = "v1.0.5",
)

new_git_repository(
  name = "prism-element",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/prism-element.git",
  tag = "v1.0.2",
)

new_git_repository(
  name = "chai",
  build_file = "bower.BUILD",
  remote = "https://github.com/chaijs/chai.git",
  tag = "2.3.0",
)

new_git_repository(
  name = "paper-menu-button",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/paper-menu-button.git",
  tag = "v1.0.3",
)

new_git_repository(
  name = "polymer",
  build_file = "bower.BUILD",
  remote = "https://github.com/Polymer/polymer.git",
  tag = "v1.2.1",
)

new_git_repository(
  name = "paper-ripple",
  build_file = "bower.BUILD",
  remote = "https://github.com/polymerelements/paper-ripple.git",
  tag = "v1.0.4",
)

new_git_repository(
  name = "iron-iconset-svg",
  build_file = "bower.BUILD",
  remote = "https://github.com/PolymerElements/iron-iconset-svg.git",
  tag = "v1.0.8",
)
