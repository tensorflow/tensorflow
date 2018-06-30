"""BUILD rules for generating file-driven proto test cases.

The decode_proto_test_suite() and encode_proto_test_suite() rules take a list
of text protos and generates a tf_py_test() for each one.
"""

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "register_extension_info")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")

def _test_name(test, path):
  return "%s_%s_test" % (test, path.split("/")[-1].split(".")[0])

def decode_proto_test_suite(name, examples):
  """Build the decode_proto py_test for each test filename."""
  for test_filename in examples:
    tf_py_test(
        name = _test_name("decode_proto", test_filename),
        srcs = ["decode_proto_op_test.py"],
        size = "small",
        data = [test_filename] + if_static(
            [],
            otherwise = [":libtestexample.so"],
        ),
        main = "decode_proto_op_test.py",
        args = [
            "--message_text_file=\"%s/%s\"" % (native.package_name(), test_filename),
        ],
        additional_deps = [
            ":py_test_deps",
            "//third_party/py/numpy",
            "//tensorflow/contrib/proto:proto",
            "//tensorflow/contrib/proto/python/ops:decode_proto_op_py",
        ],
        tags = [
            "no_pip",  # TODO(b/78026780)
            "no_windows",  # TODO(b/78028010)
        ],
    )
  native.test_suite(
      name = name,
      tests = [":" + _test_name("decode_proto", test_filename)
               for test_filename in examples],
  )

def encode_proto_test_suite(name, examples):
  """Build the encode_proto py_test for each test filename."""
  for test_filename in examples:
    tf_py_test(
        name = _test_name("encode_proto", test_filename),
        srcs = ["encode_proto_op_test.py"],
        size = "small",
        data = [test_filename] + if_static(
            [],
            otherwise = [":libtestexample.so"],
        ),
        main = "encode_proto_op_test.py",
        args = [
            "--message_text_file=\"%s/%s\"" % (native.package_name(), test_filename),
        ],
        additional_deps = [
            ":py_test_deps",
            "//third_party/py/numpy",
            "//tensorflow/contrib/proto:proto",
            "//tensorflow/contrib/proto/python/ops:decode_proto_op_py",
            "//tensorflow/contrib/proto/python/ops:encode_proto_op_py",
        ],
        tags = [
            "no_pip",  # TODO(b/78026780)
            "no_windows",  # TODO(b/78028010)
        ],
    )
  native.test_suite(
      name = name,
      tests = [":" + _test_name("encode_proto", test_filename)
               for test_filename in examples],
  )

register_extension_info(
    extension_name = "decode_proto_test_suite",
    label_regex_map = {
        "deps": "deps:decode_example_.*",
    })

register_extension_info(
    extension_name = "encode_proto_test_suite",
    label_regex_map = {
        "deps": "deps:encode_example_.*",
    })
