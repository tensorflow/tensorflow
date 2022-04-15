"""BUILD rules for generating saved model for testing."""

load("//tensorflow:tensorflow.bzl", "if_google")

def gen_saved_model(model_name = "", script = "", **kwargs):
    native.genrule(
        name = "saved_model_gen_" + model_name,
        srcs = [],
        outs = [
            model_name + "/saved_model.pb",
            model_name + "/variables/variables.data-00000-of-00001",
            model_name + "/variables/variables.index",
        ],
        cmd = if_google(
            "$(location " + script + ") --saved_model_path=$(RULEDIR)/" + model_name,
            "touch $(OUTS)",  # TODO(b/188517768): fix model gen.
        ),
        exec_tools = [script],
        **kwargs
    )
