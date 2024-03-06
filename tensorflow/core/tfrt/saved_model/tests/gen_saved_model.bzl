"""BUILD rules for generating saved model for testing."""

load("//tensorflow:tensorflow.bzl", "if_google")

def gen_saved_model(model_name = "", script = "", version = "", **kwargs):
    model_path = model_name
    if version != "":
        model_path = model_name + "/" + version

    native.genrule(
        name = "saved_model_gen_" + model_name,
        srcs = [],
        outs = [
            model_path + "/saved_model.pb",
            model_path + "/variables/variables.data-00000-of-00001",
            model_path + "/variables/variables.index",
        ],
        cmd = if_google(
            "$(location " + script + ") --saved_model_path=$(RULEDIR)/" + model_path,
            "touch $(OUTS)",  # TODO(b/188517768): fix model gen.
        ),
        tools = [script],
        **kwargs
    )

def gen_variableless_saved_model(model_name = "", script = "", **kwargs):
    native.genrule(
        name = "saved_model_gen_" + model_name,
        srcs = [],
        outs = [
            model_name + "/saved_model.pb",
        ],
        cmd = if_google(
            "$(location " + script + ") --saved_model_path=$(RULEDIR)/" + model_name,
            "touch $(OUTS)",  # TODO(b/188517768): fix model gen.
        ),
        tools = [script],
        **kwargs
    )
