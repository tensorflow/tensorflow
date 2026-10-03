# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

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
            model_path + "/fingerprint.pb",
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
