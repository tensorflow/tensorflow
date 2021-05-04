# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import tempfile

import tensorflow as tf
import tensorflow_text as tf_text


class ConvertResource(tf.test.TestCase):

  def testConvertResource(self):
    """Test general resource inputs don't crash the converter."""

    class TokenizeLayer(tf.keras.layers.Layer):

      def __init__(self, vocab_file):
        super().__init__()
        serialized_proto = tf.compat.v1.gfile.GFile(vocab_file, "rb").read()
        self.tokenizer = tf_text.SentencepieceTokenizer(
            model=serialized_proto, add_bos=True, add_eos=True)

      def call(self, inputs):
        word_ids = self.tokenizer.tokenize(inputs)
        word_ids = word_ids.to_tensor(default_value=1, shape=(None, 192))
        return word_ids

    vocab_file = tf.compat.v1.test.test_src_dir_path(
        "python/keras/integration_test/data/sentencepiece.pb")
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())

    # Create and save a Tokenizer
    tokenizer = TokenizeLayer(vocab_file)
    inputs = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    tokens = tokenizer(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=tokens)
    model.save(output_dir)

    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=output_dir,
        conversion_params=tf.experimental.tensorrt.ConversionParams())
    converter.convert()


if __name__ == "__main__":
  tf.test.main()
