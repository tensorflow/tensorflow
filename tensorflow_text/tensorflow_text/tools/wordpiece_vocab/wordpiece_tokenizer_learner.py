# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Binary for learning wordpiece vocabulary."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
from tensorflow_text.tools.wordpiece_vocab import wordpiece_tokenizer_learner_lib as learner


FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None, 'Path to wordcount file.')
flags.mark_flag_as_required('input_path', FLAGS)
flags.DEFINE_string('output_path', None, 'Path to vocab file.')
flags.mark_flag_as_required('output_path', FLAGS)

flags.DEFINE_integer('upper_thresh', 10000000,
                     'Upper threshold for binary search.')
flags.DEFINE_integer('lower_thresh', 10, 'Lower threshold for binary search.')
flags.DEFINE_integer('num_iterations', 4,
                     'Number of iterations in wordpiece learning algorithm.')
flags.DEFINE_integer('num_pad_tokens', 100, 'Number of padding tokens to '
                     'include in vocab.')
flags.DEFINE_integer('max_input_tokens', 5000000,
                     'Maximum number of input tokens, where -1 means no max.')
flags.DEFINE_integer('max_token_length', 50, 'Maximum length of a token.')
flags.DEFINE_integer('max_unique_chars', 1000,
                     'Maximum number of unique characters as tokens.')
flags.DEFINE_integer('vocab_size', 110000, 'Target size of generated vocab, '
                     'where vocab_size is an upper bound and the size of vocab '
                     'can be within slack_ratio less than the vocab_size.')
flags.DEFINE_float('slack_ratio', 0.05,
                   'Difference permitted between target and actual vocab size.')
flags.DEFINE_bool('include_joiner_token', True,
                  'Whether to include joiner token in word suffixes.')
flags.DEFINE_string('joiner', '##', 'Joiner token in word suffixes.')
flags.DEFINE_list('reserved_tokens',
                  ['<unk>', '<s>', '</s>', '<mask>',
                   '<cls>', '<sep>', '<S>', '<T>'],
                  'Reserved tokens to be included in vocab.')


def main(_):
  # Read in wordcount file.
  with open(FLAGS.input_path) as wordcount_file:
    word_counts = [(line.split()[0], int(line.split()[1]))
                   for line in wordcount_file]

  # Add in padding tokens.
  reserved_tokens = FLAGS.reserved_tokens
  if FLAGS.num_pad_tokens:
    padded_tokens = ['<pad>']
    padded_tokens += ['<pad%d>' % i for i in range(1, FLAGS.num_pad_tokens)]
    reserved_tokens = padded_tokens + reserved_tokens

  vocab = learner.learn(
      word_counts,
      vocab_size=FLAGS.vocab_size,
      reserved_tokens=reserved_tokens,
      upper_thresh=FLAGS.upper_thresh,
      lower_thresh=FLAGS.lower_thresh,
      num_iterations=FLAGS.num_iterations,
      max_input_tokens=FLAGS.max_input_tokens,
      max_token_length=FLAGS.max_token_length,
      max_unique_chars=FLAGS.max_unique_chars,
      slack_ratio=FLAGS.slack_ratio,
      include_joiner_token=FLAGS.include_joiner_token,
      joiner=FLAGS.joiner)
  vocab = ''.join([line + '\n' for line in vocab])

  # Write vocab to file.
  with open(FLAGS.output_path, 'w') as vocab_file:
    vocab_file.write(vocab)


if __name__ == '__main__':
  app.run(main)
