from tensorflow.python.platform import test
import tensorflow as tf
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.image_ops_impl import combined_non_max_suppression

uniform = random_ops.random_uniform

class CombinedNmsShapeTest(test.TestCase):

  def test_boxes_wrong_rank_raises(self):
    boxes  = uniform([4, 10, 4])      # rank-3
    scores = uniform([4, 10, 1])      # rank-3 OK
    with self.assertRaisesRegex(ValueError, "boxes"):
      combined_non_max_suppression(boxes, scores, 10, 10)

  def test_scores_wrong_rank_raises(self):
    boxes  = uniform([4, 10, 1, 4])   # rank-4 OK
    scores = uniform([4, 10])         # rank-2
    with self.assertRaisesRegex(ValueError, "scores"):
      combined_non_max_suppression(boxes, scores, 10, 10)

if __name__ == "__main__":
  test.main()
