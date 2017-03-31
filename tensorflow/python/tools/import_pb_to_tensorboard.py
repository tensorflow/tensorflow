import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework.ops import reset_default_graph

def import_to_tensorboard(modeldir,logdir):
# import_to_tensorboard(modeldir,logdir) provides a quick way to visualize 
# a .pb model in Tensorboard.
# Created in response to issue #8854
# Args:
# 	modeldir: The location of the .pb model to visualize
#	logdir: The location for the Tensorboard log to begin visualisation from.
# Usage:
# 	Call this function with your model location and desired log directory.
#	Launch Tensorboard by pointing it to the log directory.
#	View your imported .pb model as a graph.
#
#
  reset_default_graph()
  with tf.Session() as sess:
    model_filename =modeldir
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      g_in = tf.import_graph_def(graph_def)

  pbVisualWriter = tf.summary.FileWriter(logdir)
  pbVisualWriter.add_graph(sess.graph)
  print("Model Imported. Visualize by running > tensorboard --logdir={}".format(logdir))
