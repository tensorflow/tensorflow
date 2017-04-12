import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework.ops import reset_default_graph

def import_to_tensorboard(modeldir,logdir):
# View an imported protobuf model (.pb file) as a graph in Tensorboard
# Args:
# 	modeldir: The location of the protobuf (ph) model to visualize
#	  logdir: The location for the Tensorboard log to begin visualisation from.
# Usage:
# 	Call this function with your model location and desired log directory.
#	Launch Tensorboard by pointing it to the log directory.
#	View your imported .pb model as a graph.
#
#
  reset_default_graph()
  with tf.Session() as sess:
    model_filename = modeldir
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      g_in = tf.import_graph_def(graph_def)

  pb_visual_writer = tf.summary.FileWriter(logdir)
  pb_visual_writer.add_graph(sess.graph)
  print("Model Imported. Visualize by running > tensorboard --logdir={}".format(logdir))
