import tensorflow as tf

def import_to_tensorboard(model_dir, log_dir):
  """View an imported protobuf model (.pb file) as a graph in Tensorboard.
  
  Args:
  	model_dir: The location of the protobuf (pb) model to visualize
    log_dir: The location for the Tensorboard log to begin visualisation from.
    
  Usage:
  	Call this function with your model location and desired log directory.
    Launch Tensorboard by pointing it to the log directory.
    View your imported .pb model as a graph.
  """
  with tf.Session(graph=tf.Graph()) as sess:
    with tf.gfile.FastGFile(model_dir, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      g_in = tf.import_graph_def(graph_def)

    pb_visual_writer = tf.summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "> tensorboard --logdir={}".format(log_dir))
