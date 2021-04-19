from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import sys
import time

import numpy as np
import tensorflow as tf

# Loads trained model into graph (series of TensorFlow operations arranged into a graph of nodes)
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

# Takes in image file, resizes it to the desired configurations and returns the converted tensor
def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)

  # Uses different image decoder depending on type of file
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

# Loads created labels from tensorflow model
def load_labels(label_file):
  label = []

  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())

  return label


# Takes in image path and returns the predicted food name and type
def main(img):
  # Default configurations for tensorflow
  file_name = img
  model_file = "retrained_graph.pb"
  label_file = "retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  t = read_tensor_from_image_file(file_name,
                                  input_height=input_height,
                                  input_width=input_width,
                                  input_mean=input_mean,
                                  input_std=input_std)
  graph = load_graph(model_file)
  
  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name);
  output_operation = graph.get_operation_by_name(output_name);

  # Initiates a TensorFlow Graph object in which tensors are processed through operations
  with tf.Session(graph=graph) as sess:
    start = time.time()
    # Gets results for image tensor
    results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: t})
    end = time.time()
  results = np.squeeze(results)

  top_k = results.argsort()[-5:][::-1] # Finds top k largest entries for the last dimension
  labels = load_labels(label_file) # Gets labels for each of the 20 foods

  for i in top_k:
    # Dictionary with food name as key and type of food as value
    dictofclassifications = {
        "apple pie" : "dessert",
        "baby back ribs": "main course",
        "beef carpaccio": "main course",
        "beef tartare": "main course",
        "beignets": "dessert",
        "bibimbap": "main course",
        "breakfast burrito": "main course",
        "caesar salad": "appetizer",
        "cheese plate": "appetizer",
        "chicken quesadilla": "main course",
        "chicken wings": "appetizer",
        "chocolate cake": "dessert",
        "club sandwich": "main course",
        "cup cakes": "dessert",
        "donuts": "dessert",
        "dumplings": "appetizer",
        "eggs benedict": "main course",
        "falafel": "appetizer",
        "filet mignon": "main course",
        "fish and chips": "appetizer"
    }
    return labels[i], dictofclassifications.get(labels[i]) # Gets type of food (value) from food name (key) using dictofclassification