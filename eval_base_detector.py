import argparse
import sys
import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import roc_curve, auc
from models import Detector, PGDAttackDetector

parser = argparse.ArgumentParser()
parser.add_argument('--target_class', type=int, required=True)
parser.add_argument('--epsilon', metavar='max-distance', type=float, required=True)
parser.add_argument('--norm', choices=['L2', 'Linf'], default='Linf')
parser.add_argument('--optimizer', choices=['adam', 'normgrad'], default='adam')
parser.add_argument('--steps', type=int, default=200)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('-p', '--checkpoint', type=str, required=True)

args = parser.parse_args()
print(args)

np.random.seed(123)


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.reshape(x_train, [x_train.shape[0], -1])
x_test = np.reshape(x_test, [x_test.shape[0], -1])

x_min, x_max = 0.0, 1.0

scope = 'detector-class{}'.format(args.target_class)
detector = Detector(var_scope=scope, dataset='MNIST')
detector_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
detector_saver = tf.train.Saver(var_list=detector_vars)

x_test_target = x_test[y_test == args.target_class]
x_test_others = x_test[y_test != args.target_class]
y_test_others = y_test[y_test != args.target_class]

attack = PGDAttackDetector(detector=detector,
                           max_distance=args.epsilon,
                           num_steps=args.steps, step_size=args.step_size,
                           random_start=False, x_min=x_min, x_max=x_max,
                           batch_size=x_test_others.shape[0],
                           norm=args.norm, optimizer=args.optimizer)

with tf.Session() as sess:
  detector_saver.restore(sess, args.checkpoint)

  x_test_others_adv = attack.perturb(x_test_others, y_test_others, sess, verbose=True)

  x = np.concatenate([x_test_target, x_test_others_adv])
  y = np.concatenate([np.ones(x_test_target.shape[0], np.int64), np.zeros(x_test_others_adv.shape[0], np.int64)])

  logits = sess.run(detector.logits, feed_dict={detector.x_input: x})
  fpr_, tpr_, thresholds = roc_curve(y, logits)
  roc_auc = auc(fpr_, tpr_)
  print('roc_auc: {}'.format(roc_auc))

