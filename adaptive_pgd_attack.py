import numpy as np
import tensorflow as tf
from eval_utils import *
import sys
from models import MadryClassifier, Classifier, BayesClassifier, PGDAttackClassifier, PGDAttackCombined, PGDAttack

attack_method = sys.argv[-1]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

x_test = np.load('../mnist_update/data/mnist_data.npy')[60000:]
y_test = np.load('../mnist_update/data/mnist_labels.npy')[60000:]
if len(x_test.shape) > 2: x_test = x_test.reshape(x_test.shape[0], -1)
if np.max(x_test) > 1: x_test = x_test.astype(np.float32) / 255.

np.random.seed(123)
sess = tf.Session()

classifier = Classifier(var_scope='classifier', dataset='MNIST')
classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
factory = BaseDetectorFactory(eps=0.3)
classifier_saver.restore(sess, 'checkpoints/mnist/classifier')
factory.restore_base_detectors(sess)

base_detectors = factory.get_base_detectors()
bayes_classifier = BayesClassifier(base_detectors)

# compute detection thresholds on the test set
nat_preds = sess.run(classifier.predictions,
                         feed_dict={classifier.x_input: x_test})
idxs = np.random.permutation(np.arange(len(x_test))[nat_preds==y_test])[:1000]
x_test, y_test = x_test[idxs], y_test[idxs]

if attack_method == 'fgsm':
    eps8_attack_config = {
      'max_distance': 0.3,
      'num_steps': 1,
      'step_size': 0.3,
      'random_start': True,
      'norm': 'Linf',
      'x_min': 0,
      'x_max': 1.0,
      'batch_size': 50,
      'optimizer': 'adam',
    }
elif attack_method == 'pgd':
    eps8_attack_config = {
      'epsilon': 0.3,
      'num_steps': 100,
      'step_size': 0.01,
      'random_start': True,
      'norm': 'Linf',
      'x_min': 0,
      'x_max': 1.0,
      'batch_size': 50,
      'optimizer': 'adam',
    }

class PGDAttackOpt(PGDAttack):
    def __init__(self, naive_classifier, base_detector, **kwargs):
        super().__init__(**kwargs)

        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='x_input')
        self.y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')
        clf_logits = naive_classifier.forward(self.x_input)
        det_logits = base_detector.forward(self.x_input)

        label_mask = tf.one_hot(base_detector.target_class, 10, dtype=tf.float32)

        clf_target_logit = tf.reduce_sum(label_mask * clf_logits, axis=1)
        clf_other_logit = tf.reduce_max((1 - label_mask) * clf_logits - 1e4 * label_mask, axis=1)

        det_target_logit = tf.reduce_sum(label_mask * det_logits, axis=1)

        # maximize target logit and minimize 2nd best logit until we have a targeted misclassification
        mask = tf.cast(tf.greater(clf_target_logit - 0.01, clf_other_logit), tf.float32)
        clf_loss = (1-mask) * (clf_target_logit - clf_other_logit)

        # just maximize the target logit for the detector once we have a misclassification
        det_loss = mask * det_target_logit

        self.loss = clf_loss + det_loss
        self.grad = tf.gradients(self.loss, self.x_input)[0]

opt_adv = x_test.copy()
best_logit = np.asarray([-np.inf] * len(opt_adv))

for ii in range(40):
  for i in range(10):
      attack = PGDAttackOpt(classifier,
                            base_detectors[i],
                            **eps8_attack_config)
      
      x_test_adv = attack.batched_perturb(x_test, y_test, sess, batch_size=50)
      adv_preds = sess.run(classifier.predictions,
                         feed_dict={classifier.x_input: x_test_adv})
      print(x_test.shape, x_test_adv.shape, adv_preds.shape)
      det_logits = get_det_logits(x_test_adv, adv_preds, base_detectors, sess)
      
      better_adv = (adv_preds != y_test) & (det_logits > best_logit)
      best_logit[better_adv] = det_logits[better_adv]
      opt_adv[better_adv] = x_test_adv[better_adv]
      if ii == 0:
        best_logit[adv_preds==y_test] = det_logits[adv_preds==y_test]
        opt_adv[adv_preds==y_test] = x_test_adv[adv_preds==y_test]
      else:
        better_nat = (adv_preds == y_test) & (det_logits < best_logit)
        best_logit[better_nat] = det_logits[better_nat]
        opt_adv[better_nat] = x_test_adv[better_nat]
      
      print(ii, i, np.mean(best_logit > -np.inf), np.mean(best_logit[best_logit > -np.inf]))
np.save('WAE/integrated_fgsm_advs.npy', opt_adv)

#opt_adv_errors = get_adv_errors(opt_adv, y_test, logit_threshs, classifier, base_detectors, sess)
#tau = np.max(np.where(nat_accs >= np.max(nat_accs) - 0.05)[0])
#print("acc: {:.1f}%".format(100 * (1-opt_adv_errors[tau])))
