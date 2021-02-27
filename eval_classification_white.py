import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cifar10_input
from pgd_attack import PGDAttackClassifier, PGDAttackDetector
from model import Model, BayesClassifier
from eval_utils import *

AES_FILE = sys.argv[-1]
attack_method = sys.argv[-2]

if attack_method == 'fgsm':
    eps8_attack_config = {
      'max_distance': 0.3,
      'num_steps': 1,
      'step_size': 0.3,
      'random_start': False,
      'x_min': 0,
      'x_max': 1.0,
      'batch_size': 50,
      'optimizer': 'adam',
      'norm': 'Linf'
    }
elif attack_method == 'pgd':
    eps8_attack_config = {
      'max_distance': 0.3,
      'num_steps': 100,
      'step_size': 0.01,
      'random_start': False,
      'x_min': 0,
      'x_max': 1.0,
      'batch_size': 50,
      'optimizer': 'adam',
      'norm': 'Linf'
    }
# initialize gat method
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)

classifier_checkpoint_nat = 'checkpoints/mnist/classifier'
classifier_checkpoint_adv = 'checkpoints/mnist/adv_trained_prefixed_classifier/checkpoint-99900'

# mine res
# load MINE results
def mine_acc(preds_dist_nat, preds_dist_adv, nat_correct_idxs, adv_error_idxs):
    acc, err = [0], [0]#[1.], [1.]
    thrs = np.union1d(preds_dist_adv,preds_dist_nat)
    for d in thrs:
        nat2correct = np.sum(preds_dist_nat[nat_correct_idxs] <= d)
        adv2wrong = np.sum(preds_dist_adv[adv_error_idxs] <= d)
        acc.append(nat2correct*1.0/len(preds_dist_nat))
        err.append(adv2wrong*1.0/len(preds_dist_adv))
    return acc, err

preds_dist_nat = np.load('../mnist_udpate/BAES/hamming/dist_nat.npy')
nat_correct_idxs = np.load('../mnist_udpate/BAES/hamming/correct_idxs.npy')
preds_dist_adv = np.load('../mnist_udpate/BAES/hamming/{}/dist_adv.npy'.format(AES_FILE.split('/')[-1][:-4]))
adv_error_idxs = np.load('../mnist_udpate/BAES/hamming/{}/error_idxs.npy'.format(AES_FILE.split('/')[-1][:-4]))
acc, err = mine_acc(preds_dist_nat, preds_dist_adv, nat_correct_idxs, adv_error_idxs)
plt.plot(err, acc, label='hamming')

preds_dist_nat = np.load('../mnist_udpate/BAES/euclidean/dist_nat.npy')
nat_correct_idxs = np.load('../mnist_udpate/BAES/euclidean/correct_idxs.npy')
preds_dist_adv = np.load('../mnist_udpate/BAES/euclidean/{}/dist_adv.npy'.format(AES_FILE.split('/')[-1][:-4]))
adv_error_idxs = np.load('../mnist_udpate/BAES/euclidean/{}/error_idxs.npy'.format(AES_FILE.split('/')[-1][:-4]))
acc, err = mine_acc(preds_dist_nat, preds_dist_adv, nat_correct_idxs, adv_error_idxs)
plt.plot(err, acc, label='euclidean')

# load data
x_test = np.load('../mnist_update/data/mnist_data.npy')[60000:]
y_test = np.load('../mnist_update/data/mnist_labels.npy')[60000:]
if len(x_test.shape) > 2: x_test = x_test.reshape(x_test.shape[0], -1)
if np.max(x_test) > 1: x_test = x_test.astype(np.float32) / 255.

def get_integrated():
  classifier = Classifier(var_scope='classifier', dataset='MNIST')
  classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='classifier')
  classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
  factory = BaseDetectorFactory(eps=0.3)
  logit_ths = np.linspace(-250., 50.0, 1000)
  with tf.Session() as sess:
    np.random.seed(123)
    classifier_saver.restore(sess, classifier_checkpoint_nat)
    factory.restore_base_detectors(sess)
    base_detectors = factory.get_base_detectors()
    bayes_classifier = BayesClassifier(base_detectors)

    nat_accs = get_nat_accs(x_test, y_test, logit_threshs, classifier, base_detectors, sess)
    nat_preds = batched_run(classifier.predictions, classifier.x_input, x_test, sess)
    idxs = np.random.permutation(np.arange(len(x_test))[nat_preds==y_test])[:1000]
    x_test, y_test = x_test[idxs], y_test[idxs]
    x_test_adv = np.load('WAE/integrated_fgsm_advs.npy')
    assert x_test.shape == x_test_adv, 'x_test.shape {} != x_test_adv.shape {}'.format(x_test.shape, x_test_adv.shape)

    nat_accs = get_nat_accs(x_test, y_test, logit_ths, classifier,
                              base_detectors, sess)
    adv_errors = get_adv_errors(x_test_adv, y_test, logit_ths,
                                    classifier, base_detectors, sess)
    np.save('WAE/integrated_white_{}_err.npy'.format(AES_FILE.split('/')[:-4]), adv_errors)
    np.save('WAE/integrated_white_{}_acc.npy'.format(AES_FILE.split('/')[:-4]), nat_accs)

def get_tmp_integrated():
    adv_errors = np.load('WAE/integrated_white_{}_err.npy'.format(AES_FILE.split('/')[:-4]))
    nat_accs = np.load('WAE/integrated_white_{}_acc.npy'.format(AES_FILE.split('/')[:-4]))
    plt.plot(adv_errors,
                     nat_accs,
                     label='Integrated classifier')

def get_generative():
  classifier = MadryClassifier(var_scope='classifier')
  classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                      scope='classifier')
  classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
  factory = BaseDetectorFactory(eps=0.3)

  # reload robust classifier
  with tf.Session() as sess:
    classifier_saver.restore(sess, classifier_checkpoint_adv)
    factory.restore_base_detectors(sess)
    base_detectors = factory.get_base_detectors()
    bayes_classifier = BayesClassifier(base_detectors)

    np.random.seed(123)

    nat_preds = bayes_classifier.batched_run(bayes_classifier.predictions, x_test, sess)
    idxs = np.random.permutation(np.arange(len(x_test))[nat_preds==y_test])[:1000]
    x_test, y_test = x_test[idxs], y_test[idxs]
    x_test_adv = np.load('WAE/generative_fgsm_advs.npy')
    assert x_test.shape == x_test_adv, 'x_test.shape {} != x_test_adv.shape {}'.format(x_test.shape, x_test_adv.shape)
    nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
    adv_errors = bayes_classifier.adv_errors(x_test_adv, y_test, sess)
    plt.plot(adv_errors, nat_accs, label='Generative classifier')

    # robust classifier
    preds = sess.run(classifier.predictions,
                       feed_dict={
                           classifier.x_input: x_test,
                           classifier.y_input: y_test
                       })
    idxs = np.random.permuatation(np.arange(len(x_test))[preds==y_test])[:1000]
    print('robust classifier standard acc {}'.format(
        (preds == y_test).mean()))  # nat acc 0.8725
    x_test, y_test = x_test[idxs], y_test[idxs]

    attack = PGDAttackClassifier(classifier=classifier,
                                   loss_func='cw',
                                   targeted = True,
                                   **eps8_attack_config)
    res = np.zeros(len(x_test))
    for ii in range(40):
      for i in range(1,10):
        x_test_adv = attack.batched_perturb(x_test, (y_test+i)%10, sess)
        preds = batched_run(robust_classifier.predictions,
                            robust_classifier.x_input, x_test_adv, sess)
        res = np.logical_and(res, preds!=y_test)
        print('robust classifier adv acc {}, eps=8'.format(
            np.mean(res)))  # adv acc 0.4689
    plt.scatter(np.mean(res), 1.0, marker='X')

  plt.xlabel('Error on {}'.format(AES_FILE.split('/')[-1][:-4]))
  plt.ylabel('Accuracy on MNIST test set')
  plt.legend()
  plt.grid(True, alpha=0.5)
  plt.savefig('../../pics/white_{}_{}.png'.format(attack_method, AES_FILE.split('/')[-1][:-4]))