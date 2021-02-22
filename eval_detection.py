import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import Classifier, BayesClassifier, PGDAttackClassifier, PGDAttackCombined
from eval_utils import BaseDetectorFactory, load_mnist_data
from eval_utils import get_tpr, get_fpr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)
attack_file = sys.argv[-1]

# (x_train, y_train), (x_test, y_test) = load_mnist_data()
x_test = np.load('../mnist_update/data/mnist_data.npy')[60000:]
y_test = np.load('../mnist_update/data/mnist_labels.npy')[60000:]
if np.argmin(x_test.shape) == 1: x_test = x_test.transpose((0,2,3,1))
if np.max(x_test) > 1: x_test = x_test.astype(np.float32) / 255.
idxs = np.load(attack_file[:-4]+'_idxs.npy')
x_test, y_test = x_test[idxs], y_test[idxs]
x_test_adv = np.load(attack_file)
if np.argmin(x_test_adv.shape) == 1: x_test_adv = x_test_adv.transpose((0,2,3,1))
if np.max(x_test_adv) > 1: x_test_adv = x_test_adv.astype(np.float32) / 255.
assert x_test_adv.shape==x_test.shape, 'adv shape: {}, nat shape: {}'.format(x_test_adv.shape, x_test.shape)

classifier = Classifier(var_scope='classifier', dataset='MNIST')
classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
factory = BaseDetectorFactory(eps=0.3)

####mine######
def mine_roc(preds_dist_nat, preds_dist_adv):
    tpr, fpr = [0], [0]#[1.], [1.]
    thrs = np.union1d(preds_dist_adv,preds_dist_nat)
    for d in thrs:
        #adv2adv = np.sum(preds_dist_adv > d)
        nat2nat = np.sum(preds_dist_nat <= d)
        #nat2adv = np.sum(preds_dist_nat > d)
        adv2nat = np.sum(preds_dist_adv <= d)
        #tpr.append(adv2adv*1.0/len(preds_dist_adv))
        tpr.append(nat2nat*1.0/len(preds_dist_nat))
        #fpr.append(nat2adv*1.0/len(preds_dist_nat))
        fpr.append(adv2nat*1.0/len(preds_dist_adv))
    mine_auc = np.round(auc(fpr, tpr), 5)
    return tpr, fpr, mine_auc

preds_dist_nat = np.load('BAES/hamming/dist_nat.npy')
#nat_correct_idxs = np.load('../../BAES/hamming/correct_idxs.npy')
#nat_error_idxs = np.load('../../BAES/hamming/error_idxs.npy')
preds_dist_adv = np.load('BAES/hamming/{}/dist_adv.npy'.format(AES_FILE.split('/')[-1][:-4]))
#adv_correct_idxs = np.load('../../BAES/hamming/{}/correct_idxs.npy'.format(AES_FILE.split('/')[-1][:-4]))
#adv_error_idxs = np.load('../../BAES/hamming/{}/error_idxs.npy'.format(AES_FILE.split('/')[-1][:-4]))
tpr, fpr, mine_auc = mine_roc(preds_dist_nat, preds_dist_adv)
plt.plot(fpr, tpr, label='hamming auc: {}'.format(mine_auc))

preds_dist_nat = np.load('BAES/euclidean/dist_nat.npy')
preds_dist_adv = np.load('BAES/euclidean/{}/dist_adv.npy'.format(AES_FILE.split('/')[-1][:-4]))
tpr, fpr, mine_auc = mine_roc(preds_dist_nat, preds_dist_adv)
plt.plot(fpr, tpr, label='euclidean metric: {}'.format(mine_auc))

# plt.figure(figsize=(3.5 * 1.7, 2 * 1.7))
logit_ths = np.linspace(-250., 50.0, 1000)

with tf.Session() as sess:
    # Restore variables
    classifier_saver.restore(sess, 'checkpoints/mnist/classifier')
    factory.restore_base_detectors(sess)
    base_detectors = factory.get_base_detectors()
    bayes_classifier = BayesClassifier(base_detectors)

    def compute_fpr(x_test_adv, sess):
        return 

    # Integrated detection
    tpr = get_tpr(x_test, logit_ths, classifier, base_detectors, sess)
    fpr = get_fpr(x_test_adv, y_test, logit_ths, classifier, base_detectors, sess)
    plt.plot(fpr, tpr, label='Integrated detection')

    # Generative detection
    tpr = bayes_classifier.nat_tpr(x_test, sess)
    fpr = bayes_classifier.adv_fpr(x_test_adv, y_test, sess)
    plt.plot(fpr, tpr, label='Generative detection')

    # plt.ylim([0.9, 1.0])
    # plt.xlim([0.0, 0.5])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()
