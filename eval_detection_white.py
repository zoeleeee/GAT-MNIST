import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cifar10_input
from pgd_attack import PGDAttackCombined, PGDAttackClassifier
from model import Model, BayesClassifier
from eval_utils import *
from sklearn.metrics import roc_curve, auc


AES_FILE = sys.argv[-1]
attack_method = sys.argv[-2]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)

classifier_checkpoint_nat = 'checkpoints/mnist/classifier'
classifier_checkpoint_adv = 'checkpoints/mnist/adv_trained_prefixed_classifier/checkpoint-99900'

# load MINE results
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

preds_dist_nat = np.load('../../BAES/hamming/dist_nat.npy')
#nat_correct_idxs = np.load('../../BAES/hamming/correct_idxs.npy')
#nat_error_idxs = np.load('../../BAES/hamming/error_idxs.npy')
preds_dist_adv = np.load('../../BAES/hamming/{}/dist_adv.npy'.format(AES_FILE.split('/')[-1][:-4]))
#adv_correct_idxs = np.load('../../BAES/hamming/{}/correct_idxs.npy'.format(AES_FILE.split('/')[-1][:-4]))
#adv_error_idxs = np.load('../../BAES/hamming/{}/error_idxs.npy'.format(AES_FILE.split('/')[-1][:-4]))
tpr, fpr, mine_auc = mine_roc(preds_dist_nat, preds_dist_adv)
plt.plot(fpr, tpr, label='hamming auc: {}'.format(mine_auc))

preds_dist_nat = np.load('../../BAES/euclidean/dist_nat.npy')
preds_dist_adv = np.load('../../BAES/euclidean/{}/dist_adv.npy'.format(AES_FILE.split('/')[-1][:-4]))
tpr, fpr, mine_auc = mine_roc(preds_dist_nat, preds_dist_adv)
plt.plot(fpr, tpr, label='euclidean metric: {}'.format(mine_auc))

# load data
x_test = np.load('../mnist_update/data/mnist_data.npy')[60000:]
y_test = np.load('../mnist_update/data/mnist_labels.npy')[60000:]
if len(x_test.shape) > 2: x_test = x_test.reshape(x_test.shape[0], -1)
if np.max(x_test) > 1: x_test = x_test.astype(np.float32) / 255.

np.random.seed(123)

classifier = Classifier(var_scope='classifier', dataset='MNIST')
classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
factory = BaseDetectorFactory(eps=0.3)
logit_ths = np.linspace(-250., 50.0, 1000)
eval_batch_size = 100
with tf.Session() as sess:
    classifier_saver.restore(sess, classifier_checkpoint_nat)
    factory.restore_base_detectors(sess)
    base_detectors = factory.get_base_detectors()
    bayes_classifier = BayesClassifier(base_detectors)

    nat_accs = get_nat_accs(x_test, y_test, logit_ths, classifier, base_detectors, sess)
    nat_preds = batched_run(classifier.predictions, classifier.x_input, x_test, sess)
    idxs = np.random.permutation(np.arange(len(x_test))[nat_preds==y_test])[:1000]
    x_test, y_test = x_test[idxs], y_test[idxs]
    x_test_adv = np.load('WAE/integrated_{}_advs.npy'.format(attack_method))
    assert x_test.shape == x_test_adv, 'x_test.shape {} != x_test_adv.shape {}'.format(x_test.shape, x_test_adv.shape)

    # TPR on natural set for integrated detection
    nat_tpr = get_tpr(x_test, logit_ths, classifier, base_detectors, sess)
    adv_fpr = get_fpr(x_test_adv, y_test, logit_ths, classifier, base_detectors, sess)
    plt.plot(adv_fpr, nat_tpr, label='Integrated detection: {}'.format(np.round(auc(adv_fpr, nat_tpr), 5)))

    
    nat_preds = bayes_classifier.batched_run(bayes_classifier.logits, x_test, sess)
    idxs = np.random.permutation(np.arange(len(x_test))[nat_preds==y_test])[:1000]
    x_test, y_test = x_test[idxs], y_test[idxs]
    x_test_adv = np.load('WAE/generative_{}_advs.npy'.format(attack_method))
    assert x_test.shape == x_test_adv, 'x_test.shape {} != x_test_adv.shape {}'.format(x_test.shape, x_test_adv.shape)

    nat_tpr = bayes_classifier.nat_tpr(x_test, sess)
    adv_fpr = bayes_classifier.adv_fpr(x_test_adv, y_test, sess)
    plt.plot(adv_fpr, nat_tpr, label='Generative detection: {}'.format(np.round(auc(adv_fpr, nat_tpr), 5)))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig('pics/roc_white_{}_{}.png'.format(attack_method, AES_FILE.split('/')[-1][:-4]))