import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models import MadryClassifier, Classifier, BayesClassifier, PGDAttackClassifier, PGDAttackCombined
from eval_utils import BaseDetectorFactory, load_mnist_data
from eval_utils import get_adv_errors, get_nat_accs

AES_FILE= sys.argv[-1]
phase = eval(sys.argv[-2])
assert phase in [1,2], '{} not in [1,2]'.format(phase)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

np.random.seed(123)

(x_train, y_train), (x_test, y_test) = load_mnist_data()
x_test_adv = np.load(AES_FILE)
if len(x_test_adv.shape) > 2: x_test_adv = x_test_adv.reshape(x_test_adv.shape[0], -1)
if np.max(x_test_adv) > 1: x_test_adv = x_test_adv.astype(np.float32) / 255.
adv_idxs = np.load(AES_FILE[:-4]+'_idxs.npy')
x_test = x_test[adv_idxs]
y_test = y_test[adv_idxs]

def mine_acc(preds_dist_nat, preds_dist_adv, nat_correct_idxs, adv_error_idxs):
    acc, err = [0], [0]#[1.], [1.]
    thrs = np.union1d(preds_dist_adv,preds_dist_nat)
    for d in thrs:
        nat2correct = np.sum(preds_dist_nat[nat_correct_idxs] <= d)
        adv2wrong = np.sum(preds_dist_adv[adv_error_idxs] <= d)
        acc.append(nat2correct*1.0/len(preds_dist_nat))
        err.append(adv2wrong*1.0/len(preds_dist_adv))
    return acc, err

preds_dist_nat = np.load('BAES/hamming/dist_nat.npy')
nat_correct_idxs = np.load('BAES/hamming/correct_idxs.npy')
preds_dist_adv = np.load('BAES/hamming/{}/dist_adv.npy'.format(AES_FILE.split('/')[-1][:-4]))
adv_error_idxs = np.load('BAES/hamming/{}/error_idxs.npy'.format(AES_FILE.split('/')[-1][:-4]))
acc, err = mine_acc(preds_dist_nat, preds_dist_adv, nat_correct_idxs, adv_error_idxs)
plt.plot(err, acc, label='hamming')

preds_dist_nat = np.load('BAES/euclidean/dist_nat.npy')
nat_correct_idxs = np.load('BAES/euclidean/correct_idxs.npy')
preds_dist_adv = np.load('BAES/euclidean/{}/dist_adv.npy'.format(AES_FILE.split('/')[-1][:-4]))
adv_error_idxs = np.load('BAES/euclidean/{}/error_idxs.npy'.format(AES_FILE.split('/')[-1][:-4]))
acc, err = mine_acc(preds_dist_nat, preds_dist_adv, nat_correct_idxs, adv_error_idxs)
plt.plot(err, acc, label='euclidean')

def get_integrated():
    classifier = Classifier(var_scope='classifier', dataset='MNIST')
    classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='classifier')
    classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
    factory = BaseDetectorFactory(eps=0.3)
    with tf.Session() as sess:
        # Restore variables
        classifier_saver.restore(sess, 'checkpoints/mnist/classifier')
        factory.restore_base_detectors(sess)
        base_detectors = factory.get_base_detectors()

        bayes_classifier = BayesClassifier(base_detectors)
        logit_ths = np.linspace(-250., 50.0, 1000)
        nat_accs = get_nat_accs(x_test, y_test, logit_ths, classifier,
                                    base_detectors, sess)
        adv_errors = get_adv_errors(x_test_adv, y_test, logit_ths, classifier,
                                        base_detectors, sess)
        np.save('BAES/integrated/{}_err.npy'.format(AES_FILE.split('/')[:-4]), adv_errors)
        np.save('BAES/integrated/{}_acc.npy'.format(AES_FILE.split('/')[:-4]), nat_accs)

def get_tmp_integrated():
    adv_errors = np.load('BAES/integrated/{}_err.npy'.format(AES_FILE.split('/')[:-4]))
    nat_accs = np.load('BAES/integrated/{}_acc.npy'.format(AES_FILE.split('/')[:-4]))
    plt.plot(adv_errors,
                     nat_accs,
                     label='Integrated classifier')
def get_generative():
    classifier = MadryClassifier(var_scope='classifier')
    classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope='classifier')
    classifier_saver = tf.train.Saver(var_list=classifier_vars, max_to_keep=1)
    factory = BaseDetectorFactory(eps=0.3)

    
    with tf.Session() as sess:
        classifier_saver.restore(
        sess,
        'checkpoints/mnist/adv_trained_prefixed_classifier/checkpoint-99900')
        factory.restore_base_detectors(sess)

        bayes_classifier = BayesClassifier(factory.get_base_detectors())

        # Evaluate robust classifier
        nat_acc = sess.run(classifier.accuracy,
                           feed_dict={
                               classifier.x_input: x_test,
                               classifier.y_input: y_test
                           })
        print('robust classifier nat acc {}'.format(nat_acc))
        adv_acc = sess.run(classifier.accuracy,
                       feed_dict={
                           classifier.x_input: x_test_adv,
                           classifier.y_input: y_test
                       })
        print('robust classifier adv acc {}'.format(adv_acc))
        plt.scatter(1-adv_acc, nat_acc, marker='X', color='black')

        bayes_nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
        bayes_adv_errors = bayes_classifier.adv_error(x_test_adv, y_test,
                                                      sess)
        plt.plot(bayes_adv_errors, bayes_nat_accs, label='Generative classifier')
if phase == 2:
    get_tmp_integrated()
    get_generative()
elif phase == 1: get_integrated()
plt.xlabel('Error on perturbed MNIST test set')
plt.ylabel('Accuracy on MNSIT test set')
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig('pics/acc_{}.png'.format(AES_FILE.split('/')[-1][:-4]))