import numpy as np
import tensorflow as tf
import sys
import cifar10_input
from model import Model, BayesClassifier
from eval_utils import *
from pgd_attack import PGDAttackCombined, PGDAttack

attack_method = sys.argv[-1]

cifar = cifar10_input.CIFAR10Data('cifar10_data')
eval_data = cifar.eval_data

# this classifier is very expensive to run so we limit to a few samples
x_test = eval_data.xs.astype(np.float32)
y_test = eval_data.ys.astype(np.int32)

np.random.seed(123)
sess = tf.Session()

classifier = Model(mode='eval', var_scope='classifier')
classifier_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                            scope='classifier')
classifier_saver = tf.train.Saver(var_list=classifier_vars)
classifier_checkpoint = 'models/adv_trained_prefixed_classifier/checkpoint-70000'

factory = BaseDetectorFactory()
classifier_saver.restore(sess, classifier_checkpoint)
factory.restore_base_detectors(sess)
base_detectors = factory.get_base_detectors()
bayes_classifier = BayesClassifier(base_detectors)

# compute detection thresholds on the test set
nat_accs = bayes_classifier.nat_accs(x_test, y_test, sess)
nat_preds = bayes_classifier.batched_run(bayes_classifier.logits, x_test, sess)
idxs = np.random.permutation(np.arange(len(x_test))[np.argmax(nat_preds, axis=1)==y_test])[:1000]
x_test, y_test = x_test[idxs], y_test[idxs]

if attack_method == 'fgsm':
    eps8_attack_config = {
      'epsilon': 8.0,
      'num_steps': 1,
      'step_size': 8.0,
      'random_start': True,
      'norm': 'Linf'
    }
elif attack_method == 'pgd':
    eps8_attack_config = {
      'epsilon': 8.0,
      'num_steps': 100,
      'step_size': 2.5 * 8.0 / 100,
      'random_start': True,
      'norm': 'Linf'
    }

class PGDAttackOpt(PGDAttack):
    def __init__(self, bayes_classifier, target, **kwargs):
        super().__init__(**kwargs)

        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name='x_input')
        self.y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')
        logits = bayes_classifier.forward(self.x_input)

        label_mask = tf.one_hot(target, 10, dtype=tf.float32)

        clf_target_logit = tf.reduce_sum(label_mask * logits, axis=1)
        clf_other_logit = tf.reduce_max((1 - label_mask) * logits - 1e4 * label_mask, axis=1)

        # maximize target logit and minimize 2nd best logit until we have a targeted misclassification
        # then just maximize the target logit
        mask = tf.cast(tf.greater(clf_target_logit-0.01, clf_other_logit), tf.float32)
        clf_loss = clf_target_logit - (1-mask)*clf_other_logit
        
        self.loss = clf_loss
        self.grad = tf.gradients(self.loss, self.x_input)[0]

opt_adv = x_test.copy()
best_logit = np.asarray([-np.inf] * len(opt_adv))

for ii in range(40):
    for i in range(10):
        attack = PGDAttackOpt(bayes_classifier,
                              i,
                              **eps8_attack_config)
        
        x_test_adv = attack.batched_perturb(x_test, y_test, sess, batch_size=20)
        
        adv_preds = batched_run(bayes_classifier.predictions,
                                bayes_classifier.x_input, x_test_adv, sess)
        
        logits = batched_run(bayes_classifier.logits,
                             bayes_classifier.x_input, x_test_adv, sess)
        p_x = np.max(logits, axis=1)
        
        better_adv = (adv_preds != y_test) & (p_x > best_logit)
        best_logit[better_adv] = p_x[better_adv]
        opt_adv[better_adv] = x_test_adv[better_adv]
        if ii == 0:
            best_logit[adv_preds==y_test] = p_x[adv_preds==y_test]
            opt_adv[adv_preds==y_test] = x_test_adv[adv_preds==y_test]
        else:
            better_nat = (adv_preds == y_test) & (p_x < best_logit)
            best_logit[better_nat] = p_x[better_nat]
            opt_adv[better_nat] = x_test_adv[better_nat]
    
    print(i, np.mean(best_logit > -np.inf), np.mean(best_logit[best_logit > -np.inf]))
np.save('WAE/generative_fgsm_advs.npy', opt_adv)
# accuracy at 5% FPR
# opt_adv_errors = bayes_classifier.adv_errors(opt_adv, y_test, sess)
# tau = np.max(np.where(nat_accs >= np.max(nat_accs) - 0.05)[0])
# print("acc: {:.1f}%".format(100 * (1-opt_adv_errors[tau])))