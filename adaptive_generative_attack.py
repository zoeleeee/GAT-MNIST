import numpy as np
import tensorflow as tf
import sys
from eval_utils import *
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
    def __init__(self, bayes_classifier, target, **kwargs):
        super().__init__(**kwargs)

        logits = bayes_classifier.forward(self.x)

        label_mask = tf.one_hot(target, 10, dtype=tf.float32)

        clf_target_logit = tf.reduce_sum(label_mask * logits, axis=1)
        clf_other_logit = tf.reduce_max((1 - label_mask) * logits - 1e4 * label_mask, axis=1)

        # maximize target logit and minimize 2nd best logit until we have a targeted misclassification
        # then just maximize the target logit
        mask = tf.cast(tf.greater(clf_target_logit-0.01, clf_other_logit), tf.float32)
        clf_loss = clf_target_logit - (1-mask)*clf_other_logit
        
        self.loss = clf_loss
        self.grad = tf.gradients(self.loss, self.x)[0]
        self.setup_optimizer()

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