"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, ElasticNet, SaliencyMapMethod, Wasserstein
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10

attack_method = sys.argv[-1]

# Step 1: Load the MNIST dataset
min_pixel_value, max_pixel_value = 0., 1.
x_test = np.load('../mnist_update/data/mnist_data.npy')[60000:].astype(np.float32)
x_test /= 255.
y_test = np.load('../mnist_update/data/mnist_labels.npy')[60000:]
# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

# Step 1a: Swap axes to PyTorch's NCHW format

# x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
# x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

# Step 2: Load the model

model = torch.load('../mnist_update/mnist_shadow.pth')
model.eval()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32),
    nb_classes=10,
)

# Step 4: Train the ART classifier

# classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
tot_idxs = np.arange(len(y_test))[np.argmax(predictions, axis=1) == y_test]
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
if attack_method == 'PGD':
    attack = ProjectedGradientDescent(estimator=classifier, eps=0.3, eps_step=1.5e-3, num_random_init=1)

elif attack_method == 'FGSM':
    attack = FastGradientMethod(estimator=classifier, eps=0.3)

elif attack_method == 'CW':
    attack = CarliniL2Method(classifier=classifier)

elif attack_method == 'EAD':
    attack = ElasticNet(classifier=classifier)

elif attack_method == 'JSMA':
    attack = SaliencyMapMethod(classifier=classifier)

elif attack_method == 'Wasserstein':
    attack = Wasserstein(estimator=classifier, regularization=1)

# x_test_adv = attack.generate(x=x_test[tot_idxs])

# Step 7: Evaluate the ART classifier on adversarial test examples
amt = 0
for i in range(0, len(tot_idxs), 100):
    tmp_idxs = tot_idxs[i:min(len(tot_idxs),i+100)]
    advs = attack.generate(x=x_test[tmp_idxs])
    if np.max(advs) <=  1:
        advs *= 255.
    advs = advs.astype(np.uint8)

    predictions = np.argmax(classifier.predict(advs.astype(np.float32)/255.),axis=1)
    accuracy = np.sum(predictions == y_test[tmp_idxs]) / len(y_test[tmp_idxs])
    amt += np.sum(predictions == y_test[tmp_idxs])
# Step 7: Evaluate the ART classifier on adversarial test examples
    print("Accuracy on adversarial test examples: {}%".format(accuracy * min(len(tot_idxs)-i,100)))
    mask = (predictions != y_test[tmp_idxs])

    if i > 0:
    	x_test_adv = np.load('BAES/{}_AEs.npy'.format(attack_method))
    	x_test_adv = np.vstack((x_test_adv, advs[mask]))
    	x_test_idxs = np.load('BAES/{}_AEs_idxs.npy'.format(attack_method))
    	x_test_idxs = np.hstack((x_test_idxs, tmp_idxs[mask]))
    else:
    	x_test_idxs = tmp_idxs[mask]
    	x_test_adv = advs[mask]
    print(x_test_adv.shape, x_test_idxs.shape)
    np.save('BAES/{}_AEs_idxs.npy'.format(attack_method), x_test_idxs)
    np.save('BAES/{}_AEs.npy'.format(attack_method), x_test_adv)
print('{} final acc: {}'.format(attack_method, amt/len(tot_idxs)))
