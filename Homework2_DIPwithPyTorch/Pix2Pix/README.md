The provided code will train the model on the [Facades Dataset](https://cmp.felk.cvut.cz/~tylecr1/facade/). I use [Cityscapes Dataset](https://cmp.felk.cvut.cz/~tylecr1/cityscapes/) to train the model for better generalization on the validation set. You can change 'FILE' in [down_facades_dataset.sh](download_facades_dataset.sh) for more datasets.

For the Facades Dataset, the model was trained for 600 epochs.The loss of the model on the training set is 0.2063. The loss of the model on the test set is 0.4254.Some results of the training process and the final results can be viewed in the  [facades folder](facades/).

For the Cityscapes Dataset, the model was trained for 300 epochs. The loss of the model on the training set is 0.1012. The loss of the model on the test set is 0.1206.Some results of the training process and the final results can be viewed in the  [cityscapes folder](cityscapes/).
