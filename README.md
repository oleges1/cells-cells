# cells-cells
https://www.kaggle.com/c/recursion-cellular-image-classification

В качестве датасета юзается сейчас сконверченне к rgb и сохраненные в jpg картинки, которые подаются на вход в 6 каналов с двух камер, понятно что это штука спорная, поэтому можно сделать по-другому.
Датасеты:
 * [resized](https://www.kaggle.com/olegdesh/cellsresized)
 * [original size](https://www.kaggle.com/olegdesh/cellsjpg)

TRAINED MODELS: [link](https://drive.google.com/drive/folders/17nlf0TSxrZxM7JBCruto-BdiLbJfz7Nz?usp=sharing)

COLAB NOTEBOOK: [link](https://colab.research.google.com/drive/1AbzkPT1xKNdmmgRlS9s-76iTbI66dJIN)

This model is taken from [repo](https://github.com/earhian/Humpback-Whale-Identification-1st-)

TBD:
* prediction using [leak](https://www.kaggle.com/zaharch/keras-model-boosted-with-plates-leak)
* validation during training
* augmentations during training
* test time augmentations
* 2 step training with creating special model for each category as in [kernel](https://www.kaggle.com/xhlulu/recursion-2-headed-efficientnet-2-stage-training)
* Arcface loss && other tricks from face id or person re-identification (?)
