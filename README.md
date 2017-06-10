# text-rnn-tensorflow
Tutorial: Multi-layer Recurrent Neural Networks (LSTM) for text models in Python using TensorFlow.


Before going through this tutorial, I suggest to read the very very good blog note from Andrej Karpathy: http://karpathy.github.io/2015/05/21/rnn-effectiveness/

This project takes also a lot from : https://github.com/hunkim/word-rnn-tensorflow by hunkim.
(honestly: almost every thing, this **word-rnn-tensorflow** project is great)

# Requirements
- [Tensorflow 1.1.0rc0](http://www.tensorflow.org)
- [Python 3.6](https://www.python.org/downloads/release/python-360/)

# Training Tutorial
The project has a **Train_RNN.ipynb** notebook.
Open it using Jupyter.

It describes the following activities:
- Load and prepare the data:
  - Load Data
  - Vocabulary creation
  - Tensor creation
  - Batches creation
- Model set-up
- Model training:
  - train the model
  - save it to local file

# Text generation Tutorial
The project has a **Generate_text_ipynb** notebook.
Open it using Jupyter.

It describes the following activities:
- Retrieve saved model
- Predict text

# simple_model.py

This python script embeds the definition of a class for the model:
- in order to train one RNN,
- and to use a saved RNN.

It's a 'simplification' of the **word-rnn-tensorflow** project, with a lot of comments inside to describe its steps.

# To Go further
Model training and text generation is done through notebooks in this tutorial.
If you want to use a more strengthened code, a more optimized code, embedding more features, I suggest to use the [word-rnn-tensorflow project](https://github.com/hunkim/word-rnn-tensorflow)

# additional notes
The project comes with two types of input:
- __data/tinyshakespeare/input.txt__:
  - a small condensate of Shakespeare books
- __data/Artistes_et_Phalanges-David_campion/input.txt__:
  - The complete text of a french fantasy book "Artistes et Phalanges", by David Campion
  - This file book is under the following licence: Licence Creative Commons [CC BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/)
