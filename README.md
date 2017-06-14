# Image-recognition

This code implements the Back Propagation algorithm for Feed Forward Neural Networks to learn down gestures from training images available in gesture folder. The label of training data is provided in downgesture_train.list file. The label of an image is 1 if the word "down" is in its file name; otherwise the label is 0. The pixels of an image use the gray scale ranging from 0 to 1 and images are present in PGM format. 
For this implementation a neural network with 1 hidden layer with 100 neurons is used. The learning rate is 0.1 and logistic regression sigmoid function is used with 1000 training epochs. The accuracy of the network is 96.4%.
