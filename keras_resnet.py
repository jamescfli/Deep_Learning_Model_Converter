from keras.applications import resnet50, vgg16
from keras.utils.visualize_util import plot

resnet50_keras = resnet50.ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
resnet50_keras.summary()

# plot(resnet50_keras, './Keras/visualized/resnet50_notop_keras_graph.png', show_shapes=True, show_layer_names=True)
plot(resnet50_keras, to_file='./Keras/visualized/resnet50_wtop_keras_graph.png', show_shapes=True, show_layer_names=True)
