__author__ = 'bsl'

import keras.caffe.convert as convert
from keras.applications import vgg16
from keras.layers import Input
from keras.utils.visualize_util import plot


def load_vgg16_notop_from_caffemodel():

    prototxt = 'Caffe/prototxt/train_vgg16_places365_wtop.prototxt'
    caffemodel = 'Caffe/weights/vgg16_places365_wtop.caffemodel'

    model_loaded_from_caffe = convert.caffe_to_keras(prototxt, caffemodel, debug=False)

    # prepare a vgg model_loaded_from_caffe without top layers
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)
    input_tensor = Input(batch_shape=(None,) + img_size)
    model_vgg16_places365_notop = vgg16.VGG16(input_tensor=input_tensor,
                                              include_top=False)

    model_vgg16_places365_notop.get_layer('block1_conv1')\
        .set_weights(model_loaded_from_caffe.get_layer('conv1_1').get_weights())
    model_vgg16_places365_notop.get_layer('block1_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv1_2').get_weights())
    model_vgg16_places365_notop.get_layer('block2_conv1') \
        .set_weights(model_loaded_from_caffe.get_layer('conv2_1').get_weights())
    model_vgg16_places365_notop.get_layer('block2_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv2_2').get_weights())
    model_vgg16_places365_notop.get_layer('block3_conv1') \
        .set_weights(model_loaded_from_caffe.get_layer('conv3_1').get_weights())
    model_vgg16_places365_notop.get_layer('block3_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv3_2').get_weights())
    model_vgg16_places365_notop.get_layer('block3_conv3') \
        .set_weights(model_loaded_from_caffe.get_layer('conv3_3').get_weights())
    model_vgg16_places365_notop.get_layer('block4_conv1') \
        .set_weights(model_loaded_from_caffe.get_layer('conv4_1').get_weights())
    model_vgg16_places365_notop.get_layer('block4_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv4_2').get_weights())
    model_vgg16_places365_notop.get_layer('block4_conv3') \
        .set_weights(model_loaded_from_caffe.get_layer('conv4_3').get_weights())
    model_vgg16_places365_notop.get_layer('block5_conv1') \
        .set_weights(model_loaded_from_caffe.get_layer('conv5_1').get_weights())
    model_vgg16_places365_notop.get_layer('block5_conv2') \
        .set_weights(model_loaded_from_caffe.get_layer('conv5_2').get_weights())
    model_vgg16_places365_notop.get_layer('block5_conv3') \
        .set_weights(model_loaded_from_caffe.get_layer('conv5_3').get_weights())

    # rename the model
    model_vgg16_places365_notop.name = 'vgg16_notop_places365_model'

    return model_vgg16_places365_notop

if __name__ == "__main__":
    # verify change to Places365, for comparison
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)
    input_tensor = Input(batch_shape=(None,) + img_size)
    model_vgg_imagenet_notop = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # check weight change from imagenet
    model_vgg_places365_notop = load_vgg16_notop_from_caffemodel()
    from compare_model_parameters import equal_model
    print 'places365 vgg and imagenet vgg are : ' \
          + ('the same' if equal_model(model_vgg_imagenet_notop, model_vgg_places365_notop) else 'different')
