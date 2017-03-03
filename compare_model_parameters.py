__author__ = 'bsl'

from keras.applications import vgg16
from keras.layers import Input
import numpy as np

def equal_model(model1, model2):
    # model1 and model2 should have the same shape and weight size
    # no more comparison if have different number of layers
    assert model1.layers.__len__() == model2.layers.__len__(), 'models have diff nb of layers'
    nb_layers = model1.layers.__len__()
    flag_same_model = True
    for i in np.arange(nb_layers):
        # o.w. skip this layer
        if not model1.layers[i].get_weights().__len__() == model2.layers[i].get_weights().__len__() == 0:
            assert model1.layers[i].get_weights().__len__() == model2.layers[i].get_weights().__len__(), \
                'diff number of weight arrays in layer {}'.format(i)
            assert model1.layers[i].get_weights()[0].shape == model2.layers[i].get_weights()[0].shape, \
                'diff W parameter shape in layer {}'.format(i)
            assert model1.layers[i].get_weights()[1].shape == model2.layers[i].get_weights()[1].shape, \
                'diff b parameter shape in layer {}'.format(i)
            if not (np.array_equal(model1.layers[i].get_weights()[0], model2.layers[i].get_weights()[0])
                and np.array_equal(model1.layers[i].get_weights()[1], model2.layers[i].get_weights()[1])):
                flag_same_model = False
                break
    return flag_same_model


if __name__ == "__main__":
    img_width = 224
    img_height = 224
    img_size = (3, img_width, img_height)

    input_tensor = Input(batch_shape=(None,) + img_size)

    model_vgg_default_notop = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    model_vgg_imagenet_notop = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    store_path = '../../pretrain/models/'
    model_vgg_places365_notop_loaded = vgg16.VGG16(input_tensor=input_tensor, include_top=False)
    model_vgg_places365_notop_loaded.load_weights(store_path+'vgg16_places365_notop.h5')    # default: by_name=False

    print 'default vgg and imagenet vgg are : ' \
          + ('the same' if equal_model(model_vgg_default_notop, model_vgg_imagenet_notop) else 'different')
    print 'places vgg and imagenet vgg are : ' \
          + ('the same' if equal_model(model_vgg_places365_notop_loaded, model_vgg_imagenet_notop) else 'different')
