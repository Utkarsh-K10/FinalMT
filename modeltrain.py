from pickletools import optimize
import pandas as pd
import keras , glob
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.model import Model
from keras.preprocessing.image import ImageDatagenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocessinginput
from keras.models import Sequential
from keras.optimizers import SGD

IMAGE_SIZE = [224, 224]

train_path = '/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train'
valid_path = '/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Test'

vggmodel = VGG16(input_shape = IMAGE_SIZE+[3], weights = "resnet", include_top = 'False')

for layer in vggmodel.layers:
    layer.trainable = False

folders = glob('/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train/*')
x = Flatten()(vggmodel.input)
x = Dropout(rate = 0.5)(x)
prediction = Dense(unit = len(folders), activation = "softmax")(x)

finalmodel = Model(inputs = vggmodel.input, outputs = prediction)
finalmodel.summary()
opt = SGD(learning_rate = 0.1)
finalmodel.compile(
    loss= "categorical_crossentropy",
    optimizer = opt,
    metrics = ['accuracy'] 
)

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train',target_size = (224, 224),batch_size = 32,class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


finalmodel.fit(x, y, epoch = 21, steps_per_epoch = len(training_set))

def createModel(vggmodel, folderpath):
    for layer in vggmodel.layers:
        layer.trainable = False
    folder = folderpath
    x = Flatten()(vggmodel.input)
    x = Dropout(rate = 0.5)(x)
    prediction = Dense(unit = len(folders), activation = "softmax")(x)
    finalmodel = Model(inputs = vggmodel.input, outputs = prediction)
    finalmodel.summary()
    return finalmodel