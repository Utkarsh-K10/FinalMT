from keras.applications.vgg import VGG16
from keras.applications.resnet import ResNet152
from modeltrain import IMAGE_SIZE
from keras.model import Model
import glob
from keras.optimizer import SGD
from keras.layers import Dense,Flatten, Dropout
vggmodel = VGG16(input_shape = IMAGE_SIZE+[3], include_top = 'False', weights = 'resnet')
resnetv2 = ResNet152(input_shape = IMAGE_SIZE+[3], include_top = 'False', weights = 'resnet')

for layer in vggmodel.layers:
    layer.trainable = False

folders = glob('/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train/*')

x = Flatten()(vggmodel.input)    
x = Dropout(rate = 0.5)(x)
prediction = Dense(unit = len(folders), activation = 'softmax')(x)
finalmodel = Model(inputs = vggmodel.input, outputs = prediction)
opt = SGD(learning_rate = 0.003)
finalmodel.compile(
    optimizer = opt,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
