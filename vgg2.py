from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np

model_vgg16_conv = VGG16(weights= None , include_top=False)

model_vgg16_conv.summary()

input = Input(shape = (224,224,3), name = 'image_input')

output_vgg16 = model_vgg16_conv(input)

x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(2, activation='softmax', name='predictions')(x)

my_model = Model(input=input, output=x)

my_model.summary()

#adam optimiser with cross entropy
mymodel.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#data as cat and dog classification challenge - preprocessed

mymodel.fit(data, one_hot_labels, epochs=10, batch_size=32)
