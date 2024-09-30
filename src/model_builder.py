from keras.layers import Dense,Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras

base_model = VGG19(input_shape=(256,256,3), include_top= False)

for layer in base_model.layers:
  layer.trainable = False

base_model.summary()

x= Flatten()(base_model.output)

x=Dense(units =15 ,activation='softmax')(x)

#creating our model
model= Model(base_model.input,x)

model.summary()

model.compile(optimizer= 'adam'  , loss = keras.losses.categorical_crossentropy , metrics =['accuracy'])
