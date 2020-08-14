import os
import random
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from keras.regularizers import l2
os.environ["CUDA_VISIBLE_DEVICES"]="6"

W_reg = 1e-4
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(150,150,3),kernel_regularizer=l2(W_reg)),
    tf.keras.layers.MaxPooling2D(2,2),
    ## Second layer
    tf.keras.layers.Conv2D(64,(3,3), activation='relu',kernel_regularizer=l2(W_reg)),
    tf.keras.layers.MaxPooling2D(2,2),
    ## Third layer
    tf.keras.layers.Conv2D(128,(3,3), activation='relu',kernel_regularizer=l2(W_reg)),
    tf.keras.layers.MaxPooling2D(2,2),
    ## Fourth layer
    tf.keras.layers.Conv2D(256,(3,3), activation='relu',kernel_regularizer=l2(W_reg)),
    tf.keras.layers.MaxPooling2D(2,2),
    ###Flatten
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=l2(W_reg)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64,activation='tanh'),
    tf.keras.layers.Dense(1,activation='sigmoid'),
# YOUR CODE HERE
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
TRAINING_DIR = '/home/siddharth.soni/cats_and_dogs/Training/'
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
	#	width_shift_range=0.1, height_shift_range=0.1, 
		#rotation_range=40,
        	#shear_range=0.2,
        	#zoom_range=0.2,
        #	horizontal_flip=True,
       	#	fill_mode='nearest')

# TRAIN GENERATOR.
train_generator = train_datagen.flow_from_directory(
                    TRAINING_DIR,
                    batch_size=30,
                    target_size=(150,150),
                    class_mode='binary'
                    )

VALIDATION_DIR = '/home/siddharth.soni/cats_and_dogs/Validation/'
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

validation_generator = validation_datagen.flow_from_directory(
                        VALIDATION_DIR,
                        batch_size=30,
                        target_size=(150,150),
                        class_mode='binary')
# number of images is 5400
history = model.fit(train_generator,steps_per_epoch=12000//30, 
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator,validation_steps=1200//30)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
model.save('/home/siddharth.soni/cats_and_dogs/cats_and_dogs.h5')
print(acc)
print(val_acc)

f = open('history.pkl', 'wb')
pickle.dump(history.history, f)
f.close()
