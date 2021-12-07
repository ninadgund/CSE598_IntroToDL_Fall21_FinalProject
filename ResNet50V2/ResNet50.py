import numpy as np
import tensorflow as tf
from keras.applications.resnet_v2 import ResNet50V2
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from skimage import transform

# set images path
train_path = 'D:\\Chris\\CSE598\\Project\\dog-breed-identification\\train'
test_path = 'D:\\Chris\\CSE598\\Project\\dog-breed-identification\\test'


# Read the csv files
train_labels = pd.read_csv('D:\\Chris\\CSE598\\Project\\dog-breed-identification\\labels.csv')
test_labels = pd.read_csv('D:\\Chris\\CSE598\\Project\\dog-breed-identification\\sample_submission.csv')

print(train_labels.head())

#in train_labels ids are not in jpg format, hence converting them to jpg
train_labels['id'] = train_labels['id'].apply(lambda x: x + ".jpg")
test_labels['id'] = test_labels['id'].apply(lambda x: x + ".jpg")

fig = plt.figure(figsize=(18,4))
cp = sns.countplot(x = 'breed', data = train_labels)
plt.xticks(
    rotation=-90, 
    horizontalalignment='left',
    fontweight='light',
    fontsize='x-small'  
)
spacing = 0.4
fig.subplots_adjust(bottom=spacing)
plt.show()

#train_datagen is used for feature scaling and image augmentation (image augmentation is applied to avoid overfitting).
train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, 
                                   width_shift_range = 0.2, height_shift_range = 0.2, rotation_range = 20, brightness_range=[0.2,1.0])

#defining training set, here size of image is reduced to 224x224, batch of images is kept as 32 and class is defined as 'categorical'.
train_set = train_datagen.flow_from_dataframe(dataframe = train_labels, directory = train_path, x_col = "id", y_col = "breed",
                                                 batch_size = 16, subset="training", class_mode = "categorical", target_size = (224,224),
                                                 seed = 42, shuffle = True)

#defining validation set, here size of image is reduced to 224x224, batch of images is kept as 32 and class is defined as 'categorical' and subset as 'validation'.
validate_set = train_datagen.flow_from_dataframe(dataframe = train_labels, directory = train_path, x_col = "id", y_col = "breed",
                                                 batch_size = 16, subset="validation", class_mode = "categorical", target_size = (224,224),
                                                 seed = 42, shuffle = True)

#only rescaling is applied
test_datagen = ImageDataGenerator(rescale=1./255)

#defining test set
test_set = test_datagen.flow_from_dataframe(dataframe = test_labels, directory = test_path, x_col = "id", y_col = None,
                                                 batch_size = 16, class_mode = None, seed = 42, shuffle=False, target_size = (224,224))


#defining resnet50v2
resnet = ResNet50V2(input_shape = [224,224,3], weights = 'imagenet', include_top = False)

#we have to take pre trained weights, we don't want to train the weights again
for layer in resnet.layers:
    layer.trainable = False

#flattening the output of resnet for fully connected layer.
x = keras.layers.Flatten()(resnet.output)

#dropping out 40% to avoid overfitting
x = keras.layers.Dropout(0.4)(x)

#final layer will have 120 neurons
pred = keras.layers.Dense(120, activation='softmax')(x)

#creating the model 
model = tf.keras.models.Model(inputs=resnet.input, outputs=pred)

#set learning rate as 0.00001
opt = tf.keras.optimizers.Adam(learning_rate = 1e-5)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

#defining train step and validate step (total value/batch size)
train_step = train_set.n//train_set.batch_size
validate_step = validate_set.n//validate_set.batch_size

#training model
resnet50 = model.fit(train_set,validation_data = validate_set,epochs = 30,steps_per_epoch = train_step, validation_steps = validate_step)

#evaluating model
model.evaluate(validate_set)

#predicting output
STEP_SIZE_TEST=test_set.n//test_set.batch_size

#if we don't reset we'll get output in an unordered complex manner
test_set.reset()

pred=model.predict(test_set,steps=STEP_SIZE_TEST,verbose=1)

plt.plot(resnet50.history['accuracy'])
plt.plot(resnet50.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(resnet50.history['loss'])
plt.plot(resnet50.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#pred will be only probabilities, therefore takeing class of highest probability.
predicted_class_indices=np.argmax(pred,axis=1)

#labels include class indices of train set
labels = (train_set.class_indices)

#creating a dictionary
labels = dict((v,k) for k,v in labels.items())

#predictions include names of predictions instead of number
predictions = [labels[k] for k in predicted_class_indices]


imgpath = input("Enter an image path to predict: ")

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (224, 224, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

# input_image = load('D:\\Chris\\CSE598\\Project\\dog-breed-identification\\test\\00c14d34a725db12068402e4ce714d4c.jpg')
input_img = load(imgpath)

pimg = model.predict(input_img)

predict_img=np.argmax(pimg,axis=1)
predict_img_label = [labels[predict_img[0]]]
print(predict_img_label)