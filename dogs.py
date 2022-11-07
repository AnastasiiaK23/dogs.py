import io
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
from google.colab import files

SIZE = (224, 224)

def resize_image(img, label):
	img = tf.cast(img, tf.float32)
	img = tf.image.resize(img, SIZE)
	img /= 255.0
	return img, label      

train_resized = train[0].map(resize_image)
train_batches = train_resized.shuffle(1000).batch(16)

# Создание основного слоя для создания модели
base_layers = tf.keras.applications.MobileNetV2(input_shape=(SIZE[0], SIZE[1], 3), include_top=False)

# Создание модели нейронной сети
model = tf.keras.Sequential([
	base_layers,
	GlobalAveragePooling2D(),
	Dropout(0.2),
	Dense(1)
])
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_batches, epochs=1)

def load_image():
    uploaded_file = st. file_uploader(label= "Виберіть зображення для розпізнання")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st. image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

images = ['1.jpg','4.jpg', '3.jpg', '5.jpg', '2.jpg', '6.jpg']

# Перебираем все изображения и даем нейронке шанс определить что находиться на фото
for i in images:
	img = load_img(i)
	img_array = img_to_array(img)
	img_resized, _ = resize_image(img_array, _)
	img_expended = np.expand_dims(img_resized, axis=0)
	prediction = model.predict(img_expended)
	plt.figure()
	plt.imshow(img)
	label = 'Собачка' if prediction > 0 else 'Кошка'
	plt.title('{}'.format(label))

st.title( "Класифікувати зображення")
img = load_image()
result = st.button('Розпізнати зображення')
if result:
    x = preprocess_image(ing)
    preds = model.predict(x)
    st.write('Результат розпізнання:')
    print_predictions (preds)
