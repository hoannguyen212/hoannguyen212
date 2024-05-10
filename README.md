import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Hàm cân bằng màu sắc của ảnh
def balance_color(image):
    balanced_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    balanced_image[:, :, 0] = cv2.equalizeHist(balanced_image[:, :, 0])
    balanced_image = cv2.cvtColor(balanced_image, cv2.COLOR_LAB2BGR)
    return balanced_image

# Hàm load dữ liệu từ thư mục
def load_data_from_directory(directory, target_size=(32, 32)):
    images = []
    labels = []
    label_names = []

    for label_name in os.listdir(directory):
        label_dir = os.path.join(directory, label_name)
        if os.path.isdir(label_dir):
            label_names.append(label_name)
            label = len(label_names) - 1  

            for filename in os.listdir(label_dir):
                img_path = os.path.join(label_dir, filename)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    img = balance_color(img)  # Cân bằng màu sắc của ảnh
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(label)

    return np.array(images), np.array(labels), label_names

# Đường dẫn tới thư mục chứa dữ liệu
dataset_directory = "D:\\AI\\nhandienlogo\\logo"
images, labels, label_names = load_data_from_directory(dataset_directory)

# Tiền xử lý dữ liệu
x_train = images / 255.0
y_train = labels

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Chuyển đổi nhãn thành one-hot encoding
y_train = to_categorical(y_train, len(label_names))
y_test = to_categorical(y_test, len(label_names))

# Xây dựng mô hình
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    Conv2D(64, (3,3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(len(label_names)),
    Activation('softmax')
])

# Biên dịch mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Huấn luyện mô hình
epochs = 50
history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, validation_data=(x_test, y_test))

# Lưu mô hình
model.save('logoepls.keras')
