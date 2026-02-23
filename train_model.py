import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ===============================
# 1️⃣ Load All CSV Files
# ===============================

dataset_path = "dataset"

if not os.path.exists(dataset_path):
    print("❌ Dataset folder not found!")
    exit()

all_data = []

for label in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label)

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path, header=None)
            data['label'] = label
            all_data.append(data)

if len(all_data) == 0:
    print("❌ No CSV data found!")
    exit()

dataset = pd.concat(all_data, ignore_index=True)

# ===============================
# 2️⃣ Prepare Data
# ===============================

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# ===============================
# 3️⃣ Build Model
# ===============================

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(63,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 4️⃣ Train Model
# ===============================

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ===============================
# 5️⃣ Save Model
# ===============================

model.save("sign_model.keras")


print("🎉 Model trained successfully!")
