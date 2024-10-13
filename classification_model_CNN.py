import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
data = pd.read_csv("file_data.csv", on_bad_lines="skip")


data = data.dropna(subset=['Hex Code'])


def hex_to_int_list(hex_string):
    return [int(hex_string[i:i + 2], 16) for i in range(0, len(hex_string), 2)]


data['Hex List'] = data['Hex Code'].apply(hex_to_int_list)

MAX_LENGTH = 40000  


X = pad_sequences(data['Hex List'].tolist(), maxlen=MAX_LENGTH, padding='post', truncating='post')


y = data['Label'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


model = Sequential()
model.add(Embedding(input_dim=256, output_dim=50, input_length=MAX_LENGTH))  


model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))  
model.add(MaxPooling1D(pool_size=2))  
model.add(Dropout(0.5)) 

model.add(Flatten())


model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(1, activation='sigmoid'))  


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')
model.summary()


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) 

report = classification_report(y_test, y_pred, target_names=['Label 0', 'Label 1'])
print("Classification Report:\n", report)


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


with open("model_report_cnn.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {accuracy:.2f}\n")
    f.write("Model Summary:\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write("\nClassification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))