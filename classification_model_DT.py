import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

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


from sklearn import tree

model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)
accuracy =model.score(X_test,y_test)
print(f'Test Accuracy_MLP: {accuracy:.2f}')



y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int) 


report = classification_report(y_test, y_pred, target_names=['Label 0', 'Label 1'])
print("Classification Report:\n", report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)


with open("model_report_DT.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {accuracy:.2f}\n")
    
    
    f.write("\nClassification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))