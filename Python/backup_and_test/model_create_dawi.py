import pandas as pd
from tensorflow import lite
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def data_preprocessing(src_data:str):
    df = src_data

    # Get 0 ~ 4 columns value. However 0 column have no values.
    X = df.iloc[:, :4].values

    # Get column 5 values: 'Iris-setosa'
    y = df.iloc[:, 4].values    

    # 把每個類別 mapping 到某個整數，不會增加新欄位
    # https://medium.com/@PatHuang/%E5%88%9D%E5%AD%B8python%E6%89%8B%E8%A8%98-3-%E8%B3%87%E6%96%99%E5%89%8D%E8%99%95%E7%90%86-label-encoding-one-hot-encoding-85c983d63f87
    le = LabelEncoder() 

    # 字串無法套入數學模型進行運算，在此先對其進行Label encoding編碼.
    # 本質上是類別資料，並沒有順序大小之分，當我們對其進行Label encoding後，
    # 模型會認為他們之間存在著 0<1<2，因此這種無序的離散值One hot encoding會更加的合適。
    y = le.fit_transform(y)

    # Converts a class vector (integers) to binary class matrix.
    y = to_categorical(y)
    return X, y


def sequential_model(X, y):
    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=[4])) # shape: 4 -> 0,1,2,3,4 columns
    model.add(Dense(64))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
    model.fit(X, y, epochs=200)

    print(model.summary())
    return model


def convert_tflite(model, output_file:str):
    # Conver model and save tflite model.
    converter = lite.TFLiteConverter.from_keras_model(model)

    tfmodel = converter.convert()

    open(output_file, 'wb').write(tfmodel)
    print('\nConvert tflite model done!')


if __name__ == '__main__':

    src_data = pd.read_csv('./dataset/iris.data')
    out_tflite = './tflite_model/iris.tflite'

    x, y = data_preprocessing(src_data)
    iris_model = sequential_model(X=x, y=y)
    convert_tflite(model=iris_model, output_file=out_tflite)