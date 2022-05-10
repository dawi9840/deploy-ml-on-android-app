from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow import keras

class DataPreprocessing:
    def iris_data(self, src_data:str):
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

    def MNIST_data(self, ):
        '''Download the MNIST dataset from keras.
        Returns train and label dataset.'''
        
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize the input image so that each pixel value is between 0 to 1.
        train_images = train_images / 255.0
        test_images = test_images / 255.0
        print('Pixels are normalized done!\n')
        self.visualization_top25_datas(title_name='tOP 25 train dataset', images=train_images, labels=train_labels)
        return train_images, train_labels, test_images, test_labels
    
    def visualization_top25_datas(self, title_name:str, images, labels):
        # Show the first 25 images in the training dataset.
        for i in range(25):
            plt.figure(num=title_name, figsize=(10,10))
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(images[i], cmap=plt.cm.gray)
            plt.xlabel(labels[i])
            plt.show()