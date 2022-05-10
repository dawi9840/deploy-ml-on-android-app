from dataset.data import DataPreprocessing
from TF_model.model import SequentialModel
from matplotlib import pyplot as plt
import random
import numpy as np
import tensorflow as tf
import pandas as pd


def visualize_mnist_result(title_name:str, predictions, test_images, test_labels):
    '''As the model output 10 float representing the probability of 
    the input image being a digit from 0 to 9, we need to find 
    the largest probability value to find out which digit the model predicts to 
    be most likely in the image.'''

    # Although our model is relatively simple, we were able to achieve 
    # good accuracy around 98% on new images that the model has never seen before. 
    def get_label_color(val1, val2):
        # A helper function that returns 'red'/'black' depending on 
        # if its two input parameter matches or not.
        if val1 == val2:
            return 'black'
        else:
            return 'red'

    # predict_digits_row = np.argmax(predictions, axis=0)    # 豎著比較，返回列號
    predict_digits_column = np.argmax(predictions, axis=1)   # 橫著比較，返回列號
    # print(f'Prediction digits (column): {predict_digits_column}\n')

    # Then plot 100 random test images and their predicted labels.
    # If a prediction result is different from the label provided label in "test"
    # dataset, we will highlight it in red color.
 
    plt.figure(num=title_name, figsize=(19,19))

    for i in range(100):
        ax = plt.subplot(10, 10, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image_index = random.randint(0, len(predict_digits_column))
        plt.imshow(test_images[image_index], cmap=plt.cm.gray)
        ax.xaxis.label.set_color(get_label_color(predict_digits_column[image_index], test_labels[image_index]))
        plt.xlabel('Predict: %d' % predict_digits_column[image_index])
    plt.show()


def convert_to_TFLite_model(model, output_file:str):
    '''Convert model to TF Lite model, which returns float and quantized version tflite model.'''
    def save_tf_model(out_file:str, tfmodel, tips:str):
        # Save the tflite model to file to the directory.
        f = open(out_file, "wb")
        f.write(tfmodel)
        f.close()
        print(tips)

    # Convert Keras model to TF Lite format.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float_model = converter.convert()

    # Show model size in KBs.
    float_model_size = len(tflite_float_model) / 1024
    print('\n' + '*'*20)
    print(f'Float model size: {float_model_size} KBs.')
    save_tf_model(out_file='orignal_'+output_file,tfmodel=tflite_float_model, tips='Save float ftlite model done!\n')

    # Here we will use 8-bit number to approximate 
    # our 32-bit weights, which in turn shrinks the model size by a factor of 4.

    # Re-convert the model to TF Lite using quantization.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    # Show model size in KBs.
    quantized_model_size = len(tflite_quantized_model) / 1024
    reduction_size = quantized_model_size * 100 / float_model_size
    print('\n' + '*'*20)
    print(f'Quantized model size: {quantized_model_size} KBs')
    print(f'which is about {reduction_size} of the float model size.')
    save_tf_model(out_file='quantizied_'+output_file, tfmodel=tflite_quantized_model, tips='Save quantized ftlite model done!\n')

    return tflite_float_model, tflite_quantized_model


def convert_2_TFLite_model(model, output_file:str):
    '''Convert model to TF Lite model, and save quantized version tflite model.'''
    def save_tf_model(out_file:str, tfmodel, tips:str):
        # Save the tflite model to file to the directory.
        f = open(out_file, "wb")
        f.write(tfmodel)
        f.close()
        print(tips)

    # Convert Keras model to TF Lite format.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Here we will use 8-bit number to approximate 
    # our 32-bit weights, which in turn shrinks the model size by a factor of 4.

    # Re-convert the model to TF Lite using quantization.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()
    
    save_tf_model(
        out_file='quantizied_'+output_file, 
        tfmodel=tflite_quantized_model, 
        tips='Save quantized ftlite model done!\n')

    return tflite_quantized_model


def evaluate_tflite_model(tflite_model_content, test_images, test_labels):
    '''To evaluate the TF Lite model using "test" dataset.'''
    # Initialize TFLite interpreter using the model.
    interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_tensor_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest probability.
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == test_labels[index]:
            accurate_count += 1

    accuracy = accurate_count * 1.0 / len(prediction_digits)
    return accuracy


def trace_quantized_model(mnist_model, out_tflite, test_images, test_labels):
    folat_tflite_model, quantized_tflite_model = convert_to_TFLite_model(model=mnist_model, output_file=out_tflite)

    float_accuracy = evaluate_tflite_model(tflite_model_content=folat_tflite_model, test_images=test_images, test_labels=test_labels)
    quantized_accuracy = evaluate_tflite_model(tflite_model_content=quantized_tflite_model, test_images=test_images, test_labels=test_labels)
    drop_accuracy = float_accuracy - quantized_accuracy

    print(f'Float model accuracy    : {float_accuracy}')
    print(f'Quantized model accuracy: {quantized_accuracy}')
    print(f'Accuracy drop: {drop_accuracy}')


def digits_model(data_processing, sequential_model):
    out_tflite = 'mnist.tflite'
    train_images, train_labels, test_images, test_labels = data_processing.MNIST_data()
    mnist_model = sequential_model.digits_model(train_images, train_labels, fit_epochs=60)

    # Evaluate the model using all images in the test dataset.
    test_loss, test_acc = mnist_model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')

    # Predict the labels of digit images in our test dataset.
    predictions = mnist_model.predict(test_images)
    visualize_mnist_result(title_name='predict_digits', predictions=predictions, test_images=test_images, test_labels=test_labels)

    # Convert to tflite model.
    # trace_quantized_model(mnist_model, out_tflite, test_images, test_labels)
    tflite_quantized_model = convert_2_TFLite_model(model=mnist_model, output_file=out_tflite)


def iris_model(data_processing, sequential_model):
    src_data = pd.read_csv('./dataset/iris.data')
    out_tflite = 'iris.tflite'

    x, y = data_processing.iris_data(src_data)
    iris_model = sequential_model.iris_model(X=x, y=y)
    convert_2_TFLite_model(iris_model, out_tflite)


if __name__ == '__main__':
    data_processing = DataPreprocessing()
    sequential_model = SequentialModel()

    digits_model(data_processing, sequential_model)
    iris_model(data_processing, sequential_model)