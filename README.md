# deploy-ml-on-android-app
Example of deploying tflite model on android application.

Part1.兩種方式取得tflite    
Part2.將tflite 模型布署至 android app上    

# Part1.兩種方式取得tflite 
Usage1: 自行創建Tflite file   
  Create ml model with python   
  Quantized model -> convert tflite   

Usage2: 從TF Hub下載 Tflite nodel
  Download tflite model from TensorFlow Hub: E.g., [aiy/vision/classifier/birds_V1](https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3)


## Install     

**Conda virtual env**  
```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install -r requirements.txt
```   
