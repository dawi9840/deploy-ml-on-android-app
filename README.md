# deploy-ml-on-android-app
Example of deploying tflite model on android application.   



# Part1.兩種方式取得tflite 
方式1: 自行創建Tflite file   
  Create ml model with python   
  Quantized model -> convert tflite   

方式2: 從TF Hub 官方下載 Tflite nodel
  Download tflite model from TensorFlow Hub: E.g., [aiy/vision/classifier/birds_V1](https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3)   


# Part2.tflite deploy 至 android studio上使用![image](https://user-images.githubusercontent.com/19554347/167744174-4ba64076-fec3-4e1c-8f54-5e00861fffce.png)
1.前置作業: dependencies setting on android
2.Deploy tflite on Android app:
  2-1: 用Import TensorFlow Lite model 方式
  2-2: 用interpreter 呼叫 tflite model

***1.前置作業: dependencies setting on android***   
In gradle:app file:
1.With android{} block to add aaptOptions{} block. The purpose is to not compress the input and output files of tflite file.
![image](https://user-images.githubusercontent.com/19554347/167754536-9425d678-ccc0-42ad-8f16-21faa09801fa.png)   

2.With dependencies{} block to add tflite dependencies.   
![image](https://user-images.githubusercontent.com/19554347/167754733-c60f2e8c-c5ef-4e7d-9e81-7643ebd3da19.png)   

***2.Deploy tflite on Android app***   
2-1: Import TensorFlow Lite model
Step1: To create a new ml model to import tlite model In app folder.   
![image](https://user-images.githubusercontent.com/19554347/167755002-94d39cdd-7dde-446a-a3ed-261b6fc504fd.png)   

Step2: Then the imported tflite file will be placed in the automatically generated ml folder. 
![image](https://user-images.githubusercontent.com/19554347/167755361-7d096ce2-b72a-4a97-96f4-0a3cf65590a1.png)   

Click on one of the models, you will find sample code to know how to invoke.   
![image](https://user-images.githubusercontent.com/19554347/167756290-84f8fdc0-00f6-4dba-936a-5571fb8a6639.png)   
E.g., invoke usage:   
![image](https://user-images.githubusercontent.com/19554347/167756364-addd8226-033c-4210-8e10-70924f8f5e31.png)

2-2: With interpreter invoke tflite model   
Add an assert folder to the app, and then manually put the tflite file into it.   
![Untitled](https://user-images.githubusercontent.com/19554347/167756527-b01ad3b5-10b6-433e-9002-ee35c61c090a.png)   
E.g., invoke usage:   
![ww](https://user-images.githubusercontent.com/19554347/167757452-5a15805f-9ba9-4fbf-a859-d9bb24be426a.png)   

My points:   
Advantages and disadvantages of two ways to invoke tflite model on Android:   

***Use 2-1 method way to import model: ***   
Advantages: Could use the sample code for easy use to invoke model.   
Disadvantages: Only use the default CPU memory on android to invoke model.   


***Use 2-2 method way to invoke model: ***   
Advantages: Could use interpreter GpuDelegate method to invoke the tflite model with the GPU on Android.  e.g., [TensorFlow Lite PoseNet Android Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/android)   
Disadvantages: There is no sample code to tip, how to use the tflite model, which means we have to study the input and output format of the model obtained by yourself to invoke.










## Install     

**Conda virtual env**  
```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install -r requirements.txt
```   
