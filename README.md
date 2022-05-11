# deploy-ml-on-android-app
Example of training a tensorflow model and deploying tflite model on android application from scratch.  

## The flowchart:
 ![tflite_model_deploy_to_android drawio](https://user-images.githubusercontent.com/19554347/167766895-e3340be6-7793-4171-96d3-a4615c186ddb.png)   
  




## Part1: Get the tflite model of two ways on PC   
**Method1: Create Tflite model by yourself**   
*  Create ml model with python   
*  Quantized model -> convert tflite   

Usage: Run the code and will get the example tflite model.  
```bash
 python main.py
```     

**Method2: Download Tflite model from TensorFlow Hub**   
Download tflite model from TensorFlow Hub: E.g., [aiy/vision/classifier/birds_V1](https://tfhub.dev/google/lite-model/aiy/vision/classifier/birds_V1/3)    
  
   ![Screenshot 2022-05-11 140510](https://user-images.githubusercontent.com/19554347/167780010-ebc1bc1a-4bfa-472d-99ab-94e503c2d762.png)    



## Part2: Deploy tflite model on android studio      
1.Pre-work: dependencies setting on android   
2.Two ways of deploying tflite on Android app:   
*  2-1: Import TensorFlow Lite model      
*  2-2: With interpreter invoke tflite model   
-----------------------------------------------------------      
***1.Pre-work: dependencies setting on android***   
*  In the gradle:app file:   
Step1: With android{} block to add aaptOptions{} block. The purpose is to not compress the input and output files of tflite file.   
![image](https://user-images.githubusercontent.com/19554347/167754536-9425d678-ccc0-42ad-8f16-21faa09801fa.png)   
Step2: With dependencies{} block to add tflite dependencies.   
![image](https://user-images.githubusercontent.com/19554347/167754733-c60f2e8c-c5ef-4e7d-9e81-7643ebd3da19.png)   

***2.Deploy tflite on Android app***   
*  2-1: Import TensorFlow Lite model
Step1: To create a new ml model to import tlite model In app folder.   
![image](https://user-images.githubusercontent.com/19554347/167755002-94d39cdd-7dde-446a-a3ed-261b6fc504fd.png)   

Step2: Then the imported tflite file will be placed in the automatically generated ml folder. 
![image](https://user-images.githubusercontent.com/19554347/167755361-7d096ce2-b72a-4a97-96f4-0a3cf65590a1.png)   

Click on one of the models, you will find sample code to know how to invoke.   
![image](https://user-images.githubusercontent.com/19554347/167756290-84f8fdc0-00f6-4dba-936a-5571fb8a6639.png)   
E.g., invoke usage:   
![image](https://user-images.githubusercontent.com/19554347/167756364-addd8226-033c-4210-8e10-70924f8f5e31.png)

*  2-2: With interpreter invoke tflite model   
Add an assert folder to the app, and then manually put the tflite file into it.   
![Untitled](https://user-images.githubusercontent.com/19554347/167756527-b01ad3b5-10b6-433e-9002-ee35c61c090a.png)   
E.g., invoke usage:   
![ww](https://user-images.githubusercontent.com/19554347/167757452-5a15805f-9ba9-4fbf-a859-d9bb24be426a.png)   

   
## Advantages and disadvantages of two ways to invoke tflite model on Android:   

**Use 2-1 method way to import model:**   
Advantages: Could use the sample code for easy use to invoke model.   
Disadvantages: Only use the default CPU memory on android to invoke model.   


**Use 2-2 method way to invoke model:**   
Advantages: Could use interpreter GpuDelegate method to invoke the tflite model with the GPU on Android.  E.g., [TensorFlow Lite PoseNet Android Demo.](https://github.com/tensorflow/examples/tree/master/lite/examples/posenet/android)   
Disadvantages: There is no sample code to tip, how to use the tflite model, which means we have to study the input and output format of the model obtained by yourself to invoke.   


## Install     

**Conda virtual env**  
```bash

conda create --name [env_name]  python=3.8
conda activate [env_name]
pip install -r requirements.txt
```   
