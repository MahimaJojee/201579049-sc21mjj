# Code Repository of MSc Project

## Developing a Prototype Heart Disease Risk Assessment Tool for the OpenMRS System

This repository contains all the code that was involved the building of the prototype of a heart disease risk assessment tool.

### The development involved three main parts:
1. A heart disease Open Web App inside the OpenMRS system
2. A machine learning model that is pretrained and that predicts whether the person has a heart disease or not
3. A Flask wrapper that takes the OpenMRS patient data and gives it to the machine learning model and displays the results of the risk assessment.

The code for all the above three applications along with the dataset used to train the machine learning model is present in this repository under the code and dataset folders respectively.

The user interface that the OpenMRS user navigates through is as shown below:

1. On logging into the system

    <img src="https://user-images.githubusercontent.com/66803507/187025729-ad2736c0-352f-4a3b-b53d-c87d02c76dc6.png" width="800" height="450">

2. The heart disease risk assessment tool along with functionality to search patients from database

     <img src="https://user-images.githubusercontent.com/66803507/187025756-347e1087-f9b1-444c-b22a-f1f645b9f036.png" width="800" height="450">

3. The OpenMRS user serached for the patient and gets some of the fields filled from the database and th rest as entered by the user and clicks the predict risk of heart disease button.

    <img src="https://user-images.githubusercontent.com/66803507/187025831-5425ed91-9b63-4805-b788-ff6a8a25b535.png" width="800" height="450">

4. The results of the risk assessment is displayed to the user on a new page.

     <img src="https://user-images.githubusercontent.com/66803507/187025877-caf76569-06ae-4120-bdca-f04871974be2.png" width="800" height="450">


