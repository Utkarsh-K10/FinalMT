# Face Recognition

# Importing the libraries
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd
from keras.preprocessing import image

model = load_model('/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/26aprlvgg.h5')
df = pd.read_csv('/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/27aprilusers.csv')
person_name =['Sahil Bansal', 'Swastik Sharma', 'Utkarsh Kushwaha', 'Yashika Dhote', 'kavita Chirgaiya']
# ['Abhay Thagle', 'Abhinav Bhargava', 'Abhishek Garg', 'Akanksha Shukla', 'Akash Hirodiya', 'Amisha Chirote', 'Amrita Porsiya', 'Anil Mandloi', 'Anita Lowanshi', 'Ankit Saini', 'Anshu Chourey', 'Anshul Sharma', 'Anshuman Tamrakar', 'Anuj Tripathi', 'Ashish Thakur', 'Ayushi Vishwkarma', 'Deepak Kewadia', 'Dipansha Bordiya', 'Dronacharya Pal', 'Gourav Solanki', 'Harshit Nema', 'Harshit Vyas', 'Jasmeet Kaur', 'Kavita Lodhi', 'Mahima Jhade', 'Mradul Gupta', 'Muskan Ranjan', 'Nidhi Rathod', 'Owais Khan', 'Pallavi Besh', 'Pooja Gupta', 'Prathmesh Bansal', 'Priya Patidar', 'Sampada Vyas', 'Satyam Nema', 
# 'Satyam Raghuwanshi', 'Shadab Alam', 'Shakshi Singh Bais', 'Snehal Maheskey', 'Sukanya Sinha', 'Utkarsh Kushwaha', 'Vivek Menon', 'YOGESH PATEL']

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# person_name = ['Abhay Thagle', 'Abhinav Bhargava', 'Abhishek Agrawal', 'Abhishek Garg', 'Akanksha Shukla', 'Akash Hirodiya', 'Anita Lowanshi', 'Nidhi Rathod', 'Utkarsh Kushwaha']

x1 = []
y1 = []
w1 = []
h1 = []


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]
        x1.append(x)
        y1.append(y)
        w1.append(w)
        h1.append(h)
    return cropped_face

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        #Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)             
        name="Not matching"
#predicting the model to get Predicted Person index
        # for i in pred[0]:
        #     if(i>0.5):
        i = max(pred[0])
        predicted_index = list(pred[0]).index(i)
        pred_person_index = predicted_index
        print(pred_person_index)

#Validating the predicted index with Our Database and getting the name
        pred_person_name = []
        for pname in person_name:
            if person_name.index(pname) == pred_person_index:
                pred_person_name.append(pname)

#Getting all Details of name matching Index from database
        student_detail = []
        print(pred_person_name)
        if len(pred_person_name)>0:
            if df[df['Name']== pred_person_name[0]].index.values:
                student_detail.append(df.loc[pred_person_index].values.tolist())
                # student_detail = df.loc[pred_person_index].values.tolist()
            print(student_detail)

#validating the detials and Storing into Dictionary to Print it with label
            if len(student_detail)>0:
                dict = {
                    "Name:":student_detail[0][4],
                    "Enroll:":student_detail[0][2],
                    "Mobile No:": student_detail[0][3],
                    "Adhar No:": student_detail[0][0],
                    "Email:":student_detail[0][1]
                    }
                loc = [(k,v) for k, v in dict.items()]
                for idx, lbl in enumerate(loc):
                    offset =40
                    cv2.rectangle(frame,(60,y1[0]+offset*idx),(450, y1[0]-40), (255,205,205),0)
                    cv2.putText(frame, str(lbl), (60,y1[0]+offset*idx), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame,"Unknown", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        else:
            cv2.putText(frame, pred_person_name[0], (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()       

#         for i in pred[0]:
#             if i>0.5:
#                 pred_index = list(pred[0]).index(i)
#         predPersonindex = pred_index

#         for person in person_name:
#             if person_name.index(person) == predPersonindex:
#                 cv2.putText(frame,person, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
#             # else:
#             #     cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

#         # if(pred[0][8]>0.5):
#         #     name='Utkarsh'
#         # cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
#     else:
#         cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()       


# import os
# import glob

# p = sorted( filter( os.path.isdir ,glob.glob('/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train/*') ) )
# p_names = []
# for i in p:
#     p_names.append(i.split('/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train/')[1])

# # print(os.listdir('/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train'))
# p_name = ['ARADHYA DIMRI  ','Abhay Thagle','Abhinav Bhargava','Abhishek Agrawal','Abhishek Garg','Akanksha Shukla','Akash Hirodiya','Akshat Tiwari','Amisha Chirote', 'Amrita Porsiya', 'Anil Mandloi','Anita Lowanshi','Ankit Saini','Anshu Chourey','Anshul Sharma','Anshuman Tamrakar','Anuj Tripathi','Anusha Pahariya','Ashish Thakur','Ayushi Vishwkarma','Deepak Kewadia','Dipansha Bordiya','Dronacharya Pal','Dwarka Soni','Gourav Solanki','Harshit Nema','Harshit Vyas','Jasmeet Kaur','Kavita Lodhi','Mahima Jhade','Mishika Shrivastava','Mradul Gupta','Muskan Ranjan','Nidhi Rathod','Owais Khan','Pallavi Besh','Prathmesh Bansal','Pratik Barche','Priya Patidar','Ridhi Mishra','Rohit Singh tomar','SHIVANI PARMAR','Sampada Vyas  ','Satyam Nema','Satyam Raghuwanshi','Shadab Alam','Shakshi Singh Bais','Snehal Maheskey','Sukanya Sinha','Utkarsh Kushwaha','Vikas Sitole','Vivek Menon','YOGESH PATEL']
# print('derived',p_name, '\n')
# print(p_names)

# def getpredIndex(predarr):
#     for i in predarr[0]:
#         if(i>0.5):
#             predicted_index = list(predarr[0].index(i))
#     predperson_indx = predicted_index
#     return predicted_index

# def getpredpersonName()
# #predicting the model to get Predicted Person index
#         for i in pred[0]:
#             if(i>0.6):
#                 predicted_index = list(pred[0]).index(i)
#         pred_person_index = predicted_index
#         print(pred_person_index)

# #Validating the predicted index with Our Database and getting the name
#         pred_person_name = []
#         for pname in person_n:
#             if person_n.index(pname) == pred_person_index:
#                 pred_person_name.append(pname)

# #Getting all Details of name matching Index from database
#         student_detail = []
#         print(pred_person_name)
#         if len(pred_person_name)>0:
#             if df[df['Name']== pred_person_name[0]].index.values:
#                 student_detail.append(df.loc[pred_person_index].values.tolist())
#                 # student_detail = df.loc[pred_person_index].values.tolist()
#             print(student_detail)

# #validating the detials and Storing into Dictionary to Print it with label
#             if len(student_detail)>0:
#                 dict = {
#                     "Name:":student_detail[0][4],
#                     "Enroll:":student_detail[0][2],
#                     "Mobile No:": student_detail[0][3],
#                     "Adhar No:": student_detail[0][0],
#                     "Email:":student_detail[0][1]
#                     }
#                 loc = [(k,v) for k, v in dict.items()]
#                 for idx, lbl in enumerate(loc):
#                     offset =30
#                     cv2.rectangle(frame,(40,y1[0]+offset*idx),(450, y1[0]-30), (255,205,205),0)
#                     cv2.putText(frame, str(lbl), (40,y1[0]+offset*idx), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
#             else:
#                 cv2.putText(frame,"Unknown", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
#         else:
#             cv2.putText(frame, pred_person_name[0], (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
#     else:
#         cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()       
