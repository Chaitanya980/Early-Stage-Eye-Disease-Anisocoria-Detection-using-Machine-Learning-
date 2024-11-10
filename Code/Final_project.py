#!/usr/bin/env python
# coding: utf-8

# #### Importing required libraries like numpy,pandas,matplotlib and CV2

# In[21]:


import cv2
import os
import gradio as gr
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')


# #### Importing image and displaying it in notebook

# In[2]:


img = cv2.imread('./Dataset_anisocoria/test_images/test_2.jpg')
plt.imshow(img)


# #### Converting image from RGB to GRAY

# In[3]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')


# #### Detecting Eyes in image using openCV eye haarcascade

# In[4]:


eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

eyes = eye_cascade.detectMultiScale(gray)


# In[5]:


len(eyes)


# In[6]:


(x1,y1,w1,h1) = eyes[0] # left
(x2,y2,w2,h2) = eyes[1] # rigth


# #### Marking Eyes deteced

# In[7]:


l_eye_rect = cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,255,0),2)
r_eye_rect = cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,255,0),2)
plt.imshow(r_eye_rect)


# In[8]:


r_eye_rect = cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,255,0),2)
plt.imshow(r_eye_rect)


# #### cropping left and right eyes from image

# In[9]:


left_croped = img[y1:y1+h1, x1:x1+w1]
rigth_croped = img[y2:y2+h2, x2:x2+w2]

print(left_croped.shape)
print(rigth_croped.shape)


# In[10]:


plt.imshow(left_croped)


# In[11]:


plt.imshow(rigth_croped)


# ### Combining code into function which take img as input and gives cropped rigth & left eye

# In[12]:


def get_left_rigth_cropped_eyes(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    
    eyes = eye_cascade.detectMultiScale(gray)
    
    if len(eyes) != 2 :
        return 0,0
    (x1,y1,w1,h1) = eyes[0]
    (x2,y2,w2,h2) = eyes[1]
    
    left_croped = img[y2:y2+h2, x2:x2+w2]
    rigth_croped = img[y1:y1+h1, x1:x1+w1]
    return left_croped,rigth_croped


# In[15]:


img2 = cv2.imread('./Dataset_anisocoria/test_images/test_1.jpg')
plt.imshow(img2, cmap='gray')


# In[16]:


left_croped_1,rigth_croped_1 = get_left_rigth_cropped_eyes('./Dataset_anisocoria/test_images/test_2.jpg')
print(left_croped_1.shape)
print(rigth_croped_1.shape)


# In[17]:


plt.imshow(left_croped_1)


# In[18]:


plt.imshow(rigth_croped_1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Importing dataset images with OS library 

# In[19]:


path_to_raw_data = "./Dataset_anisocoria/Raw_img/"
path_to_renamed_data = "./Dataset_anisocoria/renamed_dataset/"
path_for_cropped = "./Dataset_anisocoria/Cropped_img"
img_paths = []
img_names = []


# #### Scanning the directory 

# In[22]:


for entry in os.scandir(path_to_raw_data):
    img_paths.append(entry.path)
    #print(entry.path)


# In[23]:


print('Total Image :',len(img_paths))
print('Image1 path :',img_paths[0])


# **Go through all images in dataset folder and create Renamed images for them. There will be Renamed folder after you run this code**
count = 0
# os.makedirs(path_to_renamed_data)
# os.makedirs(path_for_cropped)

for img_path in img_paths:
    count += 1
    img = cv2.imread(img_path)
    
    img_name = 'anisocoria_img_' + str(count)
    
    new_file_name = img_name + ".png"
    
    img_names.append(path_to_renamed_data+new_file_name)
    
    new_file_path = path_to_renamed_data + "/" + new_file_name
    
    cv2.imwrite(new_file_path, img)
    
print(count)
print(img_names[0])
# ### Looping all the images of renamed dataset and cropping and storing left and rigth image
for img_path in img_names:
    img = cv2.imread(img_path)
    
    left_cropped,rigth_cropped = get_left_rigth_cropped_eyes(img)
    
    if len(left_cropped) == 1 or len(rigth_cropped)==1:
        print(img_path)
        continue
    left_cropped = cv2.resize(left_cropped,(300,300))
    rigth_cropped = cv2.resize(rigth_cropped,(300,300))
    
    show(rigth_cropped)
#     img_name = img_path.split('/')[-1][:-4]
    
#     new_dir = path_for_cropped +'/'+ img_name
#     os.makedirs(new_dir)
    
#     left_path = new_dir + '/left.png'
#     rigth_path = new_dir + '/rigth.png'
    
#     cv2.imwrite(left_path, left_cropped)
#     cv2.imwrite(rigth_path, rigth_cropped)
# ### Code to scan the left & rigth cropped images from directory  
for entry in os.scandir(path_for_cropped):
    print(entry.path)
    for e in os.scandir(entry.path):
        print(e.path)
    
# In[ ]:





# In[ ]:





# In[ ]:





# # All code combined into module

# In[24]:


def show(img,text="image"):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[25]:


def get_left_rigth_cropped_eyes(img):
    
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
    
    eyes = eye_cascade.detectMultiScale(img)
    
    if len(eyes) < 2 :
        return [0],[0]
    
    (x1,y1,w1,h1) = eyes[0]
    (x2,y2,w2,h2) = eyes[1]
    
    left_croped = img[y2:y2+h2, x2:x2+w2]
    rigth_croped = img[y1:y1+h1, x1:x1+w1]
    
    return left_croped,rigth_croped


# In[26]:


def valide_radius(circles):
    l = []
    for item in circles:
        X = item[0][0]
        Y = item[0][1]
        R = item[1]
        
        if R < 33 and R > 12:
            if Y < 240 and Y > 60:
                l.append([(X,Y),R])
    return l


# In[27]:


def find_pupil(img):
    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #show(gray_img)
    gray_img= cv2.GaussianBlur(gray_img, (7,7),0)
    #show(gray_img)
    
    
    _, threshold = cv2.threshold(gray_img,25,255, cv2.THRESH_BINARY_INV)
    #show(threshold)
    
    contours, heirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    circles = [] 
    
    for cnt in contour:
        (x, y, w, h) = cv2.boundingRect(cnt)
        #cv2.drawContours(img, [cnt], -1, (0,0,255), 3)
        
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
        
        (cx,cy), radius = cv2.minEnclosingCircle(cnt)
        
        center = (int(cx), int(cy))
        radius = int(radius)
        
        circles.append([center,radius])
        
    valide_points = valide_radius(circles)
    
    if len(valide_points) == 0:
        valide_points.append(circles[0])
    
    #print(valide_points)
    X = valide_points[0][0][0]
    Y = valide_points[0][0][1]
    R = valide_points[0][1]
    
    cv2.circle(img, (X,Y), R, (255,0,255), 2)
        
    return [img,R]


# In[28]:


def anisocoria_result(img):
    left_cropped,rigth_cropped = get_left_rigth_cropped_eyes(img)
    
    if len(left_cropped) == 1 or len(rigth_cropped) == 1:
        print("Eyes not detected")
        return
        
    left_img = cv2.resize(left_cropped,(300,300))
    rigth_img = cv2.resize(rigth_cropped,(300,300))
    
    #show(left_img)
    #show(rigth_img)
    
    l_result = find_pupil(left_img)
    r_result = find_pupil(rigth_img)
    
    #print("Left Pupil Radius :",l_result[1])
    print1 = "Left Pupil Radius :" + str(l_result[1])
    print2 = "Right Pupil Radius :" + str(r_result[1])
    #print("rigth Pupil Radius :",r_result[1])
    
    #show(l_result[0])
    #show(r_result[0])
    return left_cropped,rigth_cropped,l_result[0],r_result[0],print1,print2


# In[ ]:





# In[30]:


img = cv2.imread('./Dataset_anisocoria/test_images/test_2.jpg')
#show(img)
result = anisocoria_result(img)


# In[32]:


show(result[0])
show(result[1])
show(result[2])
show(result[3])
print(result[4])
print(result[5])


# In[33]:


custom_css = """
<style>
.output-root {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: space-evenly;
}

.output-root > div {
    flex: 1;
    margin: 10px;
}
</style>
"""

iface = gr.Interface(fn=anisocoria_result,
                      inputs="image", 
                     outputs=[
        gr.outputs.Image(type="numpy", label="Image 1"),
        gr.outputs.Image(type="numpy", label="Image 2"),
        gr.outputs.Image(type="numpy", label="Image 3"),
        gr.outputs.Image(type="numpy", label="Image 4"),
        gr.outputs.Textbox(label="Text 1"),
        gr.outputs.Textbox(label="Text 2")
    ],
    title="Anisocoria Detection",
    
    
)


# Add custom CSS to the interface

html_code = iface.launch(share=True)
custom_html = f"""
<!DOCTYPE html>
<html>
<head>
{custom_css}
</head>
<body>
{html_code}
</body>
</html>
"""

# Save the custom HTML to a file
with open("custom_interface.html", "w") as file:
    file.write(custom_html)

