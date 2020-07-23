# Program To Read video 
# and Extract Frames 
import cv2 
import face_recognition
from skimage.io import imsave

file_name = 'rap_god'
# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 1
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        cv2.imwrite("./videos/input/tmp.jpg", image)
        

        image = face_recognition.load_image_file("./videos/input/tmp.jpg")
        face_landmarks_list = face_recognition.face_landmarks(image)
        if len(face_landmarks_list)>0:
            f_name = './videos/input_frames/%s_%d.jpg'%(file_name, count)
            imsave(f_name, image)
            print(f_name)
        # Saves the frames with frame-count 
         
  
            count += 1
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("./videos/input/"+file_name+'.mp4')