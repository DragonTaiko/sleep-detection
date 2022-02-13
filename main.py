import cv2
import pandas as pd
import os

def sleep_detection(filesource, view_detection=False):
    if not os.path.exists(filesource):
        print('Video not found')
        return False
    
    eye_cascPath = 'model/haarcascade_eye_tree_eyeglasses.xml'  #eye detect model
    face_cascPath = 'model/haarcascade_frontalface_alt.xml'  #face detect model
    faceCascade = cv2.CascadeClassifier(face_cascPath)
    eyeCascade = cv2.CascadeClassifier(eye_cascPath)

    cap = cv2.VideoCapture(filesource)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_blink = 400
    max_no_eyes = int(max_blink/(1000/fps))
    fps_window = max_no_eyes*2
    if fps_window % 2 == 0:
        fps_window += 1

    window = []
    face = 0
    frames = 0
    f_sleep = 0
    f_awake = 0
    states = []

    while(cap.isOpened()):
        ret, img = cap.read()
        if ret:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames += 1
            
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30),
            )
            # Detect face
            if len(faces) > 0:
                face += 1
                frame = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]:1]
                eyes = eyeCascade.detectMultiScale(
                    frame,
                    scaleFactor=1.1,
                    minNeighbors=31,
                    minSize=(30, 30),
                )
                if len(eyes) == 0:
                    window.append(1)                
                else:
                    window.append(0)
                    
                    
                if len(window) > fps_window:
                        window = window[1:]
                        
                sleep = window.count(1)
                awake = window.count(0)
                
                if sleep > awake:
                    # Sleep
                    eye_color = (0, 0, 255)
                    f_sleep += 1
                    states.append(1)
                else:
                    # Awake==
                    eye_color = (255, 0, 0)
                    f_awake += 1
                    states.append(0)
                    
                if view_detection:
                    # Draw rectangles around the face and around the eyes
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        roi = img[y:y+h, x:x+w]
                        eyes = eyeCascade.detectMultiScale(roi)
                        for (ex,ey,ew,eh) in eyes:
                                cv2.rectangle(roi,(ex,ey),(ex+ew,ey+eh), eye_color, 2)
                        
                    cv2.imshow('Face Recognition', img)
                    
            else:
                states.append(-1)
                
            if view_detection:
                cv2.imshow('Face Recognition', img)
        else:
            # End video
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return frames, face, states

def process_videos(datasource, video_path, results_path):
    data = pd.read_csv(datasource)
    results = {'filename':[], 'total_frames':[], 'face_frames':[], 'sleep_frames':[], 'awake_frames':[], 'sleep':[]}
    
    for i in range(data.shape[0]):
        filename = data.iloc[i]['filename']
        frames, face, states = sleep_detection(video_path+filename)
        file_data = pd.DataFrame(states)
    
        file_data.to_csv(results_path+filename[:-4]+'.csv', index=False)
        
        results['filename'].append(filename)
        results['total_frames'].append(frames)
        results['face_frames'].append(face)
        results['sleep_frames'].append(states.count(1))
        results['awake_frames'].append(states.count(0))
        results['sleep'].append(data.iloc[i]['sleep'])
        
    
    results_data = pd.DataFrame(results)
    results_data.to_csv(results_path+'results.csv', index=False)


#process_videos('sleep.csv', 'videos/' ,'results/')
frames, face, states = sleep_detection('videos/007_awake.mp4', view_detection=True)

