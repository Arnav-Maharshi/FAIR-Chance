import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt

# Open video capture
source = 0
cap = cv2.VideoCapture(source)  # Your local video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize MediaPipe Pose
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=False, 
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.8)

# For visualization of keypoints and connections
mp_drawing = mp.solutions.drawing_utils


# Measuring angle of joints
def indexF_MP(image, results): #, joint_list)
    # Loop through hands
    acc_percent_list = []
    for index, hand in enumerate(results.multi_hand_landmarks):
        hand_label = results.multi_handedness[index].classification[0].label

        #Loop through joint sets 
        for joint in [[6,5,0]]:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(180 - (radians*180.0/np.pi))
            
            if angle > 180.0:
                angle = np.abs(360-angle)

            round_angle = round(angle) #, 2)

            acc_percent = round((round_angle*100)/77.0)#, 2)
            acc_percent_list.append(acc_percent)

            joint_name = mp_hand.HandLandmark(joint[1]).name          
            angle_text = '{} {}: {}'.format(hand_label, joint_name, round_angle)
            percent_text = '{} {}: {} %'.format(hand_label, joint_name, acc_percent)
            joint_coord = tuple(np.multiply(b, [frame_height, frame_height]).astype(int))

            cv2.putText(image, angle_text, joint_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, percent_text, (500, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image, acc_percent_list


def indexF_PIP(image, results): #, joint_list)
    # Loop through hands
    acc_percent_list = []
    for index, hand in enumerate(results.multi_hand_landmarks):
        hand_label = results.multi_handedness[index].classification[0].label

        #Loop through joint sets 
        for joint in [[7,6,5]]:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(180 - (radians*180.0/np.pi))
            
            if angle > 180.0:
                angle = np.abs(360-angle)

            round_angle = round(angle, 2)

            acc_percent = round((round_angle*100)/110.0, 2) #Hyperflexion angle: 130
            acc_percent_list.append(acc_percent)

            joint_name = mp_hand.HandLandmark(joint[1]).name          
            angle_text = '{} {}: {}'.format(hand_label, joint_name, round_angle)
            percent_text = '{} {}: {} %'.format(hand_label, joint_name, acc_percent)
            joint_coord = tuple(np.multiply(b, [frame_height, frame_height]).astype(int))

            cv2.putText(image, angle_text, joint_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, percent_text, (500, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return image, acc_percent_list


def indexF_DIP(image, results): #, joint_list)
    # Loop through hands
    acc_percent_list = []

    for index, hand in enumerate(results.multi_hand_landmarks):
        hand_label = results.multi_handedness[index].classification[0].label

        #Loop through joint sets 
        for joint in [[8,7,6]]:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = round(np.abs(180 - (radians*180.0/np.pi))) 

            
            if angle > 180.0:
                angle = np.abs(360-angle)

            round_angle = round(angle)

            acc_percent = round((round_angle*100)/80.0) #Hyperflexion angle: above 100
            acc_percent_list.append(acc_percent)

            joint_name = mp_hand.HandLandmark(joint[1]).name          
            angle_text = '{} {}: {}'.format(hand_label, joint_name, round_angle)
            percent_text = '{} {}: {} %'.format(hand_label, joint_name, acc_percent)
            joint_coord = tuple(np.multiply(b, [frame_height, frame_height]).astype(int))

            cv2.putText(image, angle_text, joint_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, percent_text, (500, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    return image, acc_percent_list

while cv2.waitKey(1) != 27:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Flipping input so that it mirrors movements directly
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR 2 RGB
    
    # Detections
    results = hands.process(image)
    
    # Rendering results
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand, mp_hand.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(60, 160, 0), thickness=2, circle_radius=4), # Landmarks graphic
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2), # Connection line graphic
                                        )
            

        indexF_DIP(frame, results)
        #frame, accuracy = indexF_(frame, results)
           
    # Show the output with skeleton overlay
    cv2.imshow('MediaPipe Hand Estimation', frame)

# Release resources

cap.release()
cv2.destroyAllWindows()



