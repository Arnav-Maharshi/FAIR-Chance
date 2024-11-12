import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt

###################################
# 2D Euclidean Distance Version   #
###################################

# Open video capture
source = 0
cap = cv2.VideoCapture(source)  # Your local video

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize MediaPipe Pose
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=False, 
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.5)

# For visualization of keypoints and connections
mp_drawing = mp.solutions.drawing_utils


# Left or Right Hand detection with appropriate confidence score of the model
def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label # Left or Right
            score = classification.classification[0].score # Confidence score of the model
            text = '{} {}'.format(label, round(score, 2)) 
            
            # Extract Coordinates of corresponding wrist of each hand
            coords = tuple(np.multiply(np.array((hand.landmark[mp_hand.HandLandmark.WRIST].x, hand.landmark[mp_hand.HandLandmark.WRIST].y)),
                                        [frame_width, frame_height]).astype(int))
    
            output = text, coords

            
    return output


# Measuring angle of joints
def draw_finger_angles(image, results, joint_list):
    # Loop through hands
    for index, hand in enumerate(results.multi_hand_landmarks):
        hand_label = results.multi_handedness[index].classification[0].label

        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(180 - (radians*180.0/np.pi))
            
            if angle > 180.0:
                angle = np.abs(360-angle)
            
            if 67.0 <= angle <= 95.0:
                # Within range
                pass
            else:
                # NOT within range
                pass
                
  
            joint_name = mp_hand.HandLandmark(joint[1]).name          
            round_angle = round(angle, 2)
            text = '{} {}: {}'.format(hand_label, joint_name, round_angle)
            joint_coord = tuple(np.multiply(b, [frame_height, frame_height]).astype(int))

            cv2.putText(image, text, joint_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image


# Middle element/joint in each sub-list is the joint whose angle will be measured
joint_list = [[11,10,9]] #[[4,3,2], [7,6,5]] #[10,9,0], [14,13,0], [18,17,0]] # The 2 other elements in each sub-list are for refrence of the middle joint

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
            
            # Left or Right hand detection
            if get_label(num, hand, results):
                text, coord= get_label(num, hand, results)
                #cv2.putText(frame, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        draw_finger_angles(frame, results, joint_list)
           
    # Show the output with skeleton overlay
    cv2.imshow('MediaPipe Pose Estimation', frame)

# Release resources
cap.release()
cv2.destroyAllWindows()



