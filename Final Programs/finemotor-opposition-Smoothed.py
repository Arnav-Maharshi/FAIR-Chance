import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=False, 
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.5)

# For visualization of keypoints and connections
mp_drawing = mp.solutions.drawing_utils

# Open video capture
source = 0
cap = cv2.VideoCapture(source)  # Your local video
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


prev_landmarks = None
def smooth_landmarks(landmarks, prev_landmarks, alpha):
    """
    Smooths the current landmarks with a weighted average of the previous landmarks.
    ** (Using EMA (Exponential Moving Average)) **

    Check 'finemotor-indivdual-angles-Smoothed.py' for more details
    """
    if prev_landmarks is None:
        return landmarks  # No previous landmarks; return the current ones as is
    
    smoothed_landmarks = []
    for curr, prev in zip(landmarks, prev_landmarks):
        smoothed = alpha * curr + (1 - alpha) * prev
        smoothed_landmarks.append(smoothed)
    return np.array(smoothed_landmarks)


def draw_joint_distance_3d(image, results, joint_list): # Distance of thumb from respective finger joint
    # Loop through hands
    for index, hand in enumerate(results.multi_hand_landmarks):
        hand_label = results.multi_handedness[index].classification[0].label

        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z]) # Second coord
            
            distance = round(math.dist(a, b), 2)
            
            a_name = mp_hand.HandLandmark(joint[0]).name 
            b_name = mp_hand.HandLandmark(joint[1]).name
            text = '{}'.format(distance)
            joint_coord = tuple(np.multiply([b[0], b[1]], [frame_width, frame_height]).astype(int))

            cv2.putText(image, text, joint_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return image

joint_list = [[4, 8], [4, 12], [4, 16], [4, 20]]

while cv2.waitKey(1) != 27:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    # BGR 2 RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detections
    results = hands.process(image)
    

    # Rendering results
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
             # Extract landmark positions
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            
            # Apply smoothing
            smoothed_landmarks = smooth_landmarks(landmarks, prev_landmarks, alpha=0.7)
            prev_landmarks = smoothed_landmarks  # Update previous landmarks for the next frame
            
            # Apply smoothed landmarks back to hand for rendering
            for i, lm in enumerate(hand.landmark):
                lm.x, lm.y, lm.z = smoothed_landmarks[i]

            mp_drawing.draw_landmarks(frame, hand, mp_hand.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(60, 160, 0), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                        )
        draw_joint_distance_3d(frame, results, joint_list)
            

    # Show the output with skeleton overlay
    cv2.imshow('MediaPipe Pose Estimation', frame)

# Release resources
cap.release()
cv2.destroyAllWindows()


def draw_joint_distance_3d_v2(point1, point2):
  """Calculates the Euclidean distance between two 3D points.

  Args:
    point1: A tuple or list representing the first point (x1, y1, z1).
    point2: A tuple or list representing the second point (x2, y2, z2).

  Returns:
    The distance between the two points.
  """

  x1, y1, z1 = point1
  x2, y2, z2 = point2

  distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

  return distance