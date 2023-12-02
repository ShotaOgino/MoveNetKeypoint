import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2

#load model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

KEY_POINT_IND = {
    'NOSE' : 0,
    'LEFT_EYE' : 1,
    'RIGHT_EYE' : 2,
    'LEFT_EAR' : 3,
    'RIGHT_EAR' : 4,
    'LEFT_SHOULDER' : 5,
    'RIGHT_SHOULDER' : 6,
    'LEFT_ELBOW' : 7,
    'RIGHT_ELBOW' : 8,
    'LEFT_WRIST' : 9,
    'RIGHT_WRIST' : 10,
    'LEFT_HIP' : 11,
    'RIGHT_HIP' :12,
    'LEFT_KNEE' : 13,
    'RIGHT_KNEE' : 14,
    'LEFT_ANKLE' : 15,
    'RIGHT_ANKLE' : 16,
}

#draw Keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    # Get the coordinates of the left and right ear
    left_ear = shaped[3]
    right_ear = shaped[4]
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1) 
            # 全てのキーポイントを表示
            for i, kp in enumerate(shaped):
                ky, kx, kp_conf = kp

                if kp_conf > confidence_threshold:
                    cv2.putText(frame, f'{list(KEY_POINT_IND.keys())[i]}: x={kx}, y={ky}', (10, 60 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, f'{list(KEY_POINT_IND.keys())[i]}: x={kx}, y={ky}', (10, 60 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)

            print(left_ear[1]-right_ear[1])

            # 体の向きを推定
            body_direction = 'toward the camera' if left_ear[1] > right_ear[1] else 'back'

            # ビデオに体の向きを書き込む
            cv2.putText(frame, body_direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#draw edges
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

#make detection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256,256)
    input_image = tf.cast(img, dtype=tf.uint8)  # Change here
    
    # Setup input and output 
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Make predictions 
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Rendering 
    draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    
    cv2.imshow('MoveNet Lightning', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()