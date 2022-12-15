import cv2
import mediapipe as mp
import numpy as np
import os 
import torch
from model import custom_model2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def to_cat(label):
    mapping = {"bracelet": 0,
                "breakdown": 1,
                "christmas": 2,
                "down": 3,
                "firefighter": 4,
                "heavy": 5,
                "ice cream": 6,
                "india": 7,
                "kindergarten": 8,
                "kiss": 9,
                "loan": 10,
                "lunch": 11,
                "monkey": 12,
                "muscle": 13,
                "northwest": 14,
                "perfume": 15,
                "relief": 16,
                "salad": 17,
                "salt": 18,
                "secret": 19,
                "vocabulary": 20}
    return float(mapping[label])


def main():

    # load the model
    model = custom_model2(63, 128, 21, 2)
    model.load_state_dict(torch.load("model_weights.pth"))

    use_cuda = torch.cuda.is_available()
    print('use cuda: %s'%(use_cuda))
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


    cap = cv2.VideoCapture(-1)

    # file_name = file_path.split("/")[-1].split(".")[0]
    # save_name = os.path.join(save_root_folder, file_name)

    # cap = cv2.VideoCapture(file_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sets = int(num_frames/30)
    print(f"the number of frames: {num_frames}")
    print(f"total sets: {sets}")

    cum_feat = []
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        fcount = 1
        # scount = 0

    while cap.isOpened():
        # if scount == sets: ## saving only in groups of 30 datapoints
        # print("set copmlete")
        # break

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
            break 
        print(image)                                             
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(hand_landmarks.landmark)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                feats = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])  
                feats = feats.flatten()
                cum_feat.append(feats)
        if len(cum_feat) == 30:
            cum_feat_tensor = torch.from_numpy(np.array(cum_feat))
            output = model(cum_feat_tensor.type(FloatTensor))
            print(output)
            # current_name = save_name + f"_{scount}.npy"
            # np.save(current_name, cum_feat)
            # print(f"[SAVE] File: {current_name}")
            # scount += 1
            cum_feat = [] ## reset cum_feat
        fcount += 1 ## increment frame count 

        if cv2.waitKey(5) & 0xFF == 27:
            break
    # print(len(cum_feat))
    # print(cum_feat[0].shape)
    cap.release()

    pass


if __name__=="__main__":
    main()