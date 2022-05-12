import torch

net = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=2, out_channels=20, kernel_size=5, stride=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(in_channels=20, out_channels=48, kernel_size=5, stride=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=25600, out_features=500),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=500, out_features=30),
    torch.nn.ReLU(),
)

dict_mapping = {
    "conv1_weight": "0.weight",
    "conv1_bias": "0.bias",
    "conv2_weight": "3.weight",
    "conv2_bias": "3.bias",
    "conv3_weight": "6.weight",
    "conv3_bias": "6.bias",
    "fc5_ft_weight": "9.weight",
    "fc5_ft_bias": "9.bias",
    "fc6_10_weight": "11.weight",
    "fc6_10_bias": "11.bias"
}

import pickle

with open("lipnet2.pkl","rb") as r:
    state_dict = pickle.load(r)

new_state_dict = {}
for k in state_dict.keys():
    new_state_dict[dict_mapping[k]] = torch.tensor(state_dict[k])

net.load_state_dict(new_state_dict)
#load cv2 video 
import cv2 
import numpy as np  
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #resize the frame to 100x100
    frame = cv2.resize(frame, (100, 100))
    #make black and white
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #make a copy of the frame
    frame_copy = frame.copy()
    frame_copy = cv2.resize(frame, (100, 100))
    #add frame_copy and frame togheter
    frame = np.concatenate((frame, frame_copy), axis=1)
    #reshape the frame to (1, 2, 100, 100)
    frame = frame.reshape(1, 2, 100, 100) 
    tensor_frame = torch.Tensor(frame)
    prediction = net(tensor_frame)
    #print the prediction
    print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
