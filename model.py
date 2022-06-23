import torch
import matplotlib.pyplot as plt
import time
from threading import Thread
import cv2
import numpy as np                             
from pythonosc import udp_client
import pickle

OSCip="127.0.0.1" 
OSCport=9000 #VR Chat OSC port
client = udp_client.SimpleUDPClient(OSCip, OSCport)

net = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=2, out_channels=20, kernel_size=5, stride=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(in_channels=20, out_channels=48, kernel_size=5, stride=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=25600, out_features=500),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=500, out_features=30),
    #torch.nn.ReLU(inplace=True), you probably don't want this
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


#This is from the Vive API (This is specificly from VRCFT!). you can see some every basic tensor mapping, however, none of this may be correct. Note that I have not figured out how all the parameters work, and therfore is not suported yet.
# public enum LipShape_v2
#             {
#                 //None = -1,                            0  I dont know how to map these. (zero shows up as a huge negitive number) 
#                 JawRight = 0, // +JawX                  1
#                 JawLeft = 1, // -JawX 
#                 JawForward = 2,                         2    
#                 JawOpen = 3,                            3   
#                 MouthApeShape = 4,                      4       
#                 MouthUpperRight = 5, // +MouthUpper     5 I dont know how to map these.
#                 MouthUpperLeft = 6, // -MouthUpper
#                 MouthLowerRight = 7, // +MouthLower     6
#                 MouthLowerLeft = 8, // -MouthLower
#                 MouthUpperOverturn = 9,                 7
#                 MouthLowerOverturn = 10,                8
#                 MouthPout = 11,                         9
#                 MouthSmileRight = 12, // +SmileSadRight 10 I dont know how to map these.
#                 MouthSmileLeft = 13, // +SmileSadLeft   11
#                 MouthSadRight = 14, // -SmileSadRight
#                 MouthSadLeft = 15, // -SmileSadLeft
#                 CheekPuffRight = 16,                    12
#                 CheekPuffLeft = 17,                     13
#                 CheekSuck = 18,                         14
#                 MouthUpperUpRight = 19,                 15
#                 MouthUpperUpLeft = 20,                  16
#                 MouthLowerDownRight = 21,               17
#                 MouthLowerDownLeft = 22,                18 
#                 MouthUpperInside = 23,                  19
#                 MouthLowerInside = 24,                  20
#                 MouthLowerOverlay = 25,                 21 
#                 TongueLongStep1 = 26,                   22 
#                 TongueLongStep2 = 32,                   23
#                 TongueDown = 30, // -TongueY               I dont know how to map these.
#                 TongueUp = 29, // +TongueY              24 
#                 TongueRight = 28, // +TongueX           25
#                 TongueLeft = 27, // -TongueX
#                 TongueRoll = 31,                        26
#                 TongueUpLeftMorph = 34,                 27
#                 TongueUpRightMorph = 33,                28
#                 TongueDownLeftMorph = 36,               29
#                 TongueDownRightMorph = 35,              30 I dont know how to map these. (30 shows up as a negitive number)
#                 //Max = 37,                            
#             }

with open("lipnet2.pkl","rb") as r:
    state_dict = pickle.load(r)

new_state_dict = {}
for k in state_dict.keys():
    new_state_dict[dict_mapping[k]] = torch.tensor(state_dict[k])

net.load_state_dict(new_state_dict)
cap = cv2.VideoCapture("mouth.mp4")

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # define frame to a floating-point field to 8 bytes.
    frame = frame.astype(np.float)
    frame = frame/255.0
    frame_processed = frame.reshape(1, 2, 100, 100) 
    tensor_frame = torch.Tensor(frame_processed)
    prediction = net(tensor_frame)
    prediction = prediction.detach().numpy()
    x = prediction[0]
    #print the tensor with the highest value
    print(x.argmax())
    plt.bar((range(30)), x)
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()

    JawForward = x[2]
    JawOpen = x[3]
    MouthApeShape = x[4]
    MouthUpperP = x[5]
    MouthLowerP = x[6]
    MouthUpperOverturn = x[7]
    MouthLowerOverturn = x[8]
    MouthPout = x[9]
    SmileSadRightP = x[10]
    SmileSadLeftP = x[11]
    CheekPuffRight = x[12]
    CheekPuffLeft = x[13]
    CheekSuck = x[14]
    MouthUpperUpRight = x[15]
    MouthUpperUpLeft = x[16]
    MouthLowerDownRight = x[17]
    MouthLowerDownLeft = x[18]
    MouthUpperInside = x[19]
    MouthLowerInside = x[20]
    MouthLowerOverlay = x[21]
    TongueLongStep1 = x[22]
    TongueLongStep2 = x[23]
    TongueYP = x[24]
    TongueXP = x[25]
    TongueRoll = x[26]
    TongueUpLeftMorph = x[27]
    TongueUpRightMorph = x[28]
    TongueDownLeftMorph = x[29]
    #TongueDownRightMorph = x[30] 30 is an empty output on the model
    client.send_message("/avatar/parameters/JawForward", float(JawForward))
    client.send_message("/avatar/parameters/JawOpen", float(JawOpen))
    client.send_message("/avatar/parameters/MouthApeShape", float(MouthApeShape))
    client.send_message("/avatar/parameters/MouthUpper+", float(MouthUpperP))
    client.send_message("/avatar/parameters/MouthLower+", float(MouthLowerP))
    client.send_message("/avatar/parameters/MouthUpperOverturn", float(MouthUpperOverturn))
    client.send_message("/avatar/parameters/MouthLowerOverturn", float(MouthLowerOverturn))
    client.send_message("/avatar/parameters/MouthPout", float(MouthPout))
    client.send_message("/avatar/parameters/SmileSadRight+", float(SmileSadRightP))
    client.send_message("/avatar/parameters/SmileSadLeft+",  float(SmileSadLeftP))
    client.send_message("/avatar/parameters/CheekPuffRight", float(CheekPuffRight))
    client.send_message("/avatar/parameters/CheekPuffLeft", float(CheekPuffLeft))
    client.send_message("/avatar/parameters/CheekSuck", float(CheekSuck))
    client.send_message("/avatar/parameters/MouthUpperUpRight", float(MouthUpperUpRight))
    client.send_message("/avatar/parameters/MouthUpperUpLeft", float(MouthUpperUpLeft))
    client.send_message("/avatar/parameters/MouthLowerDownRight", float(MouthLowerDownRight))
    client.send_message("/avatar/parameters/MouthLowerDownLeft", float(MouthLowerDownLeft))
    client.send_message("/avatar/parameters/MouthUpperInside", float(MouthUpperInside))
    client.send_message("/avatar/parameters/MouthLowerInside", float(MouthLowerInside))
    client.send_message("/avatar/parameters/MouthLowerOverlay", float(MouthLowerOverlay))
    client.send_message("/avatar/parameters/TongueLongStep1", float(TongueLongStep1))
    client.send_message("/avatar/parameters/TongueLongStep2", float(TongueLongStep2))
    client.send_message("/avatar/parameters/TongueY+", float(TongueYP))
    client.send_message("/avatar/parameters/TongueX+", float(TongueXP))
    client.send_message("/avatar/parameters/TongueRoll",float(TongueRoll))
    client.send_message("/avatar/parameters/TongueUpLeftMorph", float(TongueUpLeftMorph))
    client.send_message("/avatar/parameters/TongueUpRightMorph", float(TongueUpRightMorph))
    client.send_message("/avatar/parameters/TongueDownLeftMorph", float(TongueDownLeftMorph))
    #client.send_message("/avatar/parameters/TongueDownRightMorph", float(TongueDownRightMorph)) i ran out of values, so i think that I have mapped the tensors wrong.


    #cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
