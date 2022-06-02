# TheVRMLFaceToolbox
A repo for near mouth blendshape prediction

# How does this work?
Its pretty easy, download this repo and run main.py. Currently we are using a test video as input but you can use whatever you want as input!

- [x] - Add the Vive Model

- [ ] - Decompile SRanipal and figure out how Vive processes the output values from the model. (PROCESS IN REVIEW)

- [ ] - Train a new monocular image model using Vive model. (PROCESS IN REVIEW)

### Change of plans (05/17/2022):

**I am making a new model from the ground up**

1. After further review of the methods above, I am faced with some tough chalenges on the legal front and therfore I do not think it is wise to continue.
2. This new in-house model will allow for greater customizabity in terms what other data and AI engineers can improve within the VR community and because this project advocates for open sourcing mouth tracking, it seems that it would be better to continue as such. 
3. The new target output that this model will be predicting will be FACS or the Facial Action Unit System which is basically the same as Vive's blenshape system (it's almost as if vive just used FACS and didnt tell anybody... who would figure.) 

Facial Action Unit predictors will be trained and evaluated on the following datasets (if I can get a hold of them since for some of these you have to request the authors for access):

Bosphorus

BP4D from FERA2015

DISFA

FERA2011

SEMAINE from FERA2015

UNBC

CK+

**With this in mind here is the new plan:**
- [ ] - Obtain data and preprocess 
- [ ] - Train and test model using data
- [ ] - Publish model
- [ ] - Rinse and repeat steps 2-3 until I get a 90% or above accuracy on validation and a "With sub-10 milliseconds response time" on cpu (As vive claims) which is most likely not possible but it wouldnt hurt to try.


***This repo is a work in progress and currently only outputs raw tensor values and not blenshapes***
