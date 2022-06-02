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

CK+ *(Obtained)*

***However a new dataset is most likely to be made for a couple of key reasons***

1. Individually the datasets lack the volume needed to train a model on FACS so unless a large majorty are obtained, the model would not be robust
   -  Without a variety of situations, lighting conditions, ect, the model would not be robust, and seeing as most of these datasets are not "In The Wild" it is harder to make a robust model.
2. Models that label FACS data on the whole face have been around for quite some time, and by using large datasets of faces of which many are also avalible, labeling them and then croping the bottom face regions, its feasable to make a corpus of data large enough and with a great enough variety to train a model with high robustness.
3. Because most of the data above doesnt allow redistabution even after being acquired, it's impossible to easily give others the chance to tackle the same problem. 


**With this in mind here is the new plan:**
- [ ] - Create dataset by using OpenFace to label a large corpus of data on FACS
- [ ] - Preprocces data (croping mouth region amoung other things)
- [ ] - Train and test model using data
- [ ] - Publish model
- [ ] - Rinse and repeat steps 2-4 until I get a 90% or above accuracy on validation and a "With sub-10 milliseconds response time" on cpu as vive claims with their system which is most likely not possible but it wouldnt hurt to try.

### What can and cant expect with this project

1. I am under no obligation to finish this project, however the general interest of the VR communtity and possiblity to democratize such technologies compels me to complete this amoung other reasons.
2. With the plan above, I have provided a pipeline in which someone could feasibly create a model, this method gives others the chance to also take part. I expect that this may happen, if another system better than mine is developed, do not alarmed if this project shuts down or changes methods drasticly.
3. You can expect that a this repo will stay MIT forever
4. You can expect a dataset release in the future and that it will be MIT
5. The first in house model in this repo will most likely not reach the benchmarks I have set for myself
6. The 10ms goal may never be reachable

***With that all said and done***
- If you are interseted in helping this repo then please DM me at Summer#2406!

Some of the critical things I need help with:
1. GPU power (I currently do not have access to the systems I normally would making it a little difficult) 
2. Data storage (I dont have much space left on my drives XD) 
3. AI modeling help (I mean im not the smartest person in the world)

***This repo is a work in progress and currently only outputs raw tensor values and not blenshapes***
