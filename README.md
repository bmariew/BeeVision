# BeeVision
This code uses pre-trained models to create a convolutional neural network (CNN) that takes in images of beehive frames and identifies the presence of key components within the image (e.g., queen bee, honey, nectar, pollen, worker brood, etc.).

## Table of Contents
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Code Information](#code-information)
- [Dataset](#dataset)
  - [Data Sources](#data-sources) 
  - [Data Features](#data-features) 
  - [Data Access Statement](#data-access-statement) 

## About the Project
This project was designed to use various pre-trained CNNs to help identify key beehive components within individual hive frame images. One potential use case is helping beekeepers monitor the status of their hive and identify potential health risks to the hive before they happen, giving the keeper an opportunity to provide intervention. Additionally, this model could be used to help beginner beekeepers in learning the different component and learning how to spot those features themselves by submitting an image to this model and getting a list of labels in return.


## Getting Started
### Prerequisites
```python
  pip install requirements.txt
  ```


#ADD DIRECTIONS FOR HOW TO HAVE FOLDERS SET UP BASED ON RELATIVE PATHS
## Code Information
- <b>annotations.ipynb</b>
  - This file was used to label the images in a more organized way than manually entering values into a spreadsheet.
- <b>EDA.ipynb</b>
  - This file generates visualizations to explore the raw dataset. 
- <b>resnet18_model.ipynb</b>
  - This file runs and evaluates the Resnet18 model. 
- <b>efficientnet_b0_model.ipynb</b>
  - This file runs and evaluates the EfficientNetB0 model. 
- <b>densenet201_model.ipynb</b>
  - This file runs and evaluates the DenseNet201 model.
- <b>helper.py</b>
  - This file contains functions to help train and evaluate the models.
- <b>metrics.py</b>
  - This file contains a collection of functions for computing standard classification evaluation metrics using the 'sklearn.metrics' library. It is designed to take true labels ('y_true') and predicted labels or probabilities ('y_pred' or 'y_prob') as input and return various metrics.
- <b>plots.py</b>
  - This file contains functions for creating various plots to visualize the results of the models.

## Dataset
### Data Sources
~3,000 images of beehives from various distances and angles collected from:
1. Google image search for beehive frames with specific components
2. Local beekeepers
### Data Features
- Typical Features
  - Queen (Marked and Unmarked)
  - Honey Capped
  - Nectar
  - Drone Brood
  - Worker Brood
  - Eggs * Larvae
  - Queen Cells
  - Drawn Comb
  - Pollen
- Atypical Features
  - Wax Moth
  - Mold
  - Chalk Brood
  - Foul Brood
  - Other
### Data Access Statement
The dataset used in this project is publicly available and can be downloaded from this repository: BeeVisionImages.zip.

### Feature Labeling
As this dataset was created from scratch, we had to label each component within each image manually. The images can be found in the image folder while the labels for each image can be found in annotations.csv. There is an additional column that provides the quality of the image (H - High, M- Medium, L - Low). Images were marked 'L' for being too small, poorly lit, blurry, etc.







  
