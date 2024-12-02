# BeeVision
This code creates a Convolutional Neural Network (CNN) that takes in images of Beehive frames and identifies whether or not key components (e.g., queen bee, honey, nectar, pollen, worker brood, etc.) are present in the image.

## Table of Contents
- [About the Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Features](#features)
- [Contributing](#contributing)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## About the Project
This project was designed to use various CNNs to help identify key beehive components within individual frame images. One potential use case is helping beekeepers monitor their hive and identify potential health risks to the hive before they happen, giving the keeper an opportunity to provide an intervention. Additionally, this model could be used to help beginner beekeepers in learning the different component and learning how to spot those features themselves by submitting an image to this model and getting a list of labels in return.


## Getting Started
## Prerequisites
## Installation
## Usage
## Data Sources
### Data Sources
~3,000 images from:
1. Google image search for beehive frames with specific components
2. Local beekeepers
### Data Features
- Wax Moth
- Mold
- Chalk Brood
- Queen (Marked and Unmarked)
- Honey (Capped and Uncapped)
- Drone Brood
- Worker Brood
- Eggs
- Larvae
- Queen Cells
- Drawn Comb
- Pollen
- Foul Brood
- Other

*WE SHOULD RENAME BEE BREAD BACK TO POLLEN, UNCAPPED HONEY TO NECTAR?. ALSO SHOULD REMOVE THE ANNOTATOR COLUMNS FOR THE REPO.*

### Feature Labeling
As this dataset was created from scratch, we had to label each component within each image manually. The images can be found in the image folder while the labels for each image can be found in annotations.csv. There is an additional column that provides the quality of the image (H - High, M- Medium, L - Low). Images were marked 'L' for being too small, poorly lit, blurry, etc.

## Features
## Contributing
## Contact
## Acknowledgments






  
