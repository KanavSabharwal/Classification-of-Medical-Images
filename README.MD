## Python Library Requirements for running the program
  -sys
  -torch
  -torchvision
  -matplotlib
  -copy
  -sklearn
  -PIL
  -numpy
  -pandas
  -os
  -re
  -cv2
  
## Usage - How to Run
> 1. Open terminal
> 2. Run the following command
	sh run.sh train_data/ test_data/

> 3. Here 'train_data' should refer to the path of your training data folder and 'test_data' should refer to the path of your test data folder

## Outputs
> 1. ckp
	a. predictions for Densenet, GoogleNet, Resnet
	b. loss-accuracy graphs for Densenet, GoogleNet, Resnet
> 2. .
	a. test_result.csv - The final test result file with predicted labels
