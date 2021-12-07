ResNet50:

To run the code, first you need to download the dataset from kaggles first:
link: https://www.kaggle.com/c/dog-breed-identification/data

Then modify the code of train_path, test_path, train_labels, test_labels to the corresponding paths in your computer. 
(train_path = folder path of train images, test_path = folder path of test images, train_labels = path of labels.csv file, test_labels = path of sample_submission.csv file)

Next, open CMD, switch to the folder of the ResNet50.py exist, enter the commend: python ResNet50.py

Now, the code will start running.

After prediction of labels, you can enter a single image to get the breed of dog in this image.
(e.g. D:\Project\dog-breed-identification\test\00c14d34a725db12068402e4ce714d4c.jpg)


