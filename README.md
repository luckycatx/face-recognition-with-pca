# Face Recognition with PCA

*Implemented with modern C++*

## Build Prerequisite
- C++20 support
- OpenCV library

## Example Usage
- *For training dataset*
```cmd
./pca_train <dataset_path>
>>
./pca_train ./dataset/train
```
- *For recognition*
```cmd
./recognize <test_path> <eigens_path> <dataset_path>
./recognize <eigens_path> <dataset_path> (using camera)
>>
./recognize ./dataset/test ./eigens ./dataset/train
./recognize ./eigens ./dataset/train
```

## Some Example Results
![avatar][res1]
![avatar][res2]
![avatar][res3]

- *Accuracy of the sample test dataset:*
![avatar][acc]

## Notes
- Images need to be saved in a subfolder in the path to the dataset, assuming the subfolder name is the corresponding subject name<br>
*e.g. **subject1** is the subject name of path* `./dataset/train/subject1`


[res1]: ./samples/res1.png
[res2]: ./samples/res2.png
[res3]: ./samples/res3.png
[acc]: ./samples/acc.png