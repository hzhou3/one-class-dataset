# One-Class Model for Fabric Defect Detection

### Prerequisite
1. python3 and pip3 are necessary.


### A. Installation
1. git clone ***
2. **cd one-class-dataset/ && pip3 install -r requirements.txt**

	(Note that the versions of some packages (e.g., torchvision or torch) may need to be matched. Google one matched pair if errors occur).

### B. Run
1. test normal images: **cd source/ && python3 normal.py**
2. test rotated images: **cd source/ && python3 rotate.py**

### C. Dataset

1. full dataset from [the Standard Fabric Defect Glossary](https://www.cottoninc.com/quality-products/textile-resources/fabric-defect-glossary/) can be found in the github repository (https://github.com/hzhou3/one-class-dataset/).
2. expanding dataset of the paper only needs to add more images to **data/rawimage_paper**.

### D. Parameters
1. image patch

	can be changed at **line 12 of source/dataset3.py**

	default value = 32

2. orientations of Gabor filter bank

	can be changed at **line 397 of source/dataset3.py**

	default value = [0, 10, 20, ..., 170]

3. bandwidths of Gabor filter bank

	can be changed at **line 410 of source/dataset3.py**

	default value = [4,6,8,10,12]

4. stride when cropping image patches

	can be changed at **line 11 of source/normal.py or source/rotate.py**

	default value = 10

5. split ratio of train and validation sets

	can be changed at **line 13 of source/dataset3.py**

	default value = 0.9

6. dimension of code of autoencoder

	can be changed at **line 27 of source/model.py**

	default value = 90

7. More angles to test

	can be changed at **line 17 of source/rotate.py**

	default value = [0.02, 0.05, 0.2, 0.5, 2, 5]

	(note that generating rotated images - see **data/rotationraw_paper and data/rotationmask_paper** - before adding more angles).
	
	
### E. Some future research directions
1. Gabor filter bank has an issue of speed. Try to achieve same TPR with less filters (< 90).

2. Neareast Neighbor Estimator can be replaced. NNE also has an issue of speed since it requires more data points to decribe an accurate classification boundary. Try to use less data points to define the classification boundary.

3. Add attention on Gabor filters. Usually, fabrics are made by vertical or horizontal lines, so more focusing on gabor filter on these directions may be a good choice. 

4. Find a better neural network to generate general feature vectors.





### one-class-dataset from [here](https://www.cottoninc.com/quality-products/textile-resources/fabric-defect-glossary/)

raw fabric images with masks in red (255, 0, 0) 


### If you use this dataset, please consider citing our papers 

```

@inproceedings{zhou2020exploring,
  title={Exploring faster RCNN for fabric defect detection},
  author={Zhou, Hao and Jang, Byunghyun and Chen, Yixin and Troendle, David},
  booktitle={2020 Third International Conference on Artificial Intelligence for Industries (AI4I)},
  pages={52--55},
  year={2020},
  organization={IEEE},
}
@article{zhou2022one,
  title={One-Class Model for Fabric Defect Detection},
  author={Zhou, Hao and Chen, Yixin and Troendle, David and Jang, Byunghyun},
  journal={arXiv preprint arXiv:2204.09648},
  year={2022}
}




```




