__author__ = 'Hao - 6/2020'

from uti import * 
import time


path = os.path.dirname(os.path.realpath(__file__))
path = path.replace("source", "data/rawimage_paper")
root = path.replace("data/rawimage_paper", "data/")
# patch_size = 32
stride = 10 # small value takes more time to process since its NN. Usually 10 is good. 
epoch = 60
count = 0
acc = 0
time_total = 0



for filename in os.listdir(path):
	time_all = 0
	count += 1
	print("Now processing ", filename)
	datasets = fabricSet_One(root, filename, angle= 0, patch_size=patch_size, stride=stride, transform = None)
	# datasets2 = fabricSet_One(root, filename, angle=0, patch_size=patch_size, stride=stride, transform = None)


	# comment out if training is not needed. Training is only needed if autoencoder is on.
	train(filename, datasets, epoch=epoch, angle = 0, gabor=True)

	# test with autoencoder
	
	tpr, time_all = postprocess_latent_code_gabor(filename, 0, datasets=datasets, datasets_train=datasets)

	# test with mean and std only

	# tpr, time_all = postprocess_latent_code_firstorder(filename, 0, datasets=datasets, datasets_train=datasets)

	# test with PCA

	# tpr, time_all = postprocess_latent_code_pca(filename, 0, datasets=datasets, datasets_train=datasets)  
	
	print("tpr is ", tpr)

	print ("time process ", time_all)
	
	print("time dataset is", datasets.time)

	time_all = time_all + datasets.time

	print("time is ", time_all)

	acc += tpr
	time_total += time_all
	print()
	print()

print("total acc = {0}".format(acc/count))
print("total time = {0}".format(time_total/count))
