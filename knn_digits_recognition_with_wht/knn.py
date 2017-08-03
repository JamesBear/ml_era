import numpy as np
import os
import math


knn_k = 10

def get_file_content(path):
	file = open(path, 'r')
	content = file.read()
	file.close()
	return content.strip()

def loadfile(path, name):
	
	digit = int(name[0])
	
	matrix = []
	#dataset = []
	content = get_file_content(path)
	for line in content.splitlines():
		row = []
		for c in line:
			row.append(int(c))
		matrix.append(row)
	
	dataset = np.array(matrix)
	
	return digit, dataset

def loadSample(targetPath) :
	samples = []
	for root,dirs, files in os.walk(targetPath):
		for f in files:
			digit, sample = loadfile(os.path.join(root, f), f)
			samples.append([digit,sample])
	return samples

def sample_distance(test_sample, training_sample):
	distance_matrix = test_sample-training_sample
	square_sum = (distance_matrix**2).sum()
	distance = math.sqrt(square_sum)
	
	return distance
	
def partial_sort(k, list):
	
	list.sort(key = lambda item:item[0])
	
	return list[:k]
		
		
def get_weighted_most_frequent(leading_k_samples):
	# test: no weight
	a = [0]*10
	for distance, digit in leading_k_samples:
		a[digit] += 1
	maxcount = max(a)
	for i in range(len(a)):
		if a[i] == maxcount:
			return i
			
def knn_classify(test_sample, samples):
	print('begin to classify...')
	distances = []
	for digit,sample in samples:
		distance = sample_distance(test_sample, sample)
		distances.append([distance,digit])
	#print(len(distances))
	leading_k_samples = partial_sort(knn_k,distances)
	#print(leading_k_samples)
	result = get_weighted_most_frequent(leading_k_samples)
	
	return result
	
def scalePicture(bit_matrix, width, height):
	newMatrix = np.zeros(shape = (height, width))
	old_height, old_width = bit_matrix.shape
	width_ratio = old_width/width
	height_ratio = old_height / height
	for y in range(height):
		for x in range(width):
			newMatrix[y,x] = bit_matrix[int(y*height_ratio), int(x*width_ratio)]
			
	return newMatrix


	
def load_picture(path):
	from PIL import Image
	im = Image.open(path) #Can be many different formats.
	pix = im.load()
	#print im.size #Get the width and hight of the image for iterating over
	#print pix[x,y] #Get the RGBA Value of the a pixel of an image
	w,h = im.size
	bit_matrix = np.zeros(shape=(h,w))
	print(bit_matrix.shape)
	for y in range(h):
		for x in range(w):
			bit_matrix[y,x] = int(sum(pix[x,y])<255*1.5)
	#print("bit_matrix:", bit_matrix)
	#print(bit_matrix.sum())
	return bit_matrix
	
def print_pic(pic_matrix):
	h, w = pic_matrix.shape
	
	for y in range(h):
		for x in range(w):
			print(int(pic_matrix[y, x]), end="")
		print("")
	
def pic_to_sample(pic_matrix, samples):
	#print(samples[0][1].shape)
	#print(samples[0][1])
	height, width = samples[0][1].shape
	scaled_pic = scalePicture(pic_matrix, width, height)
	
	print_pic(scaled_pic)
	
	return scaled_pic
	
def test_file(test_path, test_name):
			
	print("Test file:", test_path)
	xxxx,test_sample = loadfile(test_path, test_name)



	#print("distance of \n",a, "and \n", b , " is ",sample_distance(a,b))
	result = knn_classify(test_sample, trainingDigits)

	print("result class: ", result)
	
def test_picture_file(test_path):
		
	print("test picure: "+test_path)
		
	pic_matrix = load_picture(test_path)
	pic_sample = pic_to_sample(pic_matrix, trainingDigits)
	
	#print("Test file:", test_path)
	#xxxx,test_sample = loadfile(test_path, test_name)



	#print("distance of \n",a, "and \n", b , " is ",sample_distance(a,b))
	result = knn_classify(pic_sample, trainingDigits)

	print("result class: ", result)
	
	
trainingDigits = loadSample('trainingDigits')
#testDigits = loadSample('testDigits')

a = np.array([[0,1], [1, 9]])
b = np.array([[1,2], [2, 10]])



#test_file('testDigits/3_14.txt', '3_14.txt')
test_picture_file("test2.bmp")

input("Press enter to continue..")
