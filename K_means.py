from numpy import *
import numpy as np

# 读数据集
def loadDataSet(fileName):
	dataMat = []
	label = []
	label_dict = {
					"Iris-setosa": 0.0,
					"Iris-versicolor": 1.0,
					"Iris-virginica": 2.0
				 }
	fr = open(fileName)
	for i, line in enumerate(fr.readlines()):
		if i == 0:
			continue
		curLine = line.strip().split(',')
		fltLine = []
		for val in curLine[:-1]:
			fltLine.append(float(val))
		dataMat.append(fltLine)
		label.append(label_dict[curLine[-1]])
	return mat(dataMat), mat(label).T # 返回数据集和标签

# 欧式距离
def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB, 2)))

# 随机产生 k 个质心
def randCent(dataSet, k):
	col = shape(dataSet)[1] # col = 4
	centroids = mat(zeros((k, col))) # k = 3 个 col = 4 维向量
	for j in range(col):
		minJ = min(dataSet[:, j])
		maxJ = max(dataSet[:, j])
		rangeJ = float(maxJ - minJ)
		centroids[:, j] = minJ + rangeJ * random.rand(k, 1) # 随机列向量
	return centroids

# K-means 聚类函数
def KMeans(dataSet, k, distMeans = distEclud, createCent = randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2))) # 0: index, 1: 距离平方
	centroids = createCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m): # 每个点
			minDist = inf; minIndex = -1
			for j in range(k): # 最近的质心
				distJI = distMeans(centroids[j, :], dataSet[i, :]) # 与质心距离
				if distJI < minDist: # 更新最近质心
					minDist = distJI
					minIndex = j
			if clusterAssment[i, 0] != minIndex: # 最近的质心改变
				clusterChanged = True
				clusterAssment[i, :] = minIndex, minDist**2
		for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
			# nonzero 返回不为0的元组的下标 .A 转为array
			centroids[cent, :] = mean(ptsInClust, axis = 0) # 更新质心为最近簇的均值
	return centroids, clusterAssment # 返回质心，数据点所属簇，数据点与质心距离

# 二分K均值算法
def bitKeans(dataSet, k, distMeans = distEclud):
	m = shape(dataSet)[0]
	n = shape(dataSet)[1]
	clusterAssment = mat(zeros((m, 2)))
	centroid0 = mean(dataSet, axis = 0).tolist()[0] # 所有点的均值作为第一个质心
	centList = [centroid0]
	for j in range(m):
		clusterAssment[j, 1] = distMeans(mat(centroid0), dataSet[j, :])**2 # 所有点属于一个簇
	while len(centList) < k: # 每次增加一个质心/簇，直到达到k个
		lowestSSE = inf # 损失值
		for i in range(len(centList)): # 尝试计算将每个质心二分类的损失值
			ptsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :] # 属于该簇的点
			centroidMat, splitClusterAss = KMeans(ptsInCluster, 2, distMeans) # 用 K-means 对该簇做二分类
			sseSplit = sum(splitClusterAss[:, 1]) # 二分部分的误差值
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1]) # 未二分部分的误差值
			if sseSplit + sseNotSplit < lowestSSE: # 总误差值求最小值
				bestCentToSplit = i # i 是最佳拆分簇
				bestNewCents = centroidMat # 最佳的划分出的2个簇
				bestClustAss = splitClusterAss.copy() # 最佳的所有数据点所属簇
				lowestSSE = sseSplit + sseNotSplit # 最小总误差值
		# 二分后一个质心添加到最后并更新相应数据点所属簇
		bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList) 
		centList.append(bestNewCents[1, :][0].tolist()[0])
		# 另一个质心替换掉质心 i 并更新相应数据点所属簇
		bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = i 
		centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  
		# 更新所有数据点所属簇
		clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
	return mat(centList), clusterAssment # 返回质心，数据点所属簇，数据点与质心距离


if __name__ == '__main__':
	K = 3
	dataSet, label = loadDataSet("iris.txt") # 150 * 4
	myCentList, myClusterAssment = bitKeans(dataSet, K) # 二分 K-means 算法
	len = shape(label)[0]
	pos = [0, int(len/3)-1, int(2*len/3)-1, int(len)-1]
	sum = 0
	# 由于不知道三种分类对应的数字，这里把每50个结果中出现最多的那个作为正确的结果数
	for j in range(K): # 0~49 50~99 100~149
		maxCnt = -1
		for i in range(K):
			cnt = nonzero(myClusterAssment[pos[j]:pos[j+1], 0].A == i)[0].size
			if cnt > maxCnt:
				maxCnt = cnt
		sum = sum + maxCnt
	print("Accuracy: %.2f%%" % (sum*100.0 / float(len))) # 准确率
	
