



# K-means聚类算法



## 数据集

- Iris数据集
- 数据集包含150个数据样本，分为3类，每类50个数据，每个数据包含4个属性。可通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类



## 算法实现与结果



### 读Iris数据集

----

$ 数据集有四个特征\\ $
$ dataSet[i] = (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, Label)\\ $
$ (花萼长度，花萼宽度， 花瓣长度， 花瓣宽度, 花的种类)\\  $
$ Label = Iris Setosa/Iris Versicolour/Iris Virginica\\  $
$ 山鸢尾/杂色鸢尾/维吉尼亚鸢尾\\ $

```python
# K_means.py
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
```

>**dataMat**
>
>[[5.1 3.5 1.4 0.2]
>
> [4.9 3. 1.4 0.2]
>
> [4.7 3.2 1.3 0.2]
>
> [4.6 3.1 1.5 0.2]
>
> [5. 3.6 1.4 0.2]
>
> [5.4 3.9 1.7 0.4]
>
> [4.6 3.4 1.4 0.3]
>
> [5. 3.4 1.5 0.2]
>
>...]
>
>
>
>**label**
>
>[[0.]
>
> [0.]
>
> ...
>
>[1.]
>
> [1.]
>
>...
>
>[2.]
>
> [2.]]



### 欧式距离

----

```python
# 欧式距离
def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB, 2)))
```



### 随机产生k个质心

----

```python
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
```

>k 个 质心（4维向量）
>
>[[4.44877209 4.19910264 4.15300667 1.30898085]
>
> [5.98667693 2.73262682 6.65354096 1.46322745]]
>
>[[5.64805779 2.39567147 2.0901101 1.05881078]



### K-means聚类函数

----

- 算法描述

> 迭代过程
>
> ​	对于每个数据点，找到与之最近的质心
>
> ​		if 存在数据点最近的质心发生改变
>
> ​			继续迭代过程
>
> ​		else if 所有数据点最近的质心没有发生改变
>
> ​			停止迭代
>
> ​	所有质心更新为距离它最近的数据点组成的簇的均值	
>
> 返回得到的聚类结果	

```python
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
```



### 二分K-均值聚类函数

----

- 算法描述

>初始一个质心，一个簇
>
>当质心的个数 < K:
>
>​	对每个簇尝试做K-means二分类，分成2个簇，得到2个质心
>
>​	计算每次分类的损失值，取最小损失值的分类作为最终最佳的分类
>
>​	将分类得到的2个质心，一个替换原有被分类的那个质心
>
>​	另一个质心添加到质心列表的末尾
>
>​	更新所有数据点所属的簇
>
>返回得到的聚类结果

```python
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
```



### 主函数

----

```python
# 主函数
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
```

- 准确率

<img src="result.png" alt="result" style="zoom:30%;" />

