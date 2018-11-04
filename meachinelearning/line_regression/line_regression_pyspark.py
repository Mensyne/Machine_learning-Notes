
from __future__ import print_function

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint,LinearRegressionWithSGD,LinearRegressionModel

if __name__ == '__main__':
	sc=SparkContext(appName="PythonLinearRegressionwithSGDExample")
	# load parse the data
	def parsePoint(Line):
		values = [float(x) for x in Line.replace(',',' ').split(' ')]
		return LabeledPoint(values[0],values[1:])
	data = sc.textFile("data/mlib/ridge-data/lpsa.data")
	parsedData = data.map(parsePoint)

	# build the model
	model = LinearRegressionWithSGD.train(parseData,iterations=100,step=0.0000001)
	# Evalute the model on training data
	valuesAndPreds = parseData.map(lambda p:(p.label,model.predict(p.features))) 
	MSE = valuesAndPreds \
		.map(lambda vp:(vp[0]-vp[1])**2) \
		.reduce(lambda x,y:x+y)/valueAndPreds.count()
	print("Mean Squared Error = "+str(MSE))

	# save and load model
	model.save(sc,"target/tmp/pythonLinearRegressionWithSGDModel")
	sameModel = LinearRegressionModel.load(sc,'target/tmp/pythonLinearRegressionWithSGDModel')