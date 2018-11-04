
from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS,LogsiticRegressionModel
from pyspark.millib.regression import LabelPoint

if __name__ == '__main__':
	sc = SparkContext(appName='PythonLogisticReressionWithLBFGSExample')

	# Load and parse the data
	def parsePoint(line):
		values = [float(x) for x in line.split(' ')]
		return LabelPoint(values[0],values[1:])

	data = sc.textFile("data/millib/sample_svm_data.txt")
	parsedData = data.map(parsePoint)

	#  Build the model
	model = LogisticRegressionWithLBFGS.train(parsePoint)
	#Evaluating the model on training data
	labelsAndPreds = parseData.map(lambda p:(p.label,model.predict(p.features)))
	trainErr = labelAndPreds.filter(lambda lp:lp[0]! =lp[1]).count()/float(parsedData.count())
	print('Training Error = '+ str(trainErr))

	# save and label model
	model.save(sc,"target/tmp/pythonLogisticRegressionWithLBFGSModel")
	sameModel = LogisticRegressionModel.load(sc,"target/tmp/pythonLogisticRegressionWithLBFGSModel")
	
