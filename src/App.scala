package com.tedemis.sandbox

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._

/**
 * Created by eustache on 30/06/2014.
 */
object App {

  def loadLibSVMRegressionDataSets(sc: SparkContext,
                                  trainingSetPath: String,
                                  testingSetPath: String): Tuple2[RDD[LabeledPoint], RDD[LabeledPoint]] = {
    val trainDataRDD = LibSVMRegressionLoader.loadLibSVMRegressionFile(
      sc,
      trainingSetPath,
      -1,
      10)
    val testDataRDD = LibSVMRegressionLoader.loadLibSVMRegressionFile(
      sc,
      testingSetPath,
      -1,
      10)
    (trainDataRDD, testDataRDD)
  }

  def evaluate(valuesAndPreds: RDD[Tuple2[Double, Double]]) = {
    val meanError = valuesAndPreds.map{case(v, p) => v - p}.mean()
    println("Mean Error = " + meanError)
    val squareRootedMeanError = valuesAndPreds.map{case(v, p) => Math.sqrt(v - p)}.mean()
    println("Square Root Mean Error = " + squareRootedMeanError)
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((1e-9 + v - p), 2)}.mean()
    println("Mean Squared Error = " + MSE)
  }

  def initializeLocalSparkContext() = {
    val conf = new SparkConf()
      .setAppName("MLLib POC")
      .setMaster("local[1]") // set 4 local executors
      .setSparkHome("/usr/local/Cellar/apache-spark/1.0.0/libexec/")
      .set("spark.executor.memory", "1g")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    sc
  }

  def predict(testingData: RDD[LabeledPoint], model: DecisionTreeModel): RDD[Tuple2[Double, Double]] = {
    testingData.map { point: LabeledPoint =>
      val prediction = model.predict(point.features)
      Tuple2[Double, Double](point.label, prediction.asInstanceOf[Double])
    }
  }

  def main(args : scala.Array[scala.Predef.String]) : scala.Unit = {

    val sc = initializeLocalSparkContext()

    val (trainingData, testingData): (RDD[LabeledPoint], RDD[LabeledPoint]) =
      loadLibSVMRegressionDataSets(sc, "YearPredictionMSD", "YearPredictionMSD.t")

    val model = DecisionTree.train(trainingData, Algo.Regression, Variance, 5)

    val valuesAndPredictions = predict(testingData, model)

    evaluate(valuesAndPredictions)
  }

}

