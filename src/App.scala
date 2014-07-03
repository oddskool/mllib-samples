package com.tedemis.sandbox

import org.apache.spark.mllib.regression._
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
      30)
    val testDataRDD = LibSVMRegressionLoader.loadLibSVMRegressionFile(
      sc,
      testingSetPath,
      -1,
      30)
    (trainDataRDD, testDataRDD)
  }

  def evaluate(valuesAndPreds: RDD[Tuple2[Double, Double]]) = {
    println("Sample predictions")
    valuesAndPreds.take(10).map(t => println(t))
    val absoluteErrors = valuesAndPreds.map{case(v, p) => Math.abs(v - p)}.cache()
    val stableAbsoluteErrors = absoluteErrors.filter(e => !e.isNaN)
    println(stableAbsoluteErrors.count()+" stable predictions out of "+absoluteErrors.count())
    val meanError = stableAbsoluteErrors.mean()
    val stdError = stableAbsoluteErrors.stdev()
    println("Mean Error = " + meanError + " +/- " + stdError)
    val MSE = stableAbsoluteErrors.map{e => math.pow(e, 2)}.mean()
    println("Mean Squared Error = " + MSE)
    val RMSE = Math.sqrt(MSE)
    println("Root Mean Squared Error = " + RMSE)
  }

  def initializeLocalSparkContext() = {
    val conf = new SparkConf()
      .setAppName("MLLib POC")
      .setMaster("local[4]")
      .setSparkHome("/usr/local/Cellar/apache-spark/1.0.0/libexec/")
      .set("spark.executor.memory", "1g")
    val sc = new SparkContext(conf)
    sc
  }

  def predict(testingData: RDD[LabeledPoint], model: LinearRegressionModel): RDD[Tuple2[Double, Double]] = {
    testingData.map { point: LabeledPoint =>
      val prediction = model.predict(point.features)
      Tuple2[Double, Double](point.label, prediction.asInstanceOf[Double])
    }
  }

  def main(args : scala.Array[scala.Predef.String]) : scala.Unit = {

    val sc = initializeLocalSparkContext()

    val (trainingData, testingData): (RDD[LabeledPoint], RDD[LabeledPoint]) =
      loadLibSVMRegressionDataSets(sc, "YearPredictionMSD", "YearPredictionMSD.t")

    var trainedModel = LinearRegressionWithSGD.train(trainingData, 100, 1, 0.5)

    for (i <- 1 to 10) {

      println("Model: " + trainedModel.intercept + " / " + trainedModel.weights)

      val valuesAndPredictions = predict(testingData, trainedModel)

      evaluate(valuesAndPredictions)

      trainedModel =  LinearRegressionWithSGD.train(trainingData, 100, 1, 0.5)
    }
  }

}

