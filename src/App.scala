package com.tedemis.sandbox

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._

/**
 * Created by eustache on 30/06/2014.
 */
object App {

  def main(args : scala.Array[scala.Predef.String]) : scala.Unit = {
    val conf = new SparkConf()
      .setAppName("MLLib POC")
      .setMaster("local[4]")
      .setSparkHome("/usr/local/Cellar/apache-spark/1.0.0/libexec/")
    val sc = new SparkContext(conf)
    val path: String = "YearPredictionMSD.t"
    println(sc.textFile(path).count())
    val housingData = LibSVMRegressionLoader.loadLibSVMRegressionFile(
      sc,
      path,
      -1,
      10)
    housingData.cache()
//    val example = housingData.take(1)(0)
//    println(example.label+" - "+example.features)

    // Split data into training and test.
    val splits = housingData.randomSplit(Array(0.8, 0.2), seed = 42L)
    val training = splits(0)
    val test = splits(1)

    // Run training algorithm to build the model
    val model = DecisionTree.train(training, Algo.Regression, Variance, 15)

    // Compute raw scores on the test set.
    val valuesAndPreds = housingData.map { point: LabeledPoint =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

//    valuesAndPreds.take(100).map{case(v, p) => println(v+" / "+p+" = "+(v-p)+" => "+math.pow((v - p), 2))}

    // Get evaluation metrics.
    val meanError = valuesAndPreds.map{case(v, p) => v - p}.mean()
    println("Mean Error = " + MSE)
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((1e-9 + v - p), 2)}.mean()
    println("training Mean Squared Error = " + MSE)
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((1e-9 + v - p), 2)}.mean()
    println("training Mean Squared Error = " + MSE)

  }

}

