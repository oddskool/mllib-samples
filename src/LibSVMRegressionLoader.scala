package com.tedemis.sandbox

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

object LibSVMRegressionLoader {

  def loadLibSVMRegressionFile( sc: SparkContext,
                                path: String,
                                numFeatures: Int,
                                minPartitions: Int): RDD[LabeledPoint] = {
    val parsed = sc.textFile(path, minPartitions)
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
      val items = line.trim.split(' ')
      val label = items.head.toDouble
      var (indices, values) = items.tail
        .filter(item => !item.isEmpty)
        .map { item =>
          var index = 0
          var value = 0.0
          var indexAndValue: Array[String] = null
            try {
            indexAndValue = item.trim.split(':')
            index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
            value = indexAndValue(1).toDouble
          }
          catch {
            case nfe: NumberFormatException => {
              println("Error on line: <"+line+">")
              println("item: "+item)
              throw nfe
            }
          }
          (index, value)
        }
        .unzip
      indices = indices
      indices :+= 91
      values = values
      values :+= 1.0D
        (label, indices.toArray, values.toArray)
    }

    // Determine number of features.
    val d = if (numFeatures > 0) {
      numFeatures
    } else {
      parsed.persist(StorageLevel.MEMORY_ONLY)
      parsed.map { case (label, indices, values) =>
        indices.lastOption.getOrElse(0)
      }.reduce(math.max) + 1
    }

    parsed.map { case (label, indices, values) =>
      LabeledPoint(label, Vectors.sparse(d, indices, values))
    }

  }

}



