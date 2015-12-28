/* Class - ParameterTuning
 * Contains method to tune parameters for RandomForest, ADTree and Logistic Regression
 *
 * Project Members : 
 * Chinnaswamy , Ragaprabha
 * Parikh, Bijal
 * Thippabhotla Sirisha
 */

package TestRun;

import java.util.Random;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.ADTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class ParameterTuning {

	public static void tuneParameters(Instances instancesTrain, Instances instancesTest) throws Exception {
		tuneRandomForestParameters(instancesTrain, instancesTest);
		tuneLogisticParameters(instancesTrain);
		tuneADTreeParameters(instancesTrain);
	}

	public static void tuneRandomForestParameters(Instances instancesTrain,Instances instancesTest) {
		try {
			for (int i = 100; i <= 1000; i += 100) {
				for (int j = 0; j <= 11; j += 1) {
		//	int j = 0;
		//	int i = 400;
					RandomForest aClassifier = new RandomForest();
					aClassifier.setNumTrees(i);
					aClassifier.setMaxDepth(j);
					aClassifier.buildClassifier(instancesTrain);
					Evaluation eval = new Evaluation(instancesTrain);
					Random rand = new Random(0);
					eval.crossValidateModel(aClassifier, instancesTrain, 10,
							rand);
					System.out
							.println("Running for tuning parameters for RandomForest ");
					System.out.println("I : " + i);
					System.out.println("K : " + j);
					printDetails(eval);
					
					Evaluation evaltest = new Evaluation(instancesTest);
					evaltest.evaluateModel(aClassifier, instancesTest);
					System.out.println("Reevaluate on remaining 30%");
					printDetails(evaltest);
					
				}
			}
		} catch (Exception e) {
			System.out
					.println("Exception while tuning Random Forest Parameter : "
							+ e);
		}
	}

	public static void tuneLogisticParameters(Instances instancesTrain) {
		try {
			for (int i = 100; i <= 1000; i += 100) {
				for (int j = 0; j <= 11; j += 1) {
					Logistic aClassifier = new Logistic();
					double val = Math.pow(i, j);
					aClassifier.setRidge(val);
					aClassifier.buildClassifier(instancesTrain);
					Evaluation eval = new Evaluation(instancesTrain);
					Random rand = new Random(0);
					eval.crossValidateModel(aClassifier, instancesTrain, 10,
							rand);
					System.out
							.println("Running for tuning parameters for Logistic ");
					System.out.println("R : " + val);
					printDetails(eval);
				}
			}
		} catch (Exception e) {
			System.out.println("Exception while tuning Logistic Parameters : "
					+ e);
		}
	}

	public static void tuneADTreeParameters(Instances instancesTrain) {
		try {
			for (int j = 10; j <= 100; j += 10) {
				ADTree aClassifier = new ADTree();
				aClassifier.setNumOfBoostingIterations(j);
				aClassifier.buildClassifier(instancesTrain);
				Evaluation eval = new Evaluation(instancesTrain);
				Random rand = new Random(0);
				eval.crossValidateModel(aClassifier, instancesTrain, 10, rand);
				System.out.println("Running for tuning parameters for ADTree ");
				System.out.println("B : " + j);
				printDetails(eval);
			}
		} catch (Exception e) {
			System.out
					.println("Exception while tuning Random Forest Parameter : "
							+ e);
		}
	}

	public static void printDetails(Evaluation eval) {
		System.out.println("***************************");
		System.out.println("Confusion Matrix : TP "
				+ eval.confusionMatrix()[0][0] + " FN "
				+ eval.confusionMatrix()[0][1] + " FP "
				+ eval.confusionMatrix()[1][0] + " TN "
				+ eval.confusionMatrix()[1][1]);
		System.out.println("***************************");
		System.out.println(eval.toSummaryString());
		System.out.println("***************************");
		System.out.println("Area Under ROC : " + eval.areaUnderROC(0));
		System.out.println("Area Under PRC : " + eval.areaUnderPRC(0));
		System.out.println("MCC : " + eval.matthewsCorrelationCoefficient(0));
		System.out.println(" Weighted MCC : "
				+ eval.weightedMatthewsCorrelation());
		System.out
				.println("********************Weighted Values********************");
		System.out.println(" Weighted Area under ROC : "
				+ eval.weightedAreaUnderROC());
		System.out.println(" Weighted Area under PRC : "
				+ eval.weightedAreaUnderPRC());
		System.out.println(" Weighted True Positive : "
				+ eval.weightedTruePositiveRate());
		System.out.println(" Weighted True Negative : "
				+ eval.weightedTrueNegativeRate());
		System.out.println(" Weighted False Positive : "
				+ eval.weightedFalsePositiveRate());
		System.out.println(" Weighted False Negative : "
				+ eval.weightedFalseNegativeRate());
		System.out.println(" Weighted MCC : "
				+ eval.weightedMatthewsCorrelation());
		System.out.println("***************************");

	}
}
