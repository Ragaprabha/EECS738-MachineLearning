/* Class - WEKAConnectionStub
 * Used to call classifiers in weka.jar to build machine learning models and predict output
 *
 * Project Members : 
 * Chinnaswamy , Ragaprabha
 * Parikh, Bijal
 * Thippabhotla Sirisha
 */

package TestRun;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.util.Properties;
import java.util.Random;

import weka.classifiers.evaluation.*;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.ADTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class WEKAConnectionStub {

	public static final String TRAIN = "TRAIN";
	public static final String TEST = "TEST";
	public static final String TEST_NO_LABEL = "TEST_NO_LABEL";
	public Instances instancesTrain;
	public Instances instancesTest;
	public Instances instancesTestNoLabel;
	public Properties propUtil = new Properties();

	public static void main(String args[]) {
		WEKAConnectionStub impl = new WEKAConnectionStub();
		try {

			impl.loadProperties();
			impl.loadData(WEKAConnectionStub.TRAIN);
			impl.loadData(WEKAConnectionStub.TEST);
			impl.loadData(WEKAConnectionStub.TEST_NO_LABEL);
			//impl.tuneParameters();
			// impl.runClassifierWithFeatureSelection();
			// impl.runClassifier();
			 impl.getPredictions();
		} catch (Exception e) {
			System.out.println("Exception in WEKAConnectionStub main: " + e);
		}
	}

	public void loadProperties() {
		try {
			FileInputStream fs = new FileInputStream(
					"C:/Users/Bijal/workspace/RunWeka - Copy/src/TestRun/ClassifierProperty.txt");
			propUtil.load(fs);
		} catch (Exception e) {
			System.out.println("Exception in loading properties file " + e);
		}
	}

	public Instances getData(File inputFile) {
		ArffLoader arff = new ArffLoader();
		Instances instances = null;
		try {
			arff.setFile(inputFile);
			instances = arff.getDataSet();
		} catch (Exception e) {
			System.out.println("Error" + e);
		}
		return instances;
	}

	public void tuneParameters() {
		try {
			ParameterTuning.tuneParameters(instancesTrain, instancesTest);
		} catch (Exception e) {
			System.out.println("Exception while tuning Parameters " + e);
		}
	}

	public void runClassifierWithFeatureSelection() {
		try {
			FeatureOptimization.runClassifierWithFeatureSelection(
					instancesTrain, instancesTest, propUtil);
		} catch (Exception e) {
			System.out
					.println("Error while runClassifierWithFeatureSelection  "
							+ e);
		}
	}

	public void runRandomForest() {
		try {
			for (int i = 0; i < 20; i++) {
				System.out.println("Start: Iteration "+i);
				RandomForest rf = new RandomForest();
				rf.setMaxDepth(Integer.parseInt(propUtil.getProperty("K")));
				rf.setNumTrees(Integer.parseInt(propUtil.getProperty("I")));
				rf.buildClassifier(instancesTrain);
				System.out.println("Classifier Built");
				Random rand = new Random(12);
				Evaluation eval = new Evaluation(instancesTrain);
				eval.crossValidateModel(rf, instancesTrain, 10, rand);
				printDetails(eval);
				/*
				 * Evaluating model created using cross validation using
				 * remaining 30 % data
				 */
				Evaluation evaltest = new Evaluation(instancesTest);
				evaltest.evaluateModel(rf, instancesTest);
				printDetails(evaltest);
				System.out.println("End: Iteration "+i);
			}
		} catch (Exception e) {
			System.out.println("Exception in runRandomForest " + e);
		}
	}

	public void runADTree() {
		try {
			for (int i = 0; i < 20; i++) {
				System.out.println("Start: Iteration " + i);
				ADTree ad = new ADTree();
				ad.setNumOfBoostingIterations(Integer.parseInt(propUtil
						.getProperty("B")));
				ad.buildClassifier(instancesTrain);
				System.out.println("Classifier Built");
				Random rand = new Random(i);
				Evaluation eval = new Evaluation(instancesTrain);
				eval.crossValidateModel(ad, instancesTrain, 10, rand);
				printDetails(eval);
				/*
				 * Evaluating model created using cross validation using
				 * remaining 30 % data
				 */
				Evaluation evaltest = new Evaluation(instancesTest);
				evaltest.evaluateModel(ad, instancesTest, rand);
				printDetails(evaltest);
				System.out.println("End: Iteration " + i);
			}
		} catch (Exception e) {
			System.out.println("Exception in runADTree" + e);
		}
	}

	public void runLogistic() {
		try {
			for (int i = 0; i < 20; i++) {
				System.out.println("Start: Iteration " + i);
				Logistic logi = new Logistic();
				logi.setRidge(4*Math.pow(10,-4));
				logi.buildClassifier(instancesTrain);
				System.out.println("Classifier Built");
				Random rand = new Random(i);
				Evaluation eval = new Evaluation(instancesTrain);
				eval.crossValidateModel(logi, instancesTrain, 10, rand);
				printDetails(eval);
				/*
				 * Evaluating model created using cross validation using
				 * remaining 30 % data
				 */
				Evaluation evaltest = new Evaluation(instancesTest);
				evaltest.evaluateModel(logi, instancesTest, rand);
				printDetails(evaltest);
				System.out.println("End: Iteration " + i);
			}
		} catch (Exception e) {
			System.out.println("Exception in runLogistic" + e);
		}

	}

	public void runClassifier() throws Exception {
		runRandomForest();
		runADTree();
		runLogistic();
	}

	// Load the input data files in to Instances Object
	public void loadData(String type) {
		File inputFile = null;
		if (WEKAConnectionStub.TRAIN.equals(type)) {
			inputFile = new File(propUtil.getProperty("INPUT_TRAIN_70"));
			instancesTrain = (getData(inputFile));
			instancesTrain.setClassIndex(0);
		} else if (WEKAConnectionStub.TEST.equals(type)) {
			inputFile = new File(propUtil.getProperty("INPUT_TEST_30"));
			instancesTest = (getData(inputFile));
			instancesTest.setClassIndex(0);
		} else if (type.equals("TEST_NO_LABEL")) {
			inputFile = new File(propUtil.getProperty("INPUT_TEST_NO_LABEL"));
			instancesTestNoLabel = (getData(inputFile));
			instancesTestNoLabel.setClassIndex(0);
		}
	}

	public void getPredictions() {
		try {
			FileWriter fileWriter = new FileWriter(
					propUtil.getProperty("PREDICTION_FILE"));
			BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
			RandomForest rf = new RandomForest();
			rf.setMaxDepth(Integer.parseInt(propUtil.getProperty("K")));
			rf.setNumTrees(Integer.parseInt(propUtil.getProperty("I")));
			rf.buildClassifier(instancesTrain);
			for (int i = 0; i < instancesTestNoLabel.numInstances(); i++) {
				bufferedWriter.write(i + "");
				double pred = rf.classifyInstance(instancesTestNoLabel
						.instance(i));
				bufferedWriter.write("" + i);
				// bufferedWriter.write(", actual: " +
				// instancesTestNoLabel.classAttribute().value((int)
				// instancesTestNoLabel.instance(i).classValue()));
				// bufferedWriter.write(", predicted: ");
				bufferedWriter.write(instancesTestNoLabel.classAttribute()
						.value((int) pred));
				bufferedWriter.newLine();
			}
		} catch (Exception e) {
			System.out.println("Exception in getting predictions" + e);
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