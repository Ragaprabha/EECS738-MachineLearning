/* Class - FeatureOptimization
 * Contains methods for feature selection and running classifiers with 
 *
 * Project Members : 
 * Chinnaswamy , Ragaprabha
 * Parikh, Bijal
 * Thippabhotla Sirisha
 */

package TestRun;

import java.util.Properties;
import java.util.Random;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.trees.ADTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class FeatureOptimization {

	public static void runClassifierWithFeatureSelection(Instances instancesTrain,
			Instances instancesTest, Properties properties) {
		runClassifierWithFeatureSelectionRF(instancesTrain,instancesTest,properties);
		runClassifierWithFeatureSelectionADTree(instancesTrain,instancesTest,properties);	
	}
	
	public static void runClassifierWithFeatureSelectionRF(Instances instancesTrain,
			Instances instancesTest, Properties properties){
		try {
			// Optimizing greedyStepwise with respect to number of attributes to
			// be selected
			for (int i = 2; i <= 5; i++) {
				AttributeSelectedClassifier asc = new AttributeSelectedClassifier();
				RandomForest rf = new RandomForest();
				rf.setNumTrees(Integer.parseInt(properties.getProperty("I")
						.toString()));
				rf.setMaxDepth(Integer.parseInt(properties.getProperty("K")
						.toString()));
				rf.setSeed(1);
				CfsSubsetEval cse = new CfsSubsetEval();
				GreedyStepwise gs = new GreedyStepwise();
				gs.setConservativeForwardSelection(true);
				gs.setSearchBackwards(true);
				gs.setNumToSelect(10);
				asc.setSearch(gs);
				asc.setEvaluator(cse);
				asc.setClassifier(rf);
				Evaluation eval = new Evaluation(instancesTrain);
				asc.buildClassifier(instancesTrain);
				eval.crossValidateModel(asc, instancesTrain, 10, new Random(i));
				System.out
						.println("Running for tuning parameters for cfsSubSetEval seed 6");
				printDetails(eval);

				// Evaluate model
				Evaluation evaltest = new Evaluation(instancesTest);
				evaltest.evaluateModel(asc, instancesTest);
				printDetails(evaltest);
			}
		} catch (Exception e) {
			System.out.println("Error in Attribute Selection " + e);
		}
	}
	
	public static void runClassifierWithFeatureSelectionADTree(Instances instancesTrain,
			Instances instancesTest, Properties properties){
		try{
			//Optimizing BestFirst with respect to number of attributes to be selected
			for(int i = 10; i <=20; i++){
			AttributeSelectedClassifier asc = new AttributeSelectedClassifier();
			ADTree ad = new ADTree();
			ad.setNumOfBoostingIterations(Integer.parseInt(properties.getProperty("B").toString()));
			CfsSubsetEval cse = new CfsSubsetEval();
			BestFirst bf = new BestFirst();
			String[] optionsbestfirst = { "-D", "1", "-N", "5" };
			try {
				bf.setOptions(optionsbestfirst);
			} catch (Exception e) {
				e.printStackTrace();
			}
			asc.setSearch(bf);
			asc.setEvaluator(cse);
			asc.setClassifier(ad);
		
			Evaluation eval = new Evaluation(instancesTrain);
			asc.buildClassifier(instancesTrain);
			System.out.println("Model built");
			eval.crossValidateModel(asc, instancesTrain, 10, new Random(i));
			System.out.println("Running for tuning parameters for cfsSubSetEval ");
			printDetails(eval);
			}
		}
		catch(Exception e){
			System.out.println("Error in Attribute Selection "+e);
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
