EECS738 - Final Project 
Chinnaswamy Ragaprabha - 2830383 
Parikh, Bijal - 2822565
Thippabhotla, Sirisha - 2853704


 * Modifying the input files:
   	1. The format of the input files have been modified from csv to arff
   	2. The class attribute has been moved to column 1 in the modified file
   	3. Therefore, we have set the class index as 0 in our data files
   	4. We have split the training data file into 2 files using Remove Percentage Filter in WEKA GUI
   	5. The first file contains 70% of the data and is used for building models. This file is referred as EECS738_TRAIN_70.arff
   	6. The second file contains 30% of the data and is used for testing the models built. This file is referred as EECS738_TEST_30.arff
   	7. We refer the test file without label as EECS738_TEST.arff
   
 * Setting up the project: 
	1. The project runs on weka version 3.7.13 and requires weka.jar, alternatingDecisionTrees.jar, classifierBasedAttributeSelection.jar
	2. Copy the folder RunWeka on to the system
	3. Place the data files in the folder RunWeka/data/
	4. Place the jars in the folder RunWeka/src/TestRun/ (we place the jars in the same folder as code, so that the cluster can easily pick up the jars for running jobs)
	5. The path of the data files needs to be updated in the ClassifierProperty.txt file present in RunWeka/src/TestRun/
	   Eg., INPUT_TRAIN_70=C:/Users/bparikh/Downloads/EECS738_Train_70.arff. Update the path of the EECS738_TRAIN_70 EECS738_TEST_30
	6. Remove the package name from the .java file as the package name doesn’t have any significance in cluster and will result in compilation error
	
 * Compiling the code:
	1. All Java files in the folder RunWeka/src/TestRun needs to be compiled. They can be compiled using the following command
		javac -cp “.:weka.jar:alternatingDecisionTrees.jar:classifierBasedAttributeSelection.jar” <JAVA_FILENAME>.java
	
 * Submitting jobs to the cluster:
	1. Place the pbs_script_serial.pbs in the folder RunWeka/src/TestRun/ 
	2. Set the timing in the script file to adjust the computation time (Computation time varies between algorithms)
	3. Command to execute the compiled java file needs to be added in the .pbs script. They can be set using the following command
		java -cp ".:weka.jar:alternatingDecisionTrees.jar:classifierBasedAttributeSelection.jar" <JAVA_FILENAME>
	4. Jobs are submitted to cluster using the following command
		qsub pbs_script_serial.pbs
	5. JobID will be generated as soon as the job is submitted to cluster. The log and error file will be created once the job is terminated

 * Parameter Tuning:
	1. Make sure the file paths are updated in CLassifierProperty.txt
	2. Open WEKAConnectionStub.java 
		- Comment the function calls to getPredictions(), runClassifier() and runClassifierWithFeatureSelection() in the main method
		- The method tuneParameters() will tune the parameters for all the 3 classifiers and print the results
		
 * Running classifier for 20 iterations with feature selection
 	1. Make sure the file paths are updated in CLassifierProperty.txt
	2. Open WEKAConnectionStub.java 
		- Comment the function calls to getPredictions(),runClassifier() and tuneParameters() in the main method
		- runClassifierWithFeatureSelection() will run the feature selection method for all the 3 classifiers for 20 different seeds and print the 				  	  results

 * Running classifier for 20 iterations without feature selection
 	1. Make sure the file paths are updated in CLassifierProperty.txt
	2. Open WEKAConnectionStub.java 
		- Comment the function calls to getPredictions(), tuneParameters()and runClassifierWithFeatureSelection() in the main method
		- runClassifier() will run the classification method for all the 3 classifiers for 20 different seeds and print the results

 * Making predictions:
    	1. Update the file paths for EECS738_TRAIN_70.arff and EECS738_TEST.arff in INPUT_TRAIN_70 
	2. Update the file path of PREDICTION_FILE key in ClassifierProperties.txt file.
	3. Update the tuned parameterValue for RandomForest in ClassifierProperties.txt, (i.e set I = 500, K = 0)
	4. Open WEKAConnectionStub.java and comment the function calls tuneParameters(), runClassifier() and runClassifierWithFeatureSelection() in the main method.
	5. The method getPredictions() will run the Random Forest classification algorithm and prints the predicted label