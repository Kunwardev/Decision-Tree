import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Random;
import java.util.Set;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.MaximizeAction;

public class dtree {
	
	static int totalObjects;
	static int[] classObjects;
	static double[] classProb;
	static int columnsWithClass;
	static int columns;
	static String option;
	static int nodeId = 1;
	
	public static void main(String[] args) {
		
	String training_file;
	String test_file;
	int pruning_thr;
	
	if(args.length < 4) {
		System.out.println("Please enter 4 command arguments as follows: training_file, test_file, option, pruning_thr");
		return;
	}
	training_file = args[0];
	test_file = args[1];
	option = args[2];
	pruning_thr = Integer.parseInt(args[3]);
	double[][] inputMatrix = null;
	double[][] testMatrix = null;
	try {
		inputMatrix = readInputFile(training_file);
		testMatrix = readInputFile(test_file);
		
	} catch (IOException e) {
		e.printStackTrace();
	}
	
	int attributes = columns;
	
	dtree dtree1 = new dtree();
	int rounds = 1;
	if(option.equals("forest3")) {
		rounds = 3;
	} else if(option.equals("forest15")) {
		rounds = 15;
	}
	ArrayList<Node> forest = new ArrayList<Node>();
	for(int i = 0; i<rounds; i++ ) {
		Node root = dtree1.DTL(inputMatrix, attributes, classProb, pruning_thr);
		forest.add(root);
		dtree1.BFS(root,i);
		nodeId = 1;
	}
	
	dtree1.classify(forest, testMatrix);
	}
	
	private void classify(ArrayList<Node> forest, double[][] testMatrix) {

		int rows = testMatrix.length;
		double totalAccuracy = 0;
		double[][] distribution = new double[forest.size()][classProb.length];
		for(int i=0; i<rows; i++) {
			
			for(int t=0;t<forest.size(); t++) {
				Node current = forest.get(t);
				Node parent = null;
				double[] obj = testMatrix[i];
				
				while(current != null) {
					double value = obj[current.best_attribute];
					double threshold = current.best_threshold;
					if(value < threshold) {
						parent = current;
						current = current.left_child;
					} else {
						parent = current;
						current = current.right_child;
					}
				}
				distribution[t] = parent.output_class;
				
			}
			double[] finalDist = new double[classProb.length];
			for(int t=0;t<forest.size(); t++) {
				for(int c=0; c<classProb.length; c++) {
					finalDist[c] = finalDist[c] + distribution[t][c];
				}
			}
			for(int c=0; c<classProb.length; c++) {
				finalDist[c] = finalDist[c]/forest.size();
			}
			
			ArrayList<Integer> predictedList = maxIndex(finalDist);
			int actualClass = (int) testMatrix[i][columns];
			
			double accuracy = findAccuracy(predictedList, actualClass);
			totalAccuracy = totalAccuracy + accuracy;
			System.out.printf("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n", i, predictedList.get(0), actualClass, accuracy);
		}
		
		double classification_accuracy = totalAccuracy/rows;
		System.out.printf("classification accuracy=%6.4f\n", classification_accuracy);

	}

	public double findAccuracy(ArrayList<Integer> predictedList, int actualClass) {
		double accuracy = 0;
		
		if(predictedList.size() > 1) {
			for(int p : predictedList) {
				if(p == actualClass) {
					accuracy = 1/predictedList.size();
					break;
				}
			}
		}
		else if (predictedList.size() == 1 && predictedList.get(0) == actualClass) {
			accuracy = 1;
		}
		return accuracy;
	}
	private void BFS(Node root, int tree_id) {
		
		LinkedList<Node> queue = new LinkedList<Node>();
		queue.addFirst(root);
		
		while(!queue.isEmpty()) {
			Node d = queue.removeLast();
			//Remove this
				/*if(d.output_class != null) {
					for(double t : d.output_class) {
						System.out.print(t+"   ");
					}
					System.out.println();
				}*/
			//
			int feature_id = d.best_attribute;
			double threshold = d.best_threshold;
			double gain = d.max_gain;
			if(d.left_child != null) {
				queue.addFirst(d.left_child);
			}
			if(d.right_child != null) {
				queue.addFirst(d.right_child);
			}
			System.out.printf("tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f\n", tree_id, nodeId++, feature_id, threshold, gain);
		}
	}

	public Node DTL(double[][] examples, int attributes, double[] defaultDist, int pruning_thr) {
		Node tree = new Node();
		
		if(examples.length < pruning_thr) {
			
	        tree.output_class = defaultDist;
			
		}
		else if(isSame(examples)) {
	       
	       tree.output_class = distribution(examples); 
		}
		else {
			
			//[attribute, threshold]
			BestAttThr bestAttThr = chooseAttribute(examples, attributes); //CHOOSE_ATTRIBUTE(examples, attributes);
			
			// tree = a new decision tree with root test (best_attribute, best_threshold)
	        tree.best_attribute = bestAttThr.best_attribute;
	        tree.best_threshold = bestAttThr.best_threshold;
	        tree.max_gain = bestAttThr.max_gain;
	        //Split examples
	        SplitBean splitBean = split(examples, bestAttThr);
	        // examples_left = {elements of examples with best_attribute < threshold}
	        double[][] examples_left = splitBean.examples_left;
	        // examples_right = {elements of examples with best_attribute >= threshold}
	        double[][] examples_right = splitBean.examples_right;
			
	        // tree.left_child = DTL(examples_left, attributes, DISTRIBUTION(examples))
			tree.left_child = DTL(examples_left, attributes, distribution(examples), pruning_thr);
	        
			// tree.right_child = DTL(examples_right, attributes, DISTRIBUTION(examples))
			tree.right_child = DTL(examples_right, attributes, distribution(examples), pruning_thr);
		}

		return tree;
	}
	
	public ArrayList<Integer> maxIndex(double[] distribution) {
		ArrayList<Integer> maxList = new ArrayList<Integer>();
		
		double max = Double.MIN_VALUE;
		for(int i=0;i<distribution.length;i++) {
			if(distribution[i] > max) {
				maxList = new ArrayList<Integer>();
				max = distribution[i];
				maxList.add(i);
			} else if(distribution[i] == max) {
				maxList.add(i);
			}
		}
		return maxList;
	}
	
	private double[] distribution(double[][] examples) {
		double[] currentClassProb = new double[classObjects.length];
		
		int total = examples.length;
		for(int i=0; i<total; i++) {
			int classNo = (int) examples[i][columns];
			currentClassProb[classNo]++;
		}
		for(int i=0; i<classObjects.length; i++) {
			
			currentClassProb[i] = currentClassProb[i]/total;
		}
		return currentClassProb;
	}

	private SplitBean split(double[][] examples, BestAttThr bestAttThr) {
		int A = bestAttThr.best_attribute;
		double threshold = bestAttThr.best_threshold;
		ArrayList<double[]> examples_left = new ArrayList<double[]>();
		ArrayList<double[]> examples_right = new ArrayList<double[]>();
		
		for(int i=0; i<examples.length; i++) {
			if(examples[i][A] < threshold) {
				examples_left.add(examples[i]);
			}
			else {
				examples_right.add(examples[i]);
			}
		}
		
		double[][] examples_left_Array = new double[examples_left.size()][columns];
		double[][] examples_right_Array = new double[examples_right.size()][columns];
		
 		for(int i=0; i<examples_left.size(); i++) {
 			examples_left_Array[i] =  examples_left.get(i);
		}
		for(int i=0; i<examples_right.size(); i++) {
			examples_right_Array[i] = examples_right.get(i);
		}
		SplitBean splitBean = new SplitBean(examples_left_Array, examples_right_Array);
		return splitBean;
	}

	private BestAttThr chooseAttribute(double[][] examples, int attributes) {
		
		double max_gain = -1;
		double best_threshold = -1;
		int best_attribute = -1;
		int total = examples.length;
	    // For 'optimized' 
	    if(option.equals("optimized")) {
	        for(int i=0; i< attributes; i++) {
	        	
	        	double[] attribute_values = new double[total];
	        	for(int j=0; j<total; j++) {
	        		attribute_values[j] = examples[j][i];
	        	}
	            
	            double L = min(attribute_values);
	            double M = max(attribute_values);
	            for(int K = 1; K<=50; K++) { 
	                double threshold = L + K*(M-L)/51;
	                double gain = informationGain(examples, i, threshold);
	                if(gain > max_gain) {
	                    max_gain = gain;
	                    best_attribute = i;
	                    best_threshold = threshold;
	                }
	            }
	        }
	    }
	    // For 'random'     
	    else {
	        int A = randomInRange(1, attributes);
	        
	        double[] attribute_values = new double[total];
        	for(int j=0; j<total; j++) {
        		attribute_values[j] = examples[j][A];
        	}
        	
        	double L = min(attribute_values);
            double M = max(attribute_values);
            for(int K = 1; K<=50; K++) { 
                double threshold = L + K*(M-L)/51;
                double gain = informationGain(examples, A, threshold);
                if(gain > max_gain) {
                    max_gain = gain;
                    best_attribute = A;
                    best_threshold = threshold;
                }
            }
	        
	    }
	    BestAttThr bestAttThr = new BestAttThr(best_attribute, best_threshold, max_gain);
		return bestAttThr;
	}

	private int randomInRange(int min, int max) {
		Random random = new Random(); 
		int value = 0; 
		value = random.nextInt((max - min) + 1) + min;
		return value;
	}

	private double informationGain(double[][] examples, int i, double threshold) {
		
		ArrayList<double[]> k1 = new ArrayList<double[]>();
		ArrayList<double[]> k2 = new ArrayList<double[]>();
		int total = examples.length;
		for(int j=0; j<total; j++) {
			if(examples[j][i] < threshold) {
				k1.add(examples[j]);
			}
			else {
				k2.add(examples[j]);
			}
		}
	
		double H = calH(examples);
	    double K = total;
	    double K1 = k1.size();
	    double K2 = k2.size();
	    double E1 = calH(toArray(k1));
	    double E2 = calH(toArray(k2));
	    double gain = 0;
	    if(K1 != 0 && K2 !=0) {
	    	gain = H - (K1/K) * (E1) - (K2/K) * (E2); 
	    }
	    
		return gain;
	}
	
	public double[][] toArray(ArrayList<double[]> k1) {
		
		double[][] E = new double[k1.size()][columns];
		
		for(int i=0; i<k1.size(); i++) {
			E[i] = k1.get(i);
		}
		return E;
	}
	
	public double calH(double[][] examples) {
		double[] dist = distribution(examples);
		double H = 0;
		for(int h=0;h<dist.length;h++) {
			double temp = dist[h];
			if(temp != 0) {
				H = H - temp * (Math.log(temp)/Math.log(2));
			}
		}
		return H;
	}

	public double min(double[] attribute_values) {
		double min = Double.MAX_VALUE;
		for(int i=0; i< attribute_values.length; i++) {
			if(min > attribute_values[i]) {
				min = attribute_values[i];
			}
		}
		return min;
	}
	
	public double max(double[] attribute_values) {
		double max = Double.MIN_VALUE;
		for(int i=0; i< attribute_values.length; i++) {
			if(max < attribute_values[i]) {
				max = attribute_values[i];
			}
		}
		return max;
	}
	
	public boolean isSame(double[][] examples) {
		boolean flag = true;
		int classColumn = examples[0].length - 1;
		
		for(int i=0; i<examples.length - 1;i++) {
			if(examples[i][classColumn] != examples[i+1][classColumn]) {
				flag = false;
				break;
			}
		}
		return flag;
	}
	
	public static double[][] readInputFile(String path) throws IOException {
		Set<Integer> classSet = new HashSet<Integer>();
		double[][] testMatrixClass = null;
		
		ArrayList<String> input = readFile(path);
		int rowIndex = 0;
		for (String line : input) {
			
			String[] values = line.trim().split("\\s+");		
			if(testMatrixClass==null) testMatrixClass = new double[input.size()][values.length];
			classSet.add(Integer.parseInt(values[values.length-1]));
			columns = values.length - 1;
			columnsWithClass = values.length;
			for(int i = 0; i< values.length; i++) {
				double currentValue = Double.parseDouble(values[i]);
				testMatrixClass[rowIndex][i] = currentValue; 					
			}
			rowIndex++;	
		}
		classObjects = new int[classSet.size()];
		classProb = new double[classSet.size()];
		
		for(int i = 0; i<totalObjects-1; i++) {
			int classNO = (int) testMatrixClass[i][columns];
			 classObjects[classNO] = classObjects[classNO] + 1;
		} 
		
		return testMatrixClass;
	}
	
	public static ArrayList<String> readFile(String path) throws IOException {
		
		ArrayList<String> fileData = new ArrayList<String>();
		BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
		String str = reader.readLine();
		while(str != null) {
			fileData.add(str);
			str = reader.readLine();
		}
		reader.close();
		totalObjects = fileData.size();
		return fileData;
	}
	
	class Node {
		public Node left_child;
		public Node right_child;
		public int best_attribute;
		public double best_threshold;
		public double max_gain;
		public double[] output_class;
	}
	
	class BestAttThr {
		BestAttThr(int aBest_attribute, double aBest_threshold, double aMax_gain) {
			best_attribute = aBest_attribute;
			best_threshold = aBest_threshold; 
			max_gain = aMax_gain;
		}
		 int best_attribute;
		 double best_threshold;
		 double max_gain;
	}
	class SplitBean {
		SplitBean(double[][] anExamples_left, double[][] anExamples_right) {
			examples_left = anExamples_left;
			examples_right = anExamples_right; 
		}
		double[][] examples_left;
		double[][] examples_right;
	}
}
