package AI;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.StringTokenizer;
import java.util.Arrays;
import java.util.ArrayList;
import jxl.Workbook;
import jxl.write.WritableWorkbook;
import jxl.write.WritableSheet;
import jxl.write.Label;

public class Neural_network {
	private static final int Total_Size = 2126;
	private static final int Feature_Size = 21;
	private static final int Training_Size = 1701;
	private static final int Test_Size = 425;
	private static final int Hidden_Layer_Count = 100;
	private static final int Output_Layer_Count = 3;
	private static final int epoch = 10000;
	private static final double Learning_Rate = 0.001;
	private static double target_function = 0.0;
	private static int error_count = 0;
	private static int total_count = 0;
	private static String testAccuracy_;

	private static double[][] input_layer = new double[Total_Size][Feature_Size];
	private static double[] hidden_layer = new double[Hidden_Layer_Count];
	private static double[] output_layer = new double[Output_Layer_Count];
	private static double[][] training_data = new double[Training_Size][Feature_Size];
	private static double[][] test_data = new double[Test_Size][Feature_Size];
	private static int[] answer_data = new int[Total_Size];
	private static int[] pre_answer_training = new int[Training_Size];
	private static int[] pre_answer_test = new int[Test_Size];
	private static int[][] answer_training = new int[Training_Size][Output_Layer_Count];
	private static int[][] answer_test = new int[Test_Size][Output_Layer_Count];
	private static double[][] weight_input_hidden = new double[Hidden_Layer_Count][Feature_Size];
	private static double[][] weight_hidden_output = new double[Output_Layer_Count][Hidden_Layer_Count];
	private static ArrayList<Double> performance = new ArrayList<Double>();

	private static void createData(double[][] input_data, int[] answer_data) {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader("data.txt"));
			for (int i = 0; i < Total_Size; i++) {
				StringTokenizer st = new StringTokenizer(br.readLine());
				for (int j = 0; j < Feature_Size; j++) {
					input_data[i][j] = Double.parseDouble(st.nextToken());
				}
				answer_data[i] = Integer.parseInt(st.nextToken()) - 1;
			}
			br.close();
		} catch (Exception e) {
		}

		for (int i = 0; i < Total_Size; i++) {
			int seed = (int) (Math.random() * Total_Size);

			double[] input_tmp = input_data[i];
			input_data[i] = input_data[seed];
			input_data[seed] = input_tmp;

			int answer_tmp = answer_data[i];
			answer_data[i] = answer_data[seed];
			answer_data[seed] = answer_tmp;
		}

		training_data = Arrays.copyOfRange(input_layer, 0, Training_Size);
		test_data = Arrays.copyOfRange(input_layer, Training_Size, Total_Size);
		pre_answer_training = Arrays.copyOfRange(answer_data, 0, Training_Size);
		pre_answer_test = Arrays.copyOfRange(answer_data, Training_Size, Total_Size);

		nspTranslate(pre_answer_training, answer_training);
		nspTranslate(pre_answer_test, answer_test);

		weightInitialize(weight_input_hidden, weight_hidden_output);
	}

	private static void nspTranslate(int[] before_answer, int[][] after_answer) {
		for (int i = 0; i < before_answer.length; i++) {
			Arrays.fill(after_answer[i], 0);

			if (before_answer[i] == 0)
				after_answer[i][0] = 1;
			else if (before_answer[i] == 1)
				after_answer[i][1] = 1;
			else if (before_answer[i] == 2)
				after_answer[i][2] = 1;
		}
	}

	private static void weightInitialize(double[][] weight_input_hidden, double[][] weight_hidden_output) {
		double ini;

		for (int i = 0; i < weight_input_hidden.length; i++) {
			for (int j = 0; j < weight_input_hidden[i].length; j++) {
				ini = Math.random() - 0.5;
				weight_input_hidden[i][j] = ini;
			}
		}

		for (int i = 0; i < weight_hidden_output.length; i++) {
			for (int j = 0; j < weight_hidden_output[i].length; j++) {
				ini = Math.random() - 0.5;
				weight_hidden_output[i][j] = ini;
			}
		}
	}

	public static void feedFoward(double[][] layer_, int instance) {
		feedForward_input_hidden(layer_, instance);
		feedForward_hidden_output();
	}

	private static void feedForward_input_hidden(double[][] layer_, int instance) {
		for (int i = 0; i < hidden_layer.length; i++) {
			double sum = 0.0;

			for (int j = 0; j < layer_[instance].length; j++) {
				sum += layer_[instance][j] * weight_input_hidden[i][j];
			}

			hidden_layer[i] = 1.0 / (1.0 + Math.exp(-sum));
		}
	}

	private static void feedForward_hidden_output() {
		for (int i = 0; i < output_layer.length; i++) {
			double sum = 0.0;

			for (int j = 0; j < hidden_layer.length; j++) {
				sum += hidden_layer[j] * weight_hidden_output[i][j];
			}

			output_layer[i] = 1.0 / (1.0 + Math.exp(-sum));
		}
	}

	private static int getAnswerIndex(double[] output_layer) {
		int index = 0;

		for (int i = 1; i < output_layer.length; i++) {
			if (output_layer[i] > output_layer[index])
				index = i;
		}

		return index;
	}

	private static void getErrorCount(int[][] answer_list, int answer_index, int x) {
		if (answer_list[x][answer_index] != 1)
			error_count++;
	}

	private static void getTargetFunction(int[][] answer_list, double[] output_layer, int x) {
		for (int i = 0; i < output_layer.length; i++) {
			target_function += 0.5 * Math.pow(answer_list[x][i] - output_layer[i], 2);
		}
	}

	private static double getGrad_output_hidden(double[] output_layer, double[] hidden_layer, int[][] answer, int x,
			int i, int j) {
		double output_ = output_layer[i];
		double answer_ = (double) answer[x][i];

		double grad = output_ * (output_ - answer_) * (1.0 - output_) * hidden_layer[j];

		return grad;
	}

	private static double getGrad_hidden_input(double[] hidden_layer, int[][] answer_training,
			double[][] weight_hidden_output, int x, int i, int j) {
		double grad_[] = new double[3];
		double grad_sum_ = 0.0;

		for (int k = 0; k < 3; k++) {
			grad_[k] = output_layer[k] * (output_layer[k] - (double) answer_training[x][k]) * (1.0 - output_layer[k])
					* hidden_layer[i] * weight_hidden_output[k][i];
			grad_sum_ += grad_[k];
		}

		double grad = grad_sum_ * (1.0 - hidden_layer[i]) * input_layer[x][j];

		return grad;
	}

	private static void backpropagation(int instance) {
		weightUpdate_output_hidden(instance);
		weightUpdate_hidden_input(instance);
	}

	private static void weightUpdate_output_hidden(int instance) {
		for (int i = 0; i < output_layer.length; i++) {
			for (int j = 0; j < hidden_layer.length; j++) {
				double grad = getGrad_output_hidden(output_layer, hidden_layer, answer_training, instance, i, j);

				weight_hidden_output[i][j] -= (Learning_Rate * grad);
			}
		}
	}

	private static void weightUpdate_hidden_input(int instance) {
		for (int i = 0; i < hidden_layer.length; i++) {
			for (int j = 0; j < training_data[instance].length; j++) {
				double grad = getGrad_hidden_input(hidden_layer, answer_training, weight_hidden_output, instance, i, j);

				weight_input_hidden[i][j] -= (Learning_Rate * grad);
			}
		}
	}

	public static void Training_(int e) {
		for (int instance = 0; instance < Training_Size; instance++) {
			feedFoward(training_data, instance);
			getTargetFunction(answer_training, output_layer, instance);
			backpropagation(instance);
			getErrorCount(answer_training, getAnswerIndex(output_layer), instance);
			total_count++;
		}

		performance.add(target_function);

		System.out.println("--------------------------");
		System.out.println("         Training         ");
		System.out.println("--------------------------");
		System.out.println("Epoch = " + e);
		System.out.format("Target Function = %.3f\n", target_function);
		System.out.format("Accuracy = %.3f%%\n", 100.0 - error_count / (total_count / 100.0));
		System.out.println("\n");

		total_count = 0;
		target_function = 0.0;
		error_count = 0;
	}

	public static void Test_() {
		for (int instance = 0; instance < Test_Size; instance++) {
			feedFoward(test_data, instance);
			getTargetFunction(answer_test, output_layer, instance);
			getErrorCount(answer_test, getAnswerIndex(output_layer), instance);
			total_count++;
		}

		double accuracy_ = 100.0 - error_count / (total_count / 100.0);
		testAccuracy_ = Double.toString(accuracy_);

		System.out.println("--------------------------");
		System.out.println("           Test           ");
		System.out.println("--------------------------");
		System.out.format("Target Function = %.3f\n", target_function);
		System.out.format("Accuracy = %.3f%%\n", accuracy_);
		System.out.println();
	}

	public static void createExcel(int hiddenLayer_, int epoch_, double learningRate_) {
		WritableWorkbook workbook = null;
		WritableSheet sheet = null;
		Label cell = null;

		String title = "h" + hiddenLayer_ + "_e" + epoch_ + "_l" + learningRate_ + ".xls";

		File file = new File(title);

		try {
			workbook = Workbook.createWorkbook(file);
			workbook.createSheet("AI", 0);
			sheet = workbook.getSheet(0);

			for (int i = 0; i < performance.size(); i++) {
				cell = new Label(0, i, Integer.toString(i));
				sheet.addCell(cell);

				cell = new Label(1, i, Double.toString(performance.get(i)));
				sheet.addCell(cell);
			}

			cell = new Label(0, performance.size(), "Accuracy");
			sheet.addCell(cell);

			cell = new Label(1, performance.size(), testAccuracy_);
			sheet.addCell(cell);

			workbook.write();
			workbook.close();

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		createData(input_layer, answer_data);

		for (int e = 1; e <= epoch; e++) {
			Training_(e);
		}

		Test_();

		createExcel(Hidden_Layer_Count, epoch, Learning_Rate);
	}

}