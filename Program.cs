using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace MinimalMachineLearning
{
    class Program
    {
        const string DATA_PATH = "data_large/data";
        const int GLYPH_WIDTH = 28;
        const int GLYPH_HEIGHT = 28;
        const int FEATURES = GLYPH_WIDTH * GLYPH_HEIGHT;
        const int FEATURES_BIAS = FEATURES + 1;
        const int HIDDEN_UNITS= 25;
        const int HIDDEN_UNITS_BIAS = HIDDEN_UNITS + 1;
        const int DIGITS = 10;
        const int FIRST_TRAINING_GLYPH = 0;
        const int LAST_TRAINING_GLYPH = 799;
        const int FIRST_TEST_GLYPH = 800;
        const int LAST_TEST_GLYPH = 999;
 
        static int _regularization = 1;
        static double _learningRate = 1f;
        static int _randomSeed = 1337;

        static List<byte[]> _data = new List<byte[]>(DIGITS);

        static double[] _features = new double[FEATURES_BIAS];
        static double[] _hidden = new double[HIDDEN_UNITS_BIAS];
        static double[] _output = new double[DIGITS];
        static double[] _label = new double[DIGITS];

        static double[,] _theta1 = new double[HIDDEN_UNITS, FEATURES_BIAS];
        static double[,] _theta2 = new double[DIGITS, HIDDEN_UNITS_BIAS];

        static double[,] _theta1_grad = new double[HIDDEN_UNITS, FEATURES_BIAS];
        static double[,] _theta2_grad = new double[DIGITS, HIDDEN_UNITS_BIAS];

        static void Main(string[] args)
        {
            Initialization();
            Training();
            Validation();
            Classification();
        }

        private static void Initialization()
        {
            Console.WriteLine("1.) INITIALIZATION");
            Console.Write("Loading Dataset...");
            ParseDataSet("../dataset/data");
            RandomizeThetas(0.12);
            Console.WriteLine("Neural Network Stats after random initialization:");
            PrintStats(FIRST_TRAINING_GLYPH, LAST_TRAINING_GLYPH);
        }

        private static void Training()
        {
            Console.WriteLine("2.) TRAINING");
            Console.WriteLine("Training on the " + (LAST_TRAINING_GLYPH + 1) + " examples per digit from the training set.");
            Console.WriteLine("Once you are happy with the achieved accuracy hit");
            Console.WriteLine("ESCAPE to proceed to the validation step." + Environment.NewLine);
            int iterations = 0;
            while (true)
            {
                EvaluateGradients(FIRST_TRAINING_GLYPH, LAST_TRAINING_GLYPH);
                GradientDescent(_learningRate);
                Console.Write('.');
                if (++iterations > 0 && iterations % 10 == 0)
                {
                    Console.WriteLine(Environment.NewLine + "Neural Network Stats after " + iterations + " iterations:");
                    PrintStats(FIRST_TRAINING_GLYPH, LAST_TRAINING_GLYPH);
                }
                if (Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.Escape)
                    break;
            }
            Console.WriteLine(Environment.NewLine + "Neural Network Stats after " + iterations + " iterations:");
            PrintStats(FIRST_TRAINING_GLYPH, LAST_TRAINING_GLYPH);
        }

        private static void Validation()
        {
            Console.WriteLine("3.) VALIDATION");
            Console.WriteLine("Validating on " + (LAST_TEST_GLYPH - FIRST_TEST_GLYPH) + " examples per digit not used for training.");
            PrintStats(FIRST_TEST_GLYPH, LAST_TEST_GLYPH);
        }

        private static void Classification()
        {
            Console.WriteLine("4.) APPLICATION");
            Console.WriteLine("Press ENTER to apply the NN to identify random glyphs!" + Environment.NewLine);
            Console.ReadKey();
            Random rnd = new Random();
            while (true)
            {
                int digit = rnd.Next(10);
                int index = rnd.Next(1000);
                PrintDigit(digit, index);
                double confidence = 0;
                int bestIndex = PredictIndex(digit, index, out confidence);
                Console.Write("Classified as a " + bestIndex + " with confidence " + confidence.ToString("0.00"));
                while (Console.ReadKey(true).Key != ConsoleKey.Enter) ; //Wait for the ENTER key
            }
        }
        
        private static void RandomizeThetas(double epsilon)
        {
            //seed RNG for reproducability
            Random rnd = new Random(_randomSeed);

            for (int x = 0; x < HIDDEN_UNITS; x++)
                for (int y = 0; y < FEATURES_BIAS; y++)
                    _theta1[x, y] = rnd.NextDouble() * 2 * epsilon - epsilon;

            for (int x = 0; x < DIGITS; x++)
                for (int y = 0; y < HIDDEN_UNITS_BIAS; y++)
                    _theta2[x, y] = rnd.NextDouble() * 2 * epsilon - epsilon;
        }
        
        private static void LoadFeatures(int digit, int index)
        {
            int offset = index * FEATURES;
            _features[0] = 1;//BIAS
            for (int i = 0; i < FEATURES; i++)
            {
                double value = _data[digit][i + offset] / 255f;
                _features[i + 1] = value;
            }
        }

        private static void Predict(int digit, int index)
        {
            //copy from data to features
            LoadFeatures(digit, index);

            //Calculate Hidden Layer from _features & _theta1!
            _hidden[0] = 1;//BIAS
            for (int i = 0; i < HIDDEN_UNITS; i++)
            {
                double value = 0.0f;
                for (int j = 0; j < FEATURES_BIAS; j++)
                    value += _features[j] * _theta1[i, j]; //This is where most CPU cycles are spent when making a prediction
                _hidden[i + 1] = Sigmoid(value);
            }

            //Calculate Output Layer from hidden layer and _theta2!
            for (int i = 0; i < DIGITS; i++)
            {
                double value = 0.0f;
                for (int j = 0; j < HIDDEN_UNITS_BIAS; j++)
                    value += _hidden[j] * _theta2[i, j];
                _output[i] = Sigmoid(value);
                Debug.Assert(_output[i] > 0 && _output[i] < 1);
            }
        }
        
        private static void EvaluateGradients(int from, int to)
        {
            //reset the arrays used to store the gradient. (this is where the function stores it's result, yeah side-fx are bad I know :P)
            for (int x = 0; x < HIDDEN_UNITS; x++)
                for (int y = 0; y < FEATURES_BIAS; y++)
                    _theta1_grad[x, y] = 0;

            for (int x = 0; x < DIGITS; x++)
                for (int y = 0; y < HIDDEN_UNITS_BIAS; y++)
                    _theta2_grad[x, y] = 0;
            
            //some temporary arrays
            double[] delta3 = new double[DIGITS];
            double[] delta2 = new double[HIDDEN_UNITS];
            double[] target = new double[DIGITS];
            int m = 0;
            for (int i = 0; i < DIGITS; i++) //foreach digits
            {
                target[i] = 1;
                for (int j = from; j < to + 1; j++) //foreach example in the training set
                {
                    m++; //count number of examples
                    Predict(i, j); //make a prediction with current theta (while training ~30% of the CPU cycles are spend here!)

                    //How much does the prediction differ from the correct answer?
                    for (int x = 0; x < DIGITS; x++)
                        delta3[x] = _output[x] - target[x];

                    //Use backpropagation to find out how the hidden unit should have been activated to lead to the correct answer and store the delta.
                    for (int y = 1; y < HIDDEN_UNITS_BIAS; y++)
                    {
                        double temp = 0;
                        for (int x = 0; x < DIGITS; x++)
                            temp += delta3[x] * _theta2[x, y];
                        delta2[y - 1] = temp * _hidden[y] * (1 - _hidden[y]);
                    }

                    //Form the sum of the found delta2 weighted by the activation in the input layer.
                    //This can be used to compute a gradient to minimize the cost function by changing the hidden layer along the gradient.
                    //Eg "moving down the slope of the cost function"
                    for (int x = 0; x < HIDDEN_UNITS; x++)
                        for (int y = 0; y < FEATURES_BIAS; y++)
                            _theta1_grad[x, y] += delta2[x] * _features[y]; //while training ~30% of the CPU cycles are spend here!

                    //Do the same thing for delta3
                    for (int x = 0; x < DIGITS; x++)
                        for (int y = 0; y < HIDDEN_UNITS_BIAS; y++)
                            _theta2_grad[x, y] += delta3[x] * _hidden[y];
                }
                target[i] = 0;
            }

            //form the mean based on the weighted sum - this results in theta1's gradient
            double invM = 1.0f / m;
            for (int x = 0; x < HIDDEN_UNITS; x++)
                for (int y = 0; y < FEATURES_BIAS; y++)
                    _theta1_grad[x, y] *= invM;

            //Do the same thing for theta2
            for (int x = 0; x < DIGITS; x++)
                for (int y = 0; y < HIDDEN_UNITS_BIAS; y++)
                    _theta2_grad[x, y] *= invM;

            //regularize e.g. penelize high values in theta1 excluding the bias term
            for (int x = 0; x < HIDDEN_UNITS; x++)
                for (int y = 1; y < FEATURES_BIAS; y++)
                    _theta1_grad[x, y] += _regularization * invM * _theta1[x, y];

            //regularize e.g. penelize high values in theta2 excluding the bias term
            for (int x = 0; x < DIGITS; x++)
                for (int y = 1; y < HIDDEN_UNITS_BIAS; y++)
                    _theta2_grad[x, y] += _regularization * invM * _theta2[x, y];
        }
        
        private static void GradientDescent(double learningRate)
        {
            for (int x = 0; x < HIDDEN_UNITS; x++)
                for (int y = 0; y < FEATURES_BIAS; y++)
                    _theta1[x, y] -= _theta1_grad[x, y] * learningRate;

            for (int x = 0; x < DIGITS; x++)
                for (int y = 0; y < HIDDEN_UNITS_BIAS; y++)
                    _theta2[x, y] -= _theta2_grad[x, y] * learningRate;
        }

        private static double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Exp(-value));
        }

        private static double CostFunction(int from, int to, double lambda = 0)
        {
            int m = 0;
            double delta = 0;
            for (int i = 0; i < DIGITS; i++)
            {
                for (int j = from; j < to + 1; j++)
                {
                    Predict(i, j);
                    for (int k = 0; k < DIGITS; k++)
                    {
                        double p = _output[k];
                        delta -= (i == k) ? Math.Log(_output[k]) : Math.Log(1 - _output[k]); //delta will always grow or stay equal
                    }
                    Debug.Assert(!double.IsInfinity(delta));
                    m++;
                }
            }
            double cost = delta / m;
            return cost + lambda * ComputeRegularizationBase() / (2.0f * m);
        }

        private static double ComputeRegularizationBase()
        {
            double squareSum = 0;
            for (int y = 1; y < FEATURES; y++) //skip the bias terms
                for (int x = 0; x < HIDDEN_UNITS; x++)
                    squareSum += _theta1[x, y] * _theta1[x, y];
            
            for (int y = 1; y < HIDDEN_UNITS; y++) //skip the bias terms
                for (int x = 0; x < DIGITS; x++)
                    squareSum += _theta2[x, y] * _theta2[x, y];

            return squareSum;
        }

        private static int PredictIndex(int digit, int index, out double confidence)
        {
            Predict(digit, index);
            int maxIdx = 0;
            confidence = 0;
            for (int k = 0; k < DIGITS; k++)
            {
                double p = _output[k];
                if (p > confidence)
                {
                    maxIdx = k;
                    confidence = p;
                }
            }
            return maxIdx;
        }

        private static double EvaluateAccuracy(int from, int to)
        {
            int error = 0;
            for (int i = 0; i < DIGITS; i++)
            {
                for (int j = from; j < to; j++)
                {
                    Predict(i, j);
                    int maxIdx = 0;
                    double max = 0;
                    for (int k = 0; k < DIGITS; k++)
                    {
                        double p = _output[k];
                        if (p > max)
                        {
                            maxIdx = k;
                            max = p;
                        }
                    }
                    if (maxIdx != i)
                        error++;
                }
            }
            int cnt = (to - from + 1) * 10;
            return (1 - error / (double)cnt);
        }

        private static void PrintStats(int from, int to)
        {
            Console.WriteLine("Cost: " + CostFunction(from, to).ToString("0.00"));
            Console.WriteLine("Regularized Cost: " + CostFunction(from, to, _regularization).ToString("0.00"));
            double percentage = EvaluateAccuracy(from, to) * 100.0f;
            Console.WriteLine("Accuracy: " + (int)percentage + "%" + Environment.NewLine);
        }

        static void PrintDigit(int digit, int index)
        {
            //prints a specific digit from the data set in a binary representation
            LoadFeatures(digit, index);
            for (int y = 0; y < GLYPH_HEIGHT; y++)
            {
                string line = "";
                for (int x = 0; x < GLYPH_WIDTH; x++)
                    line += Weight2Char(_features[x + y * GLYPH_WIDTH + 1]); //+1 == skip bias
                Console.WriteLine(line);
            }
        }

        static char[] ascii = new char[] {' ', '.', ',','-','~','+','x','X','0','#'};

        static char Weight2Char(double weight)
        {
            int idx = (int)Math.Ceiling(weight * 9);
            return ascii[idx];
        }

        static void ParseDataSet(string filepath)
        {
            //source: http://cis.jhu.edu/~sachin/digit/digit.html
            //Each file has 1000 training examples. 
            //Each training example is of size 28x28 pixels. 
            //The pixels are stored as unsigned chars(1 byte) and take values from 0 to 255.
            //The first 28x28 bytes of the file correspond to the first training example, the next 28x28 bytes correspond to the next example and so on.
            for (int i = 0; i < DIGITS; i++)
                _data.Add(File.ReadAllBytes(filepath + i));
        }
    }
}
