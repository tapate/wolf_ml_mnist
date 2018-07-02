package com.tianyalei.nd4j.lesson2;

import com.tianyalei.nd4j.lesson2.function.MathFunction;
import com.tianyalei.nd4j.lesson2.function.SinXDivXMathFunction;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * 非线性回归
 * @author wuweifeng wrote on 2018/6/26.
 */
public class RegressionMathFunctions {
    //Random number generator seed, for reproducability
    private static final int seed = 12345;
    //Number of epochs (full passes of the data)
    private static final int nEpochs = 2000;
    //How frequently should we plot the network output?
    private static final int plotFrequency = 500;
    //Number of data points
    private static final int nSamples = 1000;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    private static final int batchSize = 100;
    //Network learning rate
    private static final double learningRate = 0.01;
    private static final Random rng = new Random(seed);
    private static final int numInputs = 1;
    private static final int numOutputs = 1;

    public static void main(String[] args) {
        //Switch these two options to do different functions with different networks
        final MathFunction fn = new SinXDivXMathFunction();
        final MultiLayerConfiguration conf = getDeepDenseLayerNetworkConfiguration();
        //Generate the training data
        final INDArray x = Nd4j.linspace(-10, 10, nSamples).reshape(nSamples, 1);
        final DataSetIterator iterator = getTrainingData(x, fn, batchSize, rng);

        //Create the network
        final MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //Train the network on the full data set, and evaluate in periodically
        final INDArray[] networkPredictions = new INDArray[nEpochs / plotFrequency];
        for (int i = 0; i < nEpochs; i++) {
            iterator.reset();
            net.fit(iterator);
            if ((i + 1) % plotFrequency == 0) {
                networkPredictions[i / plotFrequency] = net.output(x, false);
            }
        }

        //Plot the target data and the network predictions
        //plot(fn, x, fn.getFunctionValues(x), networkPredictions);
    }


    /**
     * Create a DataSetIterator for training
     *
     * @param x
     *         X values
     * @param function
     *         Function to evaluate
     * @param batchSize
     *         Batch size (number of examples for every call of DataSetIterator.next())
     * @param rng
     *         Random number generator (for repeatability)
     */
    private static DataSetIterator getTrainingData(final INDArray x, final MathFunction function, final int batchSize, final Random rng) {
        final INDArray y = function.getFunctionValues(x);
        final DataSet allData = new DataSet(x, y);

        final List<DataSet> list = allData.asList();
        Collections.shuffle(list, rng);
        return new ListDataSetIterator(list, batchSize);
    }

    /**
     * Returns the network configuration, 2 hidden DenseLayers of size 50.
     */
    private static MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration() {
        final int numHiddenNodes = 100;
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.RELU).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();
    }
}
