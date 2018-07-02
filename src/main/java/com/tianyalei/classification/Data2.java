package com.tianyalei.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
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
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

/**
 * @author wuweifeng wrote on 2018/6/20.
 */
public class Data2 {
    public static void main(String[] args) throws IOException, InterruptedException {
        int seed = 123;
        double learningRate = 0.005;
        int batchSize = 120;
        int nEpochs = 10000;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 28;

        String trainPath = new ClassPathResource("/classification/data2.csv").getFile().getPath();

        //load the training data
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new File(trainPath)));
        //labelIndex是指结果的位置，这个样本结果是第0个位置
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 2, 2);
        DataSet allData = trainIter.next();
        allData.shuffle(123);
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);  //Use 65% of data for training

        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        //构建网络
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate, 0.9))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        for (int i = 0; i < nEpochs; i++) {
            model.fit(trainingData);
        }
        System.out.println(model.summary());
        System.out.println("Evaluate model....");
        Evaluation evaluation = new Evaluation(numOutputs);
            INDArray features = testData.getFeatureMatrix();
            INDArray lables = testData.getLabels();
            INDArray predicted = model.output(features, false);

            evaluation.eval(lables, predicted);

        System.out.println(evaluation.stats());
    }
}
