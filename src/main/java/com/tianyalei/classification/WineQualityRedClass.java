package com.tianyalei.classification;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

/**
 * @author wuweifeng wrote on 2018/6/20.
 */
public class WineQualityRedClass {
    public static void main(String[] args) throws IOException, InterruptedException {
        int seed = 123;
        double learningRate = 0.005;
        int batchSize = 5;
        int nEpochs = 30;

        int numInputs = 11;
        int numOutputs = 1;
        int numHiddenNodes = 60;

        //load the training data
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new ClassPathResource("/classification/winequality-red.csv").getFile()));
        //labelIndex是指结果的位置，这个样本结果是第12个位置
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 11, 11, true);

        //测试数据
        RecordReader testRecordReader = new CSVRecordReader();
        testRecordReader.initialize(new FileSplit(new ClassPathResource("/classification/winequality-red-test.csv")
                .getFile()));
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, 1, 11, 11, true);

        //构建网络
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.updater(new Nesterovs(learningRate, 0.9))
                .updater(new Sgd(learningRate))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(numHiddenNodes)
                        .activation(Activation.TANH)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .activation(Activation.IDENTITY)
                        .build())
                .pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        //print the score with every 1 iteration
        //model.setListeners(new ScoreIterationListener(10));

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        for (int i = 0; i < nEpochs; i++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        System.out.println("Evaluate model....");

        RegressionEvaluation evaluation = new RegressionEvaluation(numOutputs);
        while (testIter.hasNext()) {
            DataSet dataSet = testIter.next();
            INDArray features = dataSet.getFeatureMatrix();
            INDArray lables = dataSet.getLabels();
            INDArray predicted = model.output(features, false);
            System.out.println(predicted);
            evaluation.eval(lables, predicted);
        }
        System.out.println(evaluation.stats());

    }
}
