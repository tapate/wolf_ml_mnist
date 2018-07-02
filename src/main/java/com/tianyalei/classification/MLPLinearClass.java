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
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

/**
 * @author wuweifeng wrote on 2018/6/20.
 */
public class MLPLinearClass {
    public static void main(String[] args) throws IOException, InterruptedException {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        String trainPath = new ClassPathResource("/classification/linear_data_train.csv").getFile().getPath();
        String testPath = new ClassPathResource("/classification/linear_data_eval.csv").getFile().getPath();

        //load the training data
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new File(trainPath)));
        //labelIndex是指结果的位置，这个样本结果是第0个位置
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 0, 2);

        //测试数据
        RecordReader testRecordReader = new CSVRecordReader();
        testRecordReader.initialize(new FileSplit(new File(testPath)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 0, 2);

        //构建网络
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(10));

        for (int i = 0; i < nEpochs; i++) {
            model.fit(trainIter);
        }
        System.out.println("Evaluate model....");
        Evaluation evaluation = new Evaluation(numOutputs);
        while (testIter.hasNext()) {
            DataSet dataSet = testIter.next();
            INDArray features = dataSet.getFeatureMatrix();
            INDArray lables = dataSet.getLabels();
            INDArray predicted = model.output(features, false);

            evaluation.eval(lables, predicted);
        }

        System.out.println(evaluation.stats());
    }
}
