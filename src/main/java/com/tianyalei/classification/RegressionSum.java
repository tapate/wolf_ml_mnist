package com.tianyalei.classification;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;

/**
 * 简单的加法学习，训练数据是生成的加法
 * 测试加法的训练
 *
 * @author wuweifeng wrote on 2018/6/22.
 */
public class RegressionSum {
    public static void main(String[] args) {
        DataSetIterator dataSetIterator = buildSet();
        int hiddenNodes = 10;
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(2).nOut(hiddenNodes).activation(Activation.TANH).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(hiddenNodes).nOut(1).activation
                        (Activation
                        .IDENTITY)
                        .build())
                .pretrain(false).backprop(true).build();
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(configuration);
        multiLayerNetwork.init();
        multiLayerNetwork.setListeners(new ScoreIterationListener(1));

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        multiLayerNetwork.setListeners(new StatsListener(statsStorage));

        int epochs = 200;
        for (int i = 0; i < epochs; i++) {
            multiLayerNetwork.fit(dataSetIterator);
        }
        //测试数据
        INDArray test = Nd4j.create(new double[]{0.11111, 0.222222222}, new int[]{1, 2});
        INDArray out = multiLayerNetwork.output(test, false);
        System.out.println(out);
    }

    private static ListDataSetIterator<DataSet> buildSet() {
        int number = 1000;
        double[] sum = new double[number];
        double[] input1 = new double[number];
        double[] input2 = new double[number];

        for (int i = 0; i < number; i++) {
            input1[i] = i;
            input2[i] = i;
            sum[i] = input1[i] + input2[i];
        }
        //变成一列矩阵
        // 1
        // 2
        // 3
        INDArray inputArray1 = Nd4j.create(input1, new int[]{number, 1});
        INDArray inputArray2 = Nd4j.create(input2, new int[]{number, 1});
        //变成二列矩阵
        // 1, 1
        // 2, 2
        // 3, 3
        INDArray inputArray = Nd4j.hstack(inputArray1, inputArray2);
        INDArray output = Nd4j.create(sum, new int[]{number, 1});
        //变成DataSet数据集，包含input和output
        DataSet dataSet = new DataSet(inputArray, output);
        //变成list形式，每行包含输入和输出
        List<DataSet> list = dataSet.asList();
        Collections.shuffle(list);
        return new ListDataSetIterator<>(list, 5);
    }
}
