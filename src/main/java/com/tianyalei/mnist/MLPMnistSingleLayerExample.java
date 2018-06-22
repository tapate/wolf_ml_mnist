package com.tianyalei.mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * A Simple Multi Layered Perceptron (MLP) applied to digit classification for
 * the MNIST Dataset (http://yann.lecun.com/exdb/mnist/).
 * <p>
 * This file builds one input layer and one hidden layer.
 * <p>
 * The input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
 * (relu) activation function. The weights for this layer are initialized by using Xavier initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer will have 1000 output signals to the hidden layer.
 * <p>
 * The hidden layer has input dimensions of 1000. These are fed from the input layer. The weights
 * for this layer is also initialized using Xavier initialization. The activation function for this
 * layer is a softmax, which normalizes all the 10 outputs such that the normalized sums
 * add up to 1. The highest of these normalized values is picked as the predicted class.
 */
public class MLPMnistSingleLayerExample {

    private static Logger log = LoggerFactory.getLogger(MLPMnistSingleLayerExample.class);

    public static void main(String[] args) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        //潜在结果（比如0到9的整数标签）的数量。
        int outputNum = 10;
        //  每一步抓取的样例数量，每批次处理的数据越多，训练速度就越快。
        int batchSize = 128;
        //一个epoch指将给定数据集全部处理一遍的周期。epoch的数量越多，遍历数据集的次数越多，准确率越高
        //但是，epoch的数量达到一定的大小之后，增益会开始减少，所以要在准确率与训练速度之间进行权衡。
        int numEpochs = 15;
        //  这个随机数生成器用一个随机种子来确保训练时使用的初始权重维持一致。下文将会说明这一点的重要性。
        int rngSeed = 123;

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        log.info("Build model....");
        /* 参数说明：
         * seed:该参数将一组随机生成的权重确定为初始权重。如果一个示例运行很多次，而每次开始时都生成一组新的随机权重，
         * 那么神经网络的表现（准确率和F1值）有可能会出现很大的差异，因为不同的初始权重可能会将算法导向误差曲面上
         * 不同的局部极小值。在其他条件不变的情况下，保持相同的随机权重可以使调整其他超参数所产生的效果表现得更加清晰。
         *
         * update-Nesterovs:learningRate参数用于设定学习速率（learningrate），即每次迭代时对于权重的调整幅度，
         * 亦称步幅。学习速率越高，神经网络“翻越”整个误差曲面的速度就越快，
         * 但也更容易错过误差极小点。学习速率较低时，网络更有可能找到极小值，但速度会变得非常慢，因为每次权重调整的幅度都比较小。
         *
         * momentum：动量（momentum）是另一项决定优化算法向最优值收敛的速度的因素。动量影响权重调整的方向，
         * 所以在代码中，我们将其视为一种权重的更新器
         *
         * optimizationAlgo: 随机梯度下降（Stochastic Gradient Descent，SGD）是一种用于优化代价函数的常见方法。
         *
         * l2：正则化（regularization）是用来防止过拟合的一种方法。过拟合是指模型对训练数据的拟合非常好，
         * 然而一旦在实际应用中遇到从未出现过的数据，运行效果就变得很不理想。我们用L2正则化来防止个别权重对总体结果产生过大的影响。
         *
         * list：函数可指定网络中层的数量；它会将您的配置复制n次，建立分层的网络结构。
         *
         * layer：具体的层，index为0代表输入层。in代表输入节点数28*28，将28*28像素的矩阵转为只有一行28*28列。
         * out代表输出节点的数量
         * activation激活函数，最后一层给SOFTMAX。https://blog.csdn.net/kangyi411/article/details/78969642
         * weightInit：各系数的初始化，如y=kx+1，那么需要给k一个默认值，一般不会给0或者1，01是两个极端，我们给默认的XAVIER
         */
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .updater(new Nesterovs(0.006, 0.9))
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                //损失函数：MSE：均方差：线性回归；NEGATIVELOGLIKELIHOOD：负对数似然函数
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        log.info("Train model....");
        //共对整个数据集扫描15遍
        for (int i = 0; i < numEpochs; i++) {
            model.fit(mnistTrain);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while (mnistTest.hasNext()) {
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
