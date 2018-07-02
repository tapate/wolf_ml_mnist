package com.tianyalei.nd4j.lesson1;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author wuweifeng wrote on 2018/6/26.
 */
public class Nd4jCreate {
    public static void main(String[] args) {
        //构造一个3行5列全0的ndarray
        System.out.println("构造一个3行5列全0的ndarray");
        INDArray zeros = Nd4j.zeros(3, 5);
        System.out.println(zeros);

        System.out.println("构造一个3行5列全1的ndarray");
        INDArray ones = Nd4j.ones(3, 5);
        System.out.println(ones);
        
        //构造一个3行5列，数组元素均为随机产生的ndarray
        System.out.println("构造一个3行5列，数组元素均为随机产生的ndarray");
        INDArray rands = Nd4j.rand(3, 5);
        System.out.println(rands);

        //构造一个3行5列，数组元素服从高斯分布（平均值为0，标准差为1）的ndarray
        System.out.println("构造一个3行5列，数组元素服从高斯分布（平均值为0，标准差为1）的ndarray");
        INDArray randns = Nd4j.randn(3, 5);
        System.out.println(randns);

        //给一个一维数据，根据shape创造ndarray
        System.out.println("给一个一维数据，根据shape创造ndarray");
        INDArray array1 = Nd4j.create(new float[]{2, 2, 2, 2}, new int[]{1, 4});
        System.out.println(array1);
        INDArray array2 = Nd4j.create(new float[]{2, 2, 2, 2}, new int[]{2, 2});
        System.out.println(array2);
    }
}
