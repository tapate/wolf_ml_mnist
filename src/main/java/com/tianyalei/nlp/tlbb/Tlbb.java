package com.tianyalei.nlp.tlbb;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

/**
 * @author wuweifeng wrote on 2018/6/29.
 */
public class Tlbb {
    private static Logger log = LoggerFactory.getLogger(Tlbb.class);

    public static void main(String[] args) throws IOException {
        String filePath = new ClassPathResource("text/tlbb_t.txt").getFile().getAbsolutePath();
        log.info("Load & Vectorize Sentences....");

        SentenceIterator iter = new BasicLineIterator(new File(filePath));

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder().
                minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();
        log.info("Fitting Word2Vec model....");
        vec.fit();
        log.info("Writing word vectors to text file....");

        // Write word vectors to file
        log.info("Writing word vectors to text file....");

        WordVectorSerializer.writeWordVectors(vec, "tlbb_vectors.txt");
        WordVectorSerializer.writeFullModel(vec, "tlbb_model.txt");
        String[] names = {"萧峰", "乔峰", "段誉", "虚竹", "王语嫣", "阿紫", "阿朱", "木婉清"};
        log.info("Closest Words:");

        for (String name : names) {
            System.out.println(name + ">>>>>>");
            Collection<String> lst = vec.wordsNearest(name, 10);
            System.out.println(lst);
        }


    }

}
