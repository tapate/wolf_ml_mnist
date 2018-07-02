package com.tianyalei.nlp.tlbb;

import org.fnlp.ml.types.Dictionary;
import org.fnlp.nlp.cn.tag.CWSTagger;
import org.fnlp.nlp.corpus.StopWords;
import org.fnlp.util.exception.LoadModelException;

import java.io.IOException;
import java.util.List;

/**
 * @author wuweifeng wrote on 2018/6/29.
 */
public class FudanTokenizer {
    private CWSTagger tag;

    private StopWords stopWords;

    public FudanTokenizer() {
        String path = this.getClass().getClassLoader().getResource("").getPath();
        System.out.println(path);
        try {
            tag = new CWSTagger(path + "models/seg.m");
        } catch (LoadModelException e) {
            e.printStackTrace();
        }

    }

    public String processSentence(String context) {
        return tag.tag(context);
    }

    public String processSentence(String sentence, boolean english) {
        if (english) {
            tag.setEnFilter(true);
        }
        return tag.tag(sentence);
    }

    public String processFile(String filename) {
        return tag.tagFile(filename);
    }

    /**
     * 设置分词词典
     */
    public boolean setDictionary() {
        String dictPath = this.getClass().getClassLoader().getResource("models/dict.txt").getPath();

        Dictionary dict;
        try {
            dict = new Dictionary(dictPath);
        } catch (IOException e) {
            return false;
        }
        tag.setDictionary(dict);
        return true;
    }

    /**
     * 去除停用词
     */
    public List<String> flitStopWords(String[] words) {
        try {
            return stopWords.phraseDel(words);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

}
