package com.tianyalei.nlp.tlbb;

import java.io.*;

/**
 * 运行后将得到一个分词后的文档
 * @author wuweifeng wrote on 2018/6/29.
 */
public class FenCi {
    private FudanTokenizer tokenizer = new FudanTokenizer();

    public void processFile() throws Exception {
        String filePath = this.getClass().getClassLoader().getResource("text/tlbb.txt").getPath();
        BufferedReader in = new BufferedReader(new FileReader(filePath));

        File outfile = new File("/Users/wuwf/project/tlbb_t.txt");
        if (outfile.exists()) {
            outfile.delete();
        }
        FileOutputStream fop = new FileOutputStream(outfile);

        // 构建FileOutputStream对象,文件不存在会自动新建
        String line = in.readLine();
        OutputStreamWriter writer = new OutputStreamWriter(fop, "UTF-8");
        while (line != null) {
            line = tokenizer.processSentence(line);
            writer.append(line);
            line = in.readLine();
        }
        in.close();
        writer.close(); // 关闭写入流,同时会把缓冲区内容写入文件
        fop.close(); // 关闭输出流,释放系统资源
    }

    public static void main(String[] args) throws Exception {
         new FenCi().processFile();
    }
}
