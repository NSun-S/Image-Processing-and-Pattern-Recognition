import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;

public class Main {
    public static BufferedImage readImage(String file) throws Exception{
        BufferedImage img;
        try {
            img = ImageIO.read(new File(file));
        } catch (IOException e) {
            throw new Exception(e.getMessage());
        }
        assert (img.getWidth() == img.getHeight());
        return img;
    }

    public static void drawImages(String title, int height, int width, BufferedImage... ImageList)
    {
        JFrame frame = new JFrame();
        frame.setTitle(title);//设置标题
        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);//默认关闭当前窗口
        frame.getContentPane().setLayout(new FlowLayout());
        for(int i = 0; i < ImageList.length; i++) {//将图片进行放置
            JLabel image = new JLabel();
            image.setSize(width, height);
            java.awt.Image org = ImageList[i].getScaledInstance(width, height, java.awt.Image.SCALE_SMOOTH);
            image.setIcon(new ImageIcon(org));
            frame.getContentPane().add(image);
        }
        frame.pack();
        frame.setVisible(true);
    }

    public static void WriteImages(String Path, BufferedImage image) throws Exception{
        File f = new File(Path);
        try{
            ImageIO.write(image,"png",f);
        } catch (IOException e) {
            throw new Exception(e.getMessage());
        }
    }

    public static void main(String args[]) throws Exception{
        BufferedImage lena = readImage("D:\\study\\图像处理与模式识别\\imageTransformation\\src\\images\\result_film.png");

        //drawImages("grayImages", lena.getWidth(), lena.getHeight(), lena);

        FourierTransform dft = new FourierTransform();
        BufferedImage resultLena = dft.generateFourierImage(lena);
        BufferedImage resultcentralizeLena = dft.centralize(resultLena);
//        drawImages("Transform Images1", resultLena.getWidth(), resultLena.getHeight(), resultLena);

        BufferedImage result2Lena = dft.getReverseFourierImage(resultLena);
        BufferedImage result3Lena = dft.getDCTImage(lena);
//        drawImages("Transform Images2", lena.getWidth(), lena.getHeight(), result2Lena);
    }

}
