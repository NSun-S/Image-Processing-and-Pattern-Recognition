import java.awt.image.*;
import java.io.File;
import java.io.IOException;
import javax.imageio.*;
import javax.swing.*;

public class MyImage extends JFrame {
    // 使用需更改图片文件的路径及文件名
    static final String filepath = "D:\\study\\图像处理与模式识别\\image\\src\\";
    static final String filename = "Naruto.jpg";
//   static final String format = ".jpg";
    static private int width, height, upper_bound, lower_bound;
    static int[] pixels;
    static int[] pixels_gray_linear_change;
    static int[] pixels_gray_stretch;
    static int[] histogram=new int[256];

    public static void main(String[] args) throws IOException,InterruptedException{
        grayImage();
        histogramEqualization();
        linearChange();
        linear_stretch();
    }

    public static void grayImage() throws IOException, InterruptedException {
        File file = new File(filepath+filename);
        BufferedImage myImage = ImageIO.read(file);

        width = myImage.getWidth();
        height = myImage.getHeight();

        pixels = new int[width*height];
        pixels_gray_linear_change = new int[width*height];
        pixels_gray_stretch = new int[width*height];

        BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        for(int i = 0; i<width;i++){
            for(int j = 0; j<height;j++){
                int RGB = myImage.getRGB(i, j);
                int gray = (int)(0.3*((RGB&0xff0000)>>16)+0.59*((RGB&0xff00)>>8)+0.11*(RGB&0xff));
                RGB = 255<<24|gray<<16|gray<<8|gray;
                grayImage.setRGB(i,j,RGB);
            }
        }
        PixelGrabber pg1 = new PixelGrabber(grayImage, 0,0,width,height,pixels,0,width);
        pg1.grabPixels();
        PixelGrabber pg2 = new PixelGrabber(grayImage, 0,0,width,height,pixels_gray_linear_change,0,width);
        pg2.grabPixels();
        PixelGrabber pg3 = new PixelGrabber(grayImage, 0,0,width,height,pixels_gray_stretch,0,width);
        pg3.grabPixels();
        File f = new File(filepath+"gray_"+filename);
        ImageIO.write(grayImage,"jpg",f);
    }

    public static void histogramEqualization() throws IOException{
        BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for(int i = 0; i < width; i++){
            for(int j = 0; j < height; j++){
                int gray = pixels[i*height + j]&0xff;
                histogram[gray]++;
            }
        }
        double[] p = new double[256];
        for(int i = 0; i < 256;i++){
            p[i] = (double)(histogram[i])/(width*height);
        }
        int[] s = new int[256];
        s[0] = (int)(p[0]*255);
        for(int i = 1; i<256;i++){
            p[i] += p[i-1];
            s[i] = (int)(p[i]*255);
        }
        for(int i = 0; i< height;i++){
            for(int j = 0;j<width;j++){
                int gray = pixels[i*width + j]&0xff;
                int hist = s[gray];
                pixels[i*width+j] = 255<<24|hist<<16|hist<<8|hist;
                grayImage.setRGB(j,i,pixels[i*width+j]);
            }
        }
        File f = new File(filepath+"histogram_"+filename);
        ImageIO.write(grayImage,"jpg",f);
    }

    public static void linearChange() throws IOException{
        upper_bound = 128;
        lower_bound = 0;
        int now_max = 0, now_min = 255;
        for(int i = 0; i<height; i++){
            for(int j = 0;j<width;j++){
                int gray = pixels_gray_linear_change[i*width+j]&0xff;
                if(gray > now_max){
                    now_max = gray;
                }
                if(gray < now_min){
                    now_min = gray;
                }
            }
        }
        System.out.println(now_min + " "+ now_max);
        BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        double changeRate = (double) (upper_bound - lower_bound)/(now_max - now_min);
        for(int i = 0; i < height; i++){
            for(int j=0;j<width;j++){
                int gray = pixels_gray_linear_change[i*width + j]&0xff;
                int newgray = (int)((gray-now_min)*changeRate + lower_bound);
                pixels_gray_linear_change[i*width + j] = 255<<24|newgray<<16|newgray<<8|newgray;
                grayImage.setRGB(j, i, pixels_gray_linear_change[i*width + j]);
            }
        }
        File f = new File(filepath + "linearChange512_"+filename);
        ImageIO.write(grayImage,"jpg",f);
    }

    public static void linear_stretch() throws IOException{
        int now_min=0,now_max=255,sum;
        double rate = 0.1;
        sum = 0;
        for(int i = 0;i<256;i++){
            sum += histogram[i];
            if(sum > width*height*rate){
                now_min = i;
                break;
            }
        }
        sum = 0;
        for(int i = 255;i>=0;i--){
            sum += histogram[i];
            if(sum >width*height*rate){
                now_max = i;
                break;
            }
        }
        System.out.println(now_min + " "+ now_max);
        upper_bound = 210;
        lower_bound = 10;
        BufferedImage grayImage = new BufferedImage(width,height,BufferedImage.TYPE_BYTE_GRAY);
        for(int i = 0; i< height;i++){
            for(int j = 0; j< width;j++){
                int gray = pixels_gray_stretch[i*width+j]&0xff;
                if(gray <= now_min){
                    gray = now_min;
                } else if(gray >= now_max){
                    gray = now_max;
                } else {
                    gray = (int)((double)(upper_bound-lower_bound)/(now_max-now_min) *(gray - now_min) + lower_bound);
                    if(gray > now_max){
                        gray = upper_bound;
                    }
                    if(gray < now_min){
                        gray = lower_bound;
                    }
                }
                pixels_gray_stretch[i*width + j] = 255<<24|gray<<16|gray<<8|gray;
                grayImage.setRGB(j, i, pixels_gray_stretch[i*width+j]);
            }
        }
        File f = new File(filepath + "stretch_"+filename);
        ImageIO.write(grayImage,"jpg",f);
    }
}
