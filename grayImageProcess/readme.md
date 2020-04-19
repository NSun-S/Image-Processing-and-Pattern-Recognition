# 图像处理与模式识别实验报告
## 实验内容
打开一幅图像，进行直方图均衡．将灰度线性变化，将灰度拉伸。
## 实验设计
在编程语言的选择上，我选择了Java，Java自带的图像处理包方便使用，对图像的存取较为方便。
图像的选择上，我选取了两张图片作为实验的对象，一张是我使用的QQ头像，另一张是我喜爱的动漫《火影忍者》的海报，其中一张的处理效果放在流程中详细讲解，另一张放在附录中。
## 实验流程
### 读取图像，将图像进行灰度化处理。
所选取的两张图像都是jpg格式的RGB图像，我们知道灰度值计算由公式Y=0.3R + 0.59G + 0.11B，因此我们将原图的三原色信息提取出来，用相应的公式进行计算，关键部分代码如下。
```
File file = new File(filepath+filename);
BufferedImage myImage = ImageIO.read(file);
width = myImage.getWidth();
height = myImage.getHeight();
for(int i = 0; i<width;i++){
    for(int j = 0; j<height;j++){
        int RGB = myImage.getRGB(i, j);
        int gray = (int)(0.3*((RGB&0xff0000)>>16)+0.59*((RGB&0xff00)>>8)+0.11*(RGB&0xff));
        RGB = 255<<24|gray<<16|gray<<8|gray;
        grayImage.setRGB(i,j,RGB);
    }
}
```
我们遍历图片的每一个像素点，计算出灰度值。原图和灰度图如下。

![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/grayImageProcess/images/image1/Naruto.jpg)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/grayImageProcess/images/image1/gray_Naruto.jpg)
### 将灰度图像进行直方图均衡
我们的图像采用离散化的直方图修正方法，先统计每一个灰度级别对应的像素点的个数，计算每个灰度级别对应的频率，然后从第0级到第255级逐渐累加频率，并计算出对应灰度级新的灰度值。下图为课件中对应的计算方法。

同样地，根据计算出的新的灰度值替代原来的灰度值，得到直方图均衡后的灰度图像。关键部分代码如下。
```
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
```
下图为灰度图（左）与直方图均衡处理后的图像（右）对比。可以看出，我们达到了增强图像，使图像包含信息最多的目的。

![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/grayImageProcess/images/image1/gray_Naruto.jpg)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/grayImageProcess/images/image1/histogram_Naruto.jpg)
### 将灰度图像进行线性变换
线性变换，顾名思义是是将原来的灰度范围线性映射到一个新的灰度范围。


仿照课件上的例子，我们将灰度范围从最开始的范围[0,226]分别变化到[0,127]和[0,63]，关键部分代码如下。
```
double changeRate = (double) (upper_bound - lower_bound)/(now_max - now_min);
for(int i = 0; i < height; i++){
    for(int j=0;j<width;j++){
        int gray = pixels_gray_linear_change[i*width + j]&0xff;
        int newgray = (int)((gray-now_min)*changeRate + lower_bound);
        pixels_gray_linear_change[i*width + j] = 255<<24|newgray<<16|newgray<<8|newgray;
        grayImage.setRGB(j, i, pixels_gray_linear_change[i*width + j]);
    }
}
```
统计图片灰度值的上下限部分代码略去，根据公式映射到新的灰度范围，然后替代原来的像素值。下面3张图是我们的效果图，左上是原灰度图，右上是映射到[0,127]区间的图像，下面是映射到[0,63]区间的图像。

![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/grayImageProcess/images/image1/gray_Naruto.jpg)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/grayImageProcess/images/image1/linearChange_Naruto.jpg)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/grayImageProcess/images/image1/linearChange64_Naruto.jpg)


### 将灰度图像进行线性拉伸
线性拉伸个人理解是线性变化的更为一般的形式，即分段线性变换，我们根据直方图修正中统计出的各灰度值对应的频率选取出大部分像素所在的灰度区间，然后根据上下限选择一个更宽的区间，以90%作为大多数的标准，得到的上下限为28和186，我们将其拉伸到[10,210]。关键部分代码如下。
```
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
```
下图为原灰度图（左）和拉伸后的图像（右）的对比图。

![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/grayImageProcess/images/image1/gray_Naruto.jpg)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/grayImageProcess/images/image1/stretch_Naruto.jpg)
## 实验感想
我使用Java语言实现了打开一幅图像，进行直方图均衡．将灰度线性变化，将灰度拉伸。刚开始我对图像处理一无所知，在读取图像过程中遇到了较多的问题，但当图片读取完进行进行接下来的操作时，我对课堂上讲授的关于灰度图像处理的内容有了更加深刻的理解。实验的效果也非常的符合预期，让我学习到了处理灰度图像的基本方法。
完整代码及另一张图片的处理结果在GitHub查看，https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/tree/master/grayImageProcess。
