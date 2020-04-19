### 实验内容
用程序实现一个数字图像的傅里叶变换和余弦变换。
### 实验设计
在最开始的设计中，我仍沿用上次的方法使用Java语言，尝试根据课上的公式实现数字图像的离散傅里叶变换（DFT）及反变换（IDFT），以及离散余弦变换（DCT）。在推进的过程中，由于自己实现的IDFT有问题，又用python的opencv库进行了尝试。
在图像的选取上，选择了经典的Lena图，还有一张类似的图片。
### 实验流程
1.编写图片IO相关接口
工欲善其事，必先利其器，在实验的第一步，我选择了完成图片相关的接口，包括读取图片接口，绘制图片接口和保存图片接口。图片的读取和保存使用了Image IO类，图片的绘制使用了JFrame类。
读取图片接口：
```java
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
```
保存图片接口：
```java
public static void WriteImages(String Path, BufferedImage image) throws Exception{
    File f = new File(Path);
    try{
        ImageIO.write(image,"png",f);
    } catch (IOException e) {
        throw new Exception(e.getMessage());
    }
}
```
绘制图片接口：
```java
public static void drawImages(String title, int height, int width, BufferedImage... ImageList)
{
    JFrame frame = new JFrame();
    frame.setTitle(title);//设置标题
    frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
	//默认关闭当前窗口
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
```
2.定义复数类

在傅里叶变换中，因此在实际计算中，我们需要复数类，分别记录实部和虚部。这部分我定义了一个复数类，实现了复数的表示以及复数间的各种计算。下面展示复数类的构造方法以及部分计算方法。
构造方法：
```java
// create a new object with the given real and imaginary parts
public Complex(double real, double imag) {
    re = real;
    im = imag;
}
```
构造方法很简单，记录实部和虚部就可以了。
复数加法：
```java
// return a new Complex object whose value is (this + b)
public Complex plus(Complex b) {
    Complex a = this;             // invoking object
    double real = a.re + b.re;
    double imag = a.im + b.im;
    return new Complex(real, imag);
}
```
复数乘法：
```java
// return a new Complex object whose value is (this * b)
public Complex times(Complex b) {
    Complex a = this;
    double real = a.re * b.re - a.im * b.im;
    double imag = a.re * b.im + a.im * b.re;
    return new Complex(real, imag);
}

// return a new object whose value is (this * alpha)
public Complex times(double alpha) {
    return new Complex(alpha * re, alpha * im);
}
```
幅值计算：
```java
public double abs()   { return Math.hypot(re, im); }  
// Math.sqrt(re*re + im*im)
```
相位计算：
```java
public double phase() { return Math.atan2(im, re); }  
// between -pi and pi
```
复数的加法和乘法按照公式实现即可，在我们的傅里叶变换中会用到幅值计算和相位计算的函数，也进行了相应的实现。在我们后续的计算中，会用到复数的相关功能。
3.DFT幅值谱计算
在这部分，我直接按照傅里叶正变换的公式进行计算，没有注意到傅里叶变换的可分离性，导致整个程序为n4的复杂度，在图片长宽都为256像素的情况下，计算量太大，然而在程序完成之前我仍未注意到傅里叶变换的可分离性，因此我采用的方法是多线程。
以256*256的图片为例说明，我在计算每一行的256个F(u,v)值时，将每一个值的计算
作为一个子线程进行计算，这样可以以并行的方式缩短运算的时间，但由于实际的计算量并未减少，计算一次仍需要近10分钟，但勉强可以支撑任务。
我定义了一个Pixel类用来记录一个像素在原图中的坐标，以及后续计算出来的对应的傅里叶变换值。
```java
class newPixel
{
    int uPos, vPos;
    double value;
    newPixel(int uPos, int vPos, double value){
        this.uPos = uPos;
        this.vPos = vPos;
        this.value = value;
    }
}
```
在实际计算时，我将这个计算过程分为了两层循环，内层循环是对具体某一个位置傅里叶变换值的计算，外层循环是遍历每一个位置通过内层循环计算其傅里叶变换值。下面来具体看两个部分的内容。
内层循环：
```java
for(int x = 0; x < image.getWidth(); x++){
     for(int y = 0; y < image.getHeight(); y++){
         Color colour = new Color(image.getRGB(x, y));
         int averageGray = (colour.getRed() + colour.getGreen() + colour.getBlue())/3;
         double phase = ((-2.0d * Math.PI)/((double)N)) * (double)(u*x + v*y);
         Complex complex = new Complex(Math.cos(phase), Math.sin(phase));
         complex = complex.times((double)averageGray);
         sum = sum.plus(complex);
    }
}
double scaleVal = (1.0d / N) *sum.abs();
```
外层循环：
```java
public BufferedImage generateFourierImage(BufferedImage image) throws Exception{
    BufferedImage result = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());
    int N = image.getHeight();
    for(int u = 0; u < result.getWidth(); u++){
        ExecutorService threadPool = Executors.newFixedThreadPool(result.getWidth());
        Set<Future<newPixel>> set = new HashSet<>();
        for(int v = 0; v < result.getHeight(); v++){
            Callable<newPixel> callable =  new CalculateFourierPixel(image, u, v, N);
            Future<newPixel> future = threadPool.submit(callable);
            set.add(future);
        }
        for (Future<newPixel> future : set) {
            newPixel tempPixel = future.get();
            int pixelValue = (int)tempPixel.value;
            result.setRGB(tempPixel.uPos, tempPixel.vPos, new Color(pixelValue, pixelValue, pixelValue).getRGB());
        }
        threadPool.shutdown();
        System.out.print("\r" + u + ":" + N);
    }
    return result;
}
```
这里可以清楚的看出，我们多线程相关的内容，每一行都新建一个线程池，然后将这一行每个位置傅里叶变换值的计算开创一个线程，大大节省了计算的时间。
下面是我们对Lena图的处理，左边是lena图的原图，为256*256大小，右边是计算得到的傅里叶幅值频谱图，下面是对幅值频谱进行中心化之后的频谱图（中心化相关函数交简单，这里不予展示）。

![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/lena/lena.png)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/lena/DFT_lena.png)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/lena/DFT_shift_lena.png)

可以看到在中心化之后的图片，中间的为低频直流分量，四周的为高频分量。
4.DFT相位谱计算
这部分和上部分函数的框架完全一致，只需要将计算幅值的公式替换为计算相位值的公式。

即只需要在原来的程序中做一行改动。
double scaleVal = Math.abs((180*sum.phase()/Math.PI));
再次运行程序就能够得到图像的相位谱。如下图所示（左图为中心化前的图像，右图为中心化后的图像）：

![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/lena/phase_lena.png)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/lena/phase2_lena.png)

5.DCT相位谱计算
这部分仍沿用上一部分的多线程结构，但是具体的计算模块需要进行替换，计算方式替换成DCT相应的计算公式（如下图所示）。

其中的C(u)和C(v)由以下公式决定。

根据上面的公式，我进行了相应的代码实现。
```java
public newPixel call(){
    double sum = 0;
    double cu,cv;
    if(u == 0) cu = 1/Math.pow(2,0.5);
    else cu = 1;
    if(v == 0) cv = 1/Math.pow(2,0.5);
    else cv = 1;
    for(int x = 0; x < image.getWidth(); x++){
    for(int y = 0; y < image.getHeight(); y++){
        Color colour = new Color(image.getRGB(x, y));
        int averageGray = (colour.getRed() + colour.getGreen() + colour.getBlue())/3;
         sum += averageGray*cu*cv*Math.cos((2*x+1.0d)*u*Math.PI/(2*N))
                    *Math.cos((2*y+1.0d)*v*Math.PI/(2*N));
}
        }
         double scaleVal = sum*(2.0d/N);
         if(scaleVal > 255)
             scaleVal = 255;
         else if(scaleVal < 0)
             scaleVal = 0;
         return new newPixel(u, v, scaleVal);
}
```
运行程序得到下图的结果。
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/lena/DCT_lena.png)

可以看出能量集中在了左上角。
6.IDFT
	在进行了上面的任务之后，我尝试了对频谱图进行反变换，这里程序和上面相仿，不再进行展示，这一过程中，不知道是程序的问题还是对公式的理解有误区，导致我反变换之后的结果得不到原图像。
进一步，我尝试了使用python中的opencv库来完成这一过程，由于opencv中封装了相应的函数，实现起来非常地简单。
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('images/lena.png',0)
dft = cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
plt.subplot(221)
plt.axis('off')
plt.title('dft')
plt.imshow(20*np.log(cv2.magnitude(dft[:,:,0],dft[:,:,1])),cmap='gray')

dft_shift = np.fft.fftshift(dft)
plt.subplot(222)
plt.axis('off')
plt.title('dft_shift')
plt.imshow(20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])),cmap='gray')
idft_shift = np.fft.ifftshift(dft_shift)
plt.subplot(223)
plt.axis('off')
plt.title('origin')
plt.imshow(img,cmap='gray')
idft = cv2.idft(idft_shift)
plt.subplot(224)
plt.axis('off')
plt.title('idft_shift')
plt.imshow(cv2.magnitude(idft[:,:,0],idft[:,:,1]),cmap='gray')
plt.show()
```
程序运行结果如下：
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/cv_DFT.png)

四、实验感想
经过了这次实验，我对傅里叶变换和离散余弦变换有了进一步的了解，初步掌握了对应的编程方法，还意外更深入学习了java多线程相关知识。不足之处在于没有自己手动实现傅里叶反变换这一过程，而是直接用了轮子，导致对这一部分的知识没有投入到具体应用。


五、附录
这里附上另一张图片的处理结果。

![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/film/film.png)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/film/film.png)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/film/DFT_shift_film.png)
![](https://github.com/NSun-S/Image-Processing-and-Pattern-Recognition/blob/master/imageTransformation/images/film/DCT_film.png)
