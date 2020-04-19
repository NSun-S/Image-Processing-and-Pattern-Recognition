import Complex.Complex;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Created by tinkie101 on 2015/08/16.
 */
public class FourierTransform {

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

    class CalculateFourierPixel implements Callable
    {
        private final BufferedImage image;
        private final int u, v, N;

        CalculateFourierPixel(final BufferedImage image, final int u, final int v, final int N)
        {
            this.image = image;
            this.u = u;
            this.v = v;
            this.N = N;
        }

        @Override
        public newPixel call()
        {
            Complex sum = new Complex(0.0d, 0.0d);

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

    //            double scaleVal = (1.0d / N) *sum.abs();
            double scaleVal = Math.abs((180*sum.phase()/Math.PI));
            if(scaleVal > 255)
                scaleVal = 255;
            else if(scaleVal < 0)
                scaleVal = 0;

            return new newPixel(u, v, scaleVal);
        }
    }

    class CalculateReverseFourierPixel implements Callable
    {
        private final BufferedImage image;
        private final int x, y, N;

        CalculateReverseFourierPixel(final BufferedImage image, final int x, final int y, final int N)
        {
            this.image = image;
            this.x = x;
            this.y = y;
            this.N = N;
        }

        @Override
        public newPixel call()
        {
            Complex sum = new Complex(0.0d, 0.0d);

            for(int u = 0; u < image.getWidth(); u++){
                for(int v = 0; v < image.getHeight(); v++){
                    Color colour = new Color(image.getRGB(u, v));

                    int averageGray = (colour.getRed() + colour.getGreen() + colour.getBlue())/3;

                    double phase = ((2.0d * Math.PI)/((double)N)) * (double)(u*x + v*y);

                    Complex complex = new Complex(Math.cos(phase), Math.sin(phase));
                    complex = complex.times((double) averageGray);
                    sum = sum.plus(complex);
                }
            }

//            sum = sum.times(1.0d/N/N);
            double scaleVal = sum.abs()*(1.0d/N);

            if(scaleVal > 255)
                scaleVal = 255;
            else if(scaleVal < 0)
                scaleVal = 0;
            return new newPixel(x, y, scaleVal);
        }
    }

    class CalculateDCTPixel implements Callable{
        private final BufferedImage image;
        private final int u, v, N;

        CalculateDCTPixel(final BufferedImage image, final int u, final int v, final int N)
        {
            this.image = image;
            this.u = u;
            this.v = v;
            this.N = N;
        }

        @Override
        public newPixel call()
        {
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
    }

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

    public BufferedImage getReverseFourierImage(BufferedImage image) throws Exception{
        BufferedImage result = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());

        int N = image.getHeight();

        for(int x = 0; x < result.getWidth(); x++){

            ExecutorService threadPool = Executors.newFixedThreadPool(result.getWidth());
            Set<Future<newPixel>> set = new HashSet<>();

            for(int y = 0; y < result.getHeight(); y++){

                Callable<newPixel> callable =  new CalculateReverseFourierPixel(image, x, y, N);
                Future<newPixel> future = threadPool.submit(callable);
                set.add(future);
            }

            for (Future<newPixel> future : set) {
                newPixel tempPixel = future.get();

                int pixelValue = (int)tempPixel.value;

                result.setRGB(tempPixel.uPos, tempPixel.vPos, new Color(pixelValue, pixelValue, pixelValue).getRGB());
            }

            threadPool.shutdown();

            System.out.print("\r" + x + ":" + N);
        }

        return result;
    }

    public BufferedImage getDCTImage(BufferedImage image) throws Exception{
        BufferedImage result = new BufferedImage(image.getWidth(), image.getHeight(), image.getType());

        int N = image.getHeight();

        for(int u = 0; u < result.getWidth(); u++){

            ExecutorService threadPool = Executors.newFixedThreadPool(result.getWidth());
            Set<Future<newPixel>> set = new HashSet<>();

            for(int v = 0; v < result.getHeight(); v++){

                Callable<newPixel> callable =  new CalculateDCTPixel(image, u, v, N);
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

    public BufferedImage centralize(BufferedImage image) throws Exception{
        int Width = image.getWidth();
        int Height = image.getHeight();

        assert (Width%2==0 && Height%2==0);
        int centor = Width/2;
        BufferedImage result = new BufferedImage(Width,Height,image.getType());

        for(int x = 0; x < Width; x++){
            for(int y = 0; y < Height ; y++){
                if(x < centor && y<centor){
                    result.setRGB(x+centor,y+centor,image.getRGB(x,y));
                }else if(x < centor){
                    result.setRGB(x+centor,y-centor,image.getRGB(x,y));
                }else if(y < centor){
                    result.setRGB(x-centor,y+centor,image.getRGB(x,y));
                }else{
                    result.setRGB(x-centor,y-centor,image.getRGB(x,y));
                }
            }

        }
        return result;
    }
}
