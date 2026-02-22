package tutorial.opencv.face.detection;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.*;
import org.opencv.face.FaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import tutorial.opencv.face.recognition.RealTimeFaceRecognition;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.*;
import java.net.URL;
import java.nio.file.Paths;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_BGR2GRAY;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;

public class RealTimeFaceDetection {

    static CascadeClassifier loadCascadeClassifier() throws Exception {
        URL url = RealTimeFaceDetection.class.getClassLoader().getResource("haarcascade_frontalface_default.xml");
        File file = Paths.get(url.toURI()).toFile();
        return new CascadeClassifier(file.getAbsolutePath());
    }

    static BufferedImage matToBufferedImage(Mat matrix) {
        BufferedImage image = new BufferedImage(matrix.width(), matrix.height(), BufferedImage.TYPE_3BYTE_BGR);
        WritableRaster raster = image.getRaster();
        DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
        byte[] data = dataBuffer.getData();
        matrix.get(0, 0, data);
        return image;
    }

    static byte[] encodeMatToJpeg(Mat mat) {
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, matOfByte);
        return matOfByte.toArray();
    }

    private static byte[] captureFrame(VideoCapture videoInput,
                                       RealTimeFaceRecognition eigenFaceRecognizer,
                                       FaceRecognizer faceRecognizer) throws Exception {

        // Reading the next video frame from the camera
        Mat matrix = new Mat();
        videoInput.read(matrix);

        // If there is next video frame
        if (videoInput.read(matrix)) {
            CascadeClassifier classifier = loadCascadeClassifier();

            MatOfRect faceDetections = new MatOfRect();
            /// Detecting faces in video frame ///
            classifier.detectMultiScale(matrix, faceDetections);
            System.out.printf("Detected %s faces %n", faceDetections.toArray().length);

            // Surrounding 'human face' with a rectangle
            for (Rect rect : faceDetections.toArray()) {
                Mat face = matrix.submat(rect);
                Mat greyMat = new Mat();
                Imgproc.cvtColor(face, greyMat, CV_BGR2GRAY);
                System.out.println("greyMat Width " + greyMat.width());
                System.out.println("greyMat Height " + greyMat.height());
;
                Mat resizeimage = new Mat();
                Size sz = new Size(125,150);
                Imgproc.resize(greyMat, resizeimage, sz);
                /* Uncomment these lines if you want to generate new training images. */
//                BufferedImage image = Mat2BufferedImage(resizeimage);
//                Path resultsDir = Paths.get("src/main/resources/images/result").toAbsolutePath();
//                ImageIO.write(image, "png", new File(resultsDir + "\\" +new Date().getTime() + ".png"));

                String personName = eigenFaceRecognizer.predict(faceRecognizer, resizeimage);
                System.out.println(">>> Person: " + personName);

                //Preparing the arguments
                Scalar color = new Scalar(0, 0, 255);
                int font = Imgproc.FONT_HERSHEY_SIMPLEX;
                int scale = 1;
                int thickness = 2;
                //Adding text to the image
                Imgproc.putText(matrix, personName, new Point(rect.x, rect.y), font, scale, color, thickness);
                Imgproc.rectangle(
                        matrix,                                                     //where to draw the box
                        new Point(rect.x, rect.y),                                  //bottom left
                        new Point(rect.x + rect.width, rect.y + rect.height), //top right
                        new Scalar(0, 0, 255)                                       //RGB colour
                );
            }

            // Creating BufferedImage from the matrix
            matToBufferedImage(matrix);
        }

        return encodeMatToJpeg(matrix);
    }

    public static void main(String[] args) throws Exception {

        // magic command - it solves any .dll not found issue
        Loader.load(opencv_java.class);

        // Instantiating the VideoCapture class (camera:: 0)
        VideoCapture defaultCamera = new VideoCapture(0);
        if (!defaultCamera.isOpened()) {
            System.out.println("camera not detected");
        } else {
            System.out.println("Camera detected ");
        }

        JFrame frame = new JFrame();
        frame.setSize(750, 600);
        RealTimeFaceRecognition openCVFaceRecognizer = new RealTimeFaceRecognition();
        FaceRecognizer faceRecognizer = openCVFaceRecognizer.trainFaceRecognizer();

        while(true) {
            byte[] image = captureFrame(defaultCamera, openCVFaceRecognizer, faceRecognizer);

            InputStream in = new ByteArrayInputStream(image);
            BufferedImage bufImage = ImageIO.read(in);

            frame.getContentPane().removeAll();
            //Set Content to the JFrame
            frame.getContentPane().add(new JLabel(new ImageIcon(bufImage)));
            frame.addWindowListener(
                    new WindowAdapter() {
                        public void windowClosing(WindowEvent e) {
                            System.exit(0);
                        }
                    });


            frame.pack();
            frame.repaint();
            frame.setVisible(true);
        }
    }

    public static Mat bufferedImage2Mat(BufferedImage image) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ImageIO.write(image, "png", byteArrayOutputStream);
        byteArrayOutputStream.flush();
        return Imgcodecs.imdecode(new MatOfByte(byteArrayOutputStream.toByteArray()), IMREAD_GRAYSCALE);
    }
}
