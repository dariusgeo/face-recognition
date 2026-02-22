package tutorial.opencv.face.recognition;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.face.FaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FilenameFilter;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import static org.opencv.core.CvType.CV_32SC1;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;
import static org.opencv.imgcodecs.Imgcodecs.imread;

/**
 * I couldn't find any tutorial on how to perform face recognition using OpenCV and Java,
 * so I decided to share a viable solution here. The solution is very inefficient in its
 * current form as the training model is built at each run, however it shows what's needed
 * to make it work.
 *
 * The class below takes two arguments: The path to the directory containing the training
 * faces and the path to the image you want to classify. Not that all images has to be of
 * the same size and that the faces already has to be cropped out of their original images
 * (Take a look here http://fivedots.coe.psu.ac.th/~ad/jg/nui07/index.html if you haven't
 * done the face detection yet).
 *
 * For the simplicity of this post, the class also requires that the training images have
 * filename format: <label>-rest_of_filename.png. For example:
 *
 * 1-jon_doe_1.png
 * 1-jon_doe_2.png
 * 2-jane_doe_1.png
 * 2-jane_doe_2.png
 * ...and so on.
 *
 * Source: http://pcbje.com/2012/12/doing-face-recognition-with-javacv/
 *
 * @author Petter Christian Bjelland
 */

public class RealTimeFaceRecognition {

    static final FilenameFilter IMG_FILTER = (dir, name) -> {
        name = name.toLowerCase();
        return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
    };

    Map<Integer, Mat> images = new HashMap<>();
    Map<Integer, String> personNames = new HashMap<>();

    static {
        // magic command - it solves any .dll not found issue
        Loader.load(opencv_java.class);
    }

    public FaceRecognizer trainFaceRecognizer() {
        Path trainingDir = Paths.get("src/main/resources/images/training").toAbsolutePath();
        File root = new File(trainingDir.toString());

        FilenameFilter imgFilter = IMG_FILTER;

        File[] imageFiles = root.listFiles(imgFilter);
        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        int counter = 0;
        String name = "error";

        for (File image : imageFiles) {
            Mat img = imread(image.getAbsolutePath(), IMREAD_GRAYSCALE);

            int label = parseLabel(image.getName());
            name = parseName(image.getName());

            images.put(counter, img);
            Mat row = labels.row(counter);
            row.setTo(new Scalar(label));
            personNames.put(counter, name);

            counter++;
        }


        FaceRecognizer faceRecognizer = EigenFaceRecognizer.create();
//         FaceRecognizer faceRecognizer = FisherFaceRecognizer.create();
//         FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();

        faceRecognizer.train(new ArrayList<>(images.values()), labels);

        return faceRecognizer;
    }

    public String predict(FaceRecognizer faceRecognizer, Mat testImage) throws Exception {
//        Path testFilePath = Paths.get("src/main/resources/images/test/" + imageName).toAbsolutePath();
        Path resultsDir = Paths.get("src/main/resources/images/result").toAbsolutePath();

        int[] label = new int[1];
        double[] confidence = new double[1];
        faceRecognizer.predict(testImage, label, confidence);
        int predictedLabel = label[0];

        System.out.println("Predicted label: " + predictedLabel);
        // predictedLabel-1 because images are counted from 1
//        BufferedImage image = Mat2BufferedImage(images.get(predictedLabel));
//        ImageIO.write(image, "png", new File(resultsDir + "\\" + personNames.get(predictedLabel) + ".png"));

        return personNames.get(predictedLabel - 1);
    }

    public static void main(String[] args) throws Exception {


        Path resultsDir = Paths.get("src/main/resources/images/result").toAbsolutePath();
        Path trainingDir = Paths.get("src/main/resources/images/training").toAbsolutePath();
        Path testFilePath = Paths.get("src/main/resources/images/test").toAbsolutePath();
        Mat testImage = imread(testFilePath + "/1-andrew_1.png", IMREAD_GRAYSCALE);

        File root = new File(trainingDir.toString());

        File[] imageFiles = root.listFiles(IMG_FILTER);
        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        Map<Integer, Mat> images = new HashMap<>();
        Map<Integer, String> personNames = new HashMap<>();

        int counter = 0;
        String name = "error";

        for (File image : imageFiles) {
            Mat img = imread(image.getAbsolutePath(), IMREAD_GRAYSCALE);

            int label = parseLabel(image.getName());
            name = parseName(image.getName());

            images.put(counter, img);
            Mat row = labels.row(counter);
            row.setTo(new Scalar(label));
            personNames.put(counter, name);

            counter++;
        }


         FaceRecognizer faceRecognizer = org.opencv.face.EigenFaceRecognizer.create();
//         FaceRecognizer faceRecognizer = FisherFaceRecognizer.create();
//         FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();

        faceRecognizer.train(new ArrayList<>(images.values()), labels);

        int[] label = new int[1];
        double[] confidence = new double[1];
        faceRecognizer.predict(testImage, label, confidence);
        int predictedLabel = label[0];

        System.out.println("Predicted label: " + predictedLabel);
        // predictedLabel-1 because images are counted from 1
        BufferedImage image = Mat2BufferedImage(images.get(predictedLabel-1));
        ImageIO.write(image, "png", new File(resultsDir + "\\" + personNames.get(predictedLabel-1) + ".png"));
    }

    public static BufferedImage Mat2BufferedImage(Mat matrix)throws Exception {
        MatOfByte mob=new MatOfByte();
        Imgcodecs.imencode(".png", matrix, mob);
        byte ba[] = mob.toArray();

        return ImageIO.read(new ByteArrayInputStream(ba));
    }

    static int parseLabel(String filename) {
        return Integer.parseInt(filename.split("\\-")[0]);
    }

    static String parseName(String filename) {
        return filename.substring(filename.indexOf("-") + 1, filename.indexOf("_"));
    }
}