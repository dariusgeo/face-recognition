package tutorial.opencv.face.recognition;

import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.face.FaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;

import java.awt.image.BufferedImage;
import java.io.File;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for RealTimeFaceRecognition.
 *
 * OpenCV native libraries are loaded automatically via the static initializer
 * in RealTimeFaceRecognition, so every test in this class has OpenCV available.
 */
class RealTimeFaceRecognitionTest {

    // -----------------------------------------------------------------------
    // parseLabel
    // -----------------------------------------------------------------------

    @Test
    void parseLabel_singleDigitLabel_returnsInt() {
        assertEquals(1, RealTimeFaceRecognition.parseLabel("1-andrew_1.png"));
    }

    @Test
    void parseLabel_highDigitLabel_returnsInt() {
        assertEquals(8, RealTimeFaceRecognition.parseLabel("8-gabi_1.png"));
    }

    @Test
    void parseLabel_multiDigitLabel_returnsInt() {
        assertEquals(12, RealTimeFaceRecognition.parseLabel("12-someface_0.png"));
    }

    // -----------------------------------------------------------------------
    // parseName
    // -----------------------------------------------------------------------

    @Test
    void parseName_shortName_returnsName() {
        assertEquals("andrew", RealTimeFaceRecognition.parseName("1-andrew_1.png"));
    }

    @Test
    void parseName_differentShortName_returnsName() {
        assertEquals("aree", RealTimeFaceRecognition.parseName("2-aree_3.png"));
    }

    @Test
    void parseName_longName_returnsName() {
        assertEquals("peeranut", RealTimeFaceRecognition.parseName("3-peeranut_0.png"));
        assertEquals("watcharin", RealTimeFaceRecognition.parseName("6-watcharin_1.png"));
    }

    // -----------------------------------------------------------------------
    // Mat2BufferedImage  (requires OpenCV)
    // -----------------------------------------------------------------------

    @Test
    void mat2BufferedImage_grayscaleMat_returnsNonNull() throws Exception {
        // 1-channel grayscale Mat: height=150, width=125
        Mat mat = new Mat(150, 125, CvType.CV_8UC1);
        BufferedImage result = RealTimeFaceRecognition.Mat2BufferedImage(mat);
        assertNotNull(result);
    }

    @Test
    void mat2BufferedImage_preservesDimensions() throws Exception {
        int width = 125, height = 150;
        Mat mat = new Mat(height, width, CvType.CV_8UC1);
        BufferedImage result = RealTimeFaceRecognition.Mat2BufferedImage(mat);
        assertEquals(width, result.getWidth());
        assertEquals(height, result.getHeight());
    }

    @Test
    void mat2BufferedImage_colorMat_returnsNonNull() throws Exception {
        Mat mat = new Mat(100, 100, CvType.CV_8UC3);
        BufferedImage result = RealTimeFaceRecognition.Mat2BufferedImage(mat);
        assertNotNull(result);
    }

    // -----------------------------------------------------------------------
    // trainFaceRecognizer  (requires OpenCV + training images on disk)
    // -----------------------------------------------------------------------

    @Test
    void trainFaceRecognizer_withExistingTrainingImages_returnsNonNullRecognizer() {
        RealTimeFaceRecognition recognizer = new RealTimeFaceRecognition();
        FaceRecognizer result = recognizer.trainFaceRecognizer();
        assertNotNull(result);
    }

    @Test
    void trainFaceRecognizer_populatesPersonNamesMap() {
        RealTimeFaceRecognition recognizer = new RealTimeFaceRecognition();
        recognizer.trainFaceRecognizer();
        assertFalse(recognizer.personNames.isEmpty());
    }

    @Test
    void trainFaceRecognizer_populatesImagesMap() {
        RealTimeFaceRecognition recognizer = new RealTimeFaceRecognition();
        recognizer.trainFaceRecognizer();
        assertFalse(recognizer.images.isEmpty());
    }

    // -----------------------------------------------------------------------
    // predict  (requires OpenCV + training images + a test Mat)
    // -----------------------------------------------------------------------

    @Test
    void predict_withTrainingImage_returnsNonNullName() throws Exception {
        RealTimeFaceRecognition recognizer = new RealTimeFaceRecognition();
        FaceRecognizer faceRecognizer = recognizer.trainFaceRecognizer();

        // Use one of the training images as input — EigenFace should map it
        File trainingImg = new File("src/main/resources/images/training/1-andrew_1.png");
        Mat testImage = Imgcodecs.imread(trainingImg.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
        assertFalse(testImage.empty(), "Training image could not be loaded");

        String name = recognizer.predict(faceRecognizer, testImage);
        assertNotNull(name);
    }
}