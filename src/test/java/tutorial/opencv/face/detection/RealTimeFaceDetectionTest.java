package tutorial.opencv.face.detection;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.objdetect.CascadeClassifier;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for RealTimeFaceDetection.
 *
 * RealTimeFaceDetection has no static OpenCV initializer, so we load
 * the native library once in @BeforeAll before any test runs.
 */
class RealTimeFaceDetectionTest {

    @BeforeAll
    static void loadOpenCV() {
        Loader.load(opencv_java.class);
    }

    // -----------------------------------------------------------------------
    // bufferedImage2Mat
    // -----------------------------------------------------------------------

    @Test
    void bufferedImage2Mat_rgbImage_returnsNonEmptyMat() throws Exception {
        BufferedImage img = new BufferedImage(50, 60, BufferedImage.TYPE_3BYTE_BGR);
        Mat result = RealTimeFaceDetection.bufferedImage2Mat(img);
        assertNotNull(result);
        assertFalse(result.empty());
    }

    @Test
    void bufferedImage2Mat_preservesWidthAndHeight() throws Exception {
        int width = 125, height = 150;
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
        Mat result = RealTimeFaceDetection.bufferedImage2Mat(img);
        assertEquals(width,  result.cols());
        assertEquals(height, result.rows());
    }

    @Test
    void bufferedImage2Mat_squareImage_isSquareMat() throws Exception {
        BufferedImage img = new BufferedImage(80, 80, BufferedImage.TYPE_3BYTE_BGR);
        Mat result = RealTimeFaceDetection.bufferedImage2Mat(img);
        assertEquals(result.rows(), result.cols());
    }

    @Test
    void bufferedImage2Mat_outputIsGrayscale() throws Exception {
        BufferedImage img = new BufferedImage(50, 60, BufferedImage.TYPE_3BYTE_BGR);
        Mat result = RealTimeFaceDetection.bufferedImage2Mat(img);
        assertEquals(1, result.channels());
    }

    @Test
    void bufferedImage2Mat_intRgbImage_returnsNonEmptyMat() throws Exception {
        BufferedImage img = new BufferedImage(50, 60, BufferedImage.TYPE_INT_RGB);
        Mat result = RealTimeFaceDetection.bufferedImage2Mat(img);
        assertNotNull(result);
        assertFalse(result.empty());
    }

    // -----------------------------------------------------------------------
    // loadCascadeClassifier
    // -----------------------------------------------------------------------

    @Test
    void loadCascadeClassifier_returnsNonNullClassifier() throws Exception {
        CascadeClassifier classifier = RealTimeFaceDetection.loadCascadeClassifier();
        assertNotNull(classifier);
    }

    @Test
    void loadCascadeClassifier_classifierIsNotEmpty() throws Exception {
        CascadeClassifier classifier = RealTimeFaceDetection.loadCascadeClassifier();
        assertFalse(classifier.empty());
    }

    // -----------------------------------------------------------------------
    // matToBufferedImage
    // -----------------------------------------------------------------------

    @Test
    void matToBufferedImage_returnsNonNull() {
        Mat mat = new Mat(60, 50, CvType.CV_8UC3);
        BufferedImage result = RealTimeFaceDetection.matToBufferedImage(mat);
        assertNotNull(result);
    }

    @Test
    void matToBufferedImage_preservesDimensions() {
        int width = 120, height = 100;
        Mat mat = new Mat(height, width, CvType.CV_8UC3);
        BufferedImage result = RealTimeFaceDetection.matToBufferedImage(mat);
        assertEquals(width,  result.getWidth());
        assertEquals(height, result.getHeight());
    }

    @Test
    void matToBufferedImage_squareMat_isSquareImage() {
        Mat mat = new Mat(80, 80, CvType.CV_8UC3);
        BufferedImage result = RealTimeFaceDetection.matToBufferedImage(mat);
        assertEquals(result.getWidth(), result.getHeight());
    }

    // -----------------------------------------------------------------------
    // encodeMatToJpeg
    // -----------------------------------------------------------------------

    @Test
    void encodeMatToJpeg_returnsNonEmptyByteArray() {
        Mat mat = new Mat(100, 100, CvType.CV_8UC3);
        byte[] result = RealTimeFaceDetection.encodeMatToJpeg(mat);
        assertNotNull(result);
        assertTrue(result.length > 0);
    }

    @Test
    void encodeMatToJpeg_producesValidJpegMagicBytes() {
        Mat mat = new Mat(100, 100, CvType.CV_8UC3);
        byte[] result = RealTimeFaceDetection.encodeMatToJpeg(mat);
        // JPEG files start with FF D8 FF
        assertEquals((byte) 0xFF, result[0]);
        assertEquals((byte) 0xD8, result[1]);
        assertEquals((byte) 0xFF, result[2]);
    }

    @Test
    void encodeMatToJpeg_largerMatProducesLargerOutput() {
        Mat small = new Mat(10,  10,  CvType.CV_8UC3);
        Mat large = new Mat(200, 200, CvType.CV_8UC3);
        byte[] smallBytes = RealTimeFaceDetection.encodeMatToJpeg(small);
        byte[] largeBytes = RealTimeFaceDetection.encodeMatToJpeg(large);
        assertTrue(largeBytes.length > smallBytes.length);
    }
}