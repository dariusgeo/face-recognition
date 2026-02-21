package tutorial.opencv.face.detection;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.opencv.opencv_java;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;

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
        // bufferedImage2Mat reads with IMREAD_GRAYSCALE, so 1-channel output
        assertEquals(width,  result.cols());
        assertEquals(height, result.rows());
    }

    @Test
    void bufferedImage2Mat_squareImage_isSquareMat() throws Exception {
        BufferedImage img = new BufferedImage(80, 80, BufferedImage.TYPE_3BYTE_BGR);
        Mat result = RealTimeFaceDetection.bufferedImage2Mat(img);
        assertEquals(result.rows(), result.cols());
    }
}