# Face Recognition with OpenCV and Java

Real-time face detection and recognition using [JavaCV](https://github.com/bytedeco/javacv) (OpenCV Java bindings) and an EigenFace recognizer.

## How it works

1. **Training** — `RealTimeFaceRecognition` reads labelled face images from `src/main/resources/images/training/`, trains an `EigenFaceRecognizer`, and builds a label-to-name mapping.
2. **Detection** — `RealTimeFaceDetection` captures frames from the webcam, runs a Haar Cascade classifier to find faces, resizes each detected face to 125×150 px, and passes it to the recognizer.
3. **Recognition** — The predicted person's name is drawn on the frame alongside a bounding rectangle, and the live feed is displayed in a Swing window.

## Project structure

```
src/
├── main/
│   ├── java/tutorial/opencv/face/
│   │   ├── detection/RealTimeFaceDetection.java   # webcam capture + face detection
│   │   └── recognition/RealTimeFaceRecognition.java # EigenFace trainer + predictor
│   └── resources/
│       ├── haarcascade_frontalface_default.xml    # face detector model
│       └── images/
│           ├── training/   # labelled training images (format: <label>-<name>_<n>.png)
│           └── test/       # sample test images
└── test/
    └── java/tutorial/opencv/face/
        ├── detection/RealTimeFaceDetectionTest.java
        └── recognition/RealTimeFaceRecognitionTest.java
```

## Training image naming convention

Training images must follow the pattern `<label>-<name>_<index>.png`, for example:

```
1-andrew_1.png
2-aree_3.png
3-peeranut_0.png
```

All images must be the same size (125×150 px, grayscale).

## Requirements

- Java 8+
- Maven 3.x
- A webcam (for real-time detection)

## Build & run

```bash
# Run all tests and generate JaCoCo coverage report
mvn verify

# View coverage report
open target/site/jacoco/index.html

# Run the real-time face detection (requires webcam)
mvn exec:java -Dexec.mainClass="tutorial.opencv.face.detection.RealTimeFaceDetection"
```

## Dependencies

| Dependency | Version |
|---|---|
| [JavaCV](https://github.com/bytedeco/javacv) | 1.5.10 |
| [JavaCPP](https://github.com/bytedeco/javacpp) | 1.5.10 |
| JUnit Jupiter | 5.10.2 |
| JaCoCo | 0.8.14 |