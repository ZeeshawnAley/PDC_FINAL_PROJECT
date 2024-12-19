using System;
using System.Diagnostics;
using System.Threading.Tasks;
using OpenCvSharp;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Enter the path to the image file:");
        string imagePath = Console.ReadLine();

        if (string.IsNullOrEmpty(imagePath))
        {
            Console.WriteLine("Invalid image path. Exiting...");
            return;
        }

        Mat image;
        try
        {
            image = Cv2.ImRead(imagePath, ImreadModes.Color);
            if (image.Empty())
            {
                Console.WriteLine("Failed to load the image. Exiting...");
                return;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading image: {ex.Message}");
            return;
        }

        Console.WriteLine("Do you want to process the image? (yes/no):");
        string response = Console.ReadLine()?.ToLower();

        if (response != "yes")
        {
            Console.WriteLine("Image processing aborted by the user. Exiting...");
            return;
        }

        Console.WriteLine("Starting Image Processing...");

        // Serial Processing
        Stopwatch stopwatch = Stopwatch.StartNew();
        Mat serialResult = SerialProcessing(image);
        stopwatch.Stop();
        double serialTime = stopwatch.ElapsedMilliseconds;
        Console.WriteLine($"Serial Processing Time: {serialTime} ms");

        // Parallel Processing
        stopwatch.Restart();
        Mat parallelResult = ParallelProcessing(image);
        stopwatch.Stop();
        double parallelTime = stopwatch.ElapsedMilliseconds;
        Console.WriteLine($"Parallel Processing Time: {parallelTime} ms");

        // Compute Metrics
        double psnr = ComputePSNR(serialResult, parallelResult);
        double ssim = ComputeSSIM(serialResult, parallelResult);

        Console.WriteLine("Performance Metrics: ");
        Console.WriteLine($"PSNR: {psnr}");
        Console.WriteLine($"SSIM: {ssim}");

        // Display Results
        Cv2.ImShow("Original Image", image);
        Cv2.ImShow("Serial Result", serialResult);
        Cv2.ImShow("Parallel Result", parallelResult);

        Cv2.ImWrite("serial_result.jpg", serialResult);
        Cv2.ImWrite("parallel_result.jpg", parallelResult);

        // Scalability Testing
        TestScalability(image);

        Console.WriteLine("Processing Complete. Press any key to exit.");
        Cv2.WaitKey(0);
    }

    static Mat SerialProcessing(Mat image)
    {
        Mat grayImage = new Mat();
        Mat blurredImage = new Mat();
        Mat edgeDetectedImage = new Mat();

        // Convert to grayscale
        Cv2.CvtColor(image, grayImage, ColorConversionCodes.BGR2GRAY);

        // Apply Gaussian blur
        Cv2.GaussianBlur(grayImage, blurredImage, new Size(5, 5), 1.5);

        // Perform edge detection
        Cv2.Canny(blurredImage, edgeDetectedImage, 100, 200);

        EnhanceImage(grayImage);
        SegmentImage(grayImage);
        DetectObjects(grayImage);

        return edgeDetectedImage.Clone();
    }

    static Mat ParallelProcessing(Mat image)
    {
        Mat grayImage = new Mat();
        Mat blurredImage = new Mat();
        Mat edgeDetectedImage = new Mat();

        // Convert to grayscale and blur in parallel
        Parallel.Invoke(
            () => Cv2.CvtColor(image, grayImage, ColorConversionCodes.BGR2GRAY),
            () => Cv2.GaussianBlur(image, blurredImage, new Size(5, 5), 1.5)
        );

        // Edge detection 
        Cv2.Canny(blurredImage, edgeDetectedImage, 100, 200);

        EnhanceImage(grayImage);
        SegmentImage(grayImage);
        DetectObjects(grayImage);

        return edgeDetectedImage.Clone();
    }

    static void EnhanceImage(Mat image)
    {
        Cv2.EqualizeHist(image, image);
    }

    static void SegmentImage(Mat image)
    {
        Mat binary = new Mat();
        Cv2.Threshold(image, binary, 128, 255, ThresholdTypes.Binary);
    }

    static void DetectObjects(Mat image)
    {
        HierarchyIndex[] hierarchy = new HierarchyIndex[0];
        Point[][] contours;
        Cv2.FindContours(image, out contours, out hierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
        foreach (var contour in contours)
        {
            Cv2.DrawContours(image, new[] { contour }, -1, new Scalar(0, 255, 0), 2);
        }
    }

    static void TestScalability(Mat image)
    {
        Console.WriteLine("Testing Scalability with Different Resolutions...");
        int[] scales = { 50, 100, 200, 400 };
        foreach (var scale in scales)
        {
            Mat resized = new Mat();
            Cv2.Resize(image, resized, new Size(image.Width * scale / 100, image.Height * scale / 100));

            Stopwatch stopwatch = Stopwatch.StartNew();
            Mat result = ParallelProcessing(resized);
            stopwatch.Stop();
            Console.WriteLine($"Scale: {scale}% - Processing Time: {stopwatch.ElapsedMilliseconds} ms");
        }
    }

    static double ComputePSNR(Mat img1, Mat img2)
    {
        Mat diff = new Mat();
        Cv2.Absdiff(img1, img2, diff);
        diff.ConvertTo(diff, MatType.CV_32F);
        diff = diff.Mul(diff);

        Scalar s = Cv2.Sum(diff);
        double sse = s.Val0 + s.Val1 + s.Val2; // Sum of squared errors

        if (sse <= 1e-10) return 0; // Avoid divide-by-zero

        double mse = sse / (double)(img1.Total() * img1.Channels());
        double psnr = 10.0 * Math.Log10((255 * 255) / mse);
        return psnr;
    }

    static double ComputeSSIM(Mat img1, Mat img2)
    {
        Mat img1f = new Mat();
        Mat img2f = new Mat();
        img1.ConvertTo(img1f, MatType.CV_32F);
        img2.ConvertTo(img2f, MatType.CV_32F);

        Mat mu1 = new Mat();
        Mat mu2 = new Mat();
        Cv2.GaussianBlur(img1f, mu1, new Size(11, 11), 1.5);
        Cv2.GaussianBlur(img2f, mu2, new Size(11, 11), 1.5);

        Mat mu1Sq = mu1.Mul(mu1);
        Mat mu2Sq = mu2.Mul(mu2);
        Mat mu1Mu2 = mu1.Mul(mu2);

        Mat sigma1Sq = new Mat();
        Mat sigma2Sq = new Mat();
        Mat sigma12 = new Mat();

        Cv2.GaussianBlur(img1f.Mul(img1f), sigma1Sq, new Size(11, 11), 1.5);
        Cv2.GaussianBlur(img2f.Mul(img2f), sigma2Sq, new Size(11, 11), 1.5);
        Cv2.GaussianBlur(img1f.Mul(img2f), sigma12, new Size(11, 11), 1.5);

        sigma1Sq -= mu1Sq;
        sigma2Sq -= mu2Sq;
        sigma12 -= mu1Mu2;

        const double C1 = 6.5025, C2 = 58.5225;

        Mat C1Mat = new Mat(img1.Size(), MatType.CV_32F, new Scalar(C1));
        Mat C2Mat = new Mat(img1.Size(), MatType.CV_32F, new Scalar(C2));

        Mat numerator1 = mu1Mu2 * 2 + C1Mat;
        Mat numerator2 = sigma12 * 2 + C2Mat;
        Mat denominator1 = mu1Sq + mu2Sq + C1Mat;
        Mat denominator2 = sigma1Sq + sigma2Sq + C2Mat;

        Mat ssimMap = (numerator1.Mul(numerator2)) / (denominator1.Mul(denominator2));
        Scalar mssim = Cv2.Mean(ssimMap);
        return mssim.Val0;
    }
}
