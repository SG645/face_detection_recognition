/**
 * Real-time Face Detection and Recognition System
 * Uses OpenCV and OpenCV Face module for face detection and recognition
 */

 #include <opencv2/opencv.hpp>
 #include <opencv2/face.hpp>
 #include <iostream>
 #include <map>
 #include <vector>
 #include <string>
 
 // Detects faces in an input frame using Haar Cascade Classifier
 std::vector<cv::Rect> detectFaces(cv::Mat& frame, cv::CascadeClassifier& faceDetector) {
     std::vector<cv::Rect> faces;
     cv::Mat grayFrame;
     
     // Convert frame to grayscale for face detection
     cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
     cv::equalizeHist(grayFrame, grayFrame); // Improve detection in varying lighting
     
     // Detect faces
     faceDetector.detectMultiScale(grayFrame, faces, 1.1, 3, 0, cv::Size(30, 30));
     return faces;
 }
 
 // Adds a new face to the recognition system
 bool addNewFace(cv::VideoCapture& cap, cv::CascadeClassifier& faceDetector,
                 cv::Ptr<cv::face::LBPHFaceRecognizer>& recognizer,
                 std::vector<cv::Mat>& faceImages, std::vector<int>& faceLabels,
                 std::map<int, std::string>& nameMap, int& nextLabel) {
     
     std::string name;
     std::cout << "Enter name for the new face: ";
     std::cin.ignore();
     std::getline(std::cin, name);
     
     std::cout << "Capturing face images. Please look at the camera..." << std::endl;
     
     std::vector<cv::Mat> tempFaces;
     int count = 0;
     const int REQUIRED_FACES = 10;
     
     while (count < REQUIRED_FACES) {
         cv::Mat frame;
         cap >> frame;
         if (frame.empty()) break;
         
         cv::flip(frame, frame, 1); // Mirror image for more intuitive interaction
         
         std::vector<cv::Rect> faces = detectFaces(frame, faceDetector);
         
         if (faces.size() == 1) { // Ensure only one face is detected
             cv::Rect faceRect = faces[0];
             cv::Mat faceROI = frame(faceRect);
             cv::Mat grayFace;
             cv::cvtColor(faceROI, grayFace, cv::COLOR_BGR2GRAY);
             cv::resize(grayFace, grayFace, cv::Size(100, 100));
             
             tempFaces.push_back(grayFace.clone());
             count++;
             
             // Display progress
             cv::rectangle(frame, faceRect, cv::Scalar(0, 255, 0), 2);
             std::string msg = "Capturing: " + std::to_string(count) + "/" + std::to_string(REQUIRED_FACES);
             cv::putText(frame, msg, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
         } else {
             // Guide the user
             std::string msg = "Position your face in the frame (only one face)";
             cv::putText(frame, msg, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
         }
         
         cv::imshow("Add New Face", frame);
         
         // Allow user to cancel with ESC
         if (cv::waitKey(100) == 27) {
             cv::destroyWindow("Add New Face");
             return false;
         }
     }
     
     cv::destroyWindow("Add New Face");
     
     if (count == REQUIRED_FACES) {
         // Add all captured faces to the training set
         for (const auto& face : tempFaces) {
             faceImages.push_back(face);
             faceLabels.push_back(nextLabel);
         }
         
         // Store name mapping
         nameMap[nextLabel] = name;
         nextLabel++;
         
         // Train or update the recognizer
         if (faceImages.size() == tempFaces.size()) {
             // First person, train the recognizer
             recognizer->train(faceImages, faceLabels);
         } else {
             // Update existing model with new face
             std::vector<int> newLabels(tempFaces.size(), nextLabel - 1);
             recognizer->update(tempFaces, newLabels);
         }
         
         std::cout << "New face added successfully!" << std::endl;
         return true;
     } else {
         std::cout << "Failed to capture enough face images." << std::endl;
         return false;
     }
 }
 
 // Main function
 int main() {
     // Step 1: Load the face detection model
     cv::CascadeClassifier faceDetector;
     if (!faceDetector.load("haarcascade_frontalface_default.xml")) {
         std::cerr << "Error: Could not load face detector model!" << std::endl;
         std::cerr << "Make sure 'haarcascade_frontalface_default.xml' is in the current directory." << std::endl;
         return -1;
     }
     
     // Step 2: Create face recognizer
     cv::Ptr<cv::face::LBPHFaceRecognizer> recognizer = cv::face::LBPHFaceRecognizer::create();
     
     // Step 3: Initialize variables for face recognition
     std::vector<cv::Mat> faceImages;  // Images of faces
     std::vector<int> faceLabels;      // Labels for each face image
     std::map<int, std::string> nameMap; // Mapping from label to person name
     int nextLabel = 0;                // Next available label
     bool recognizerTrained = false;   // Flag to track if recognizer has been trained
     
     // Step 4: Open webcam
     cv::VideoCapture cap(0);
     if (!cap.isOpened()) {
         std::cerr << "Error: Could not open webcam!" << std::endl;
         return -1;
     }
     
     // Print instructions
     std::cout << "===== Real-time Face Detection and Recognition System =====" << std::endl;
     std::cout << "Press 'a' to add a new face to the system" << std::endl;
     std::cout << "Press 'q' or ESC to quit" << std::endl;
     
     // Step 5: Main processing loop
     while (true) {
         // Capture frame from webcam
         cv::Mat frame;
         cap >> frame;
         if (frame.empty()) break;
         
         cv::flip(frame, frame, 1); // Mirror image for more intuitive interaction
         
         // Detect faces in the current frame
         std::vector<cv::Rect> faces = detectFaces(frame, faceDetector);
         
         // Process each detected face
         for (const auto& faceRect : faces) {
             // Draw rectangle around the face
             cv::rectangle(frame, faceRect, cv::Scalar(0, 255, 0), 2);
             
             // Perform face recognition if the recognizer has been trained
             if (recognizerTrained) {
                 // Extract and preprocess the face region
                 cv::Mat faceROI = frame(faceRect);
                 cv::Mat grayFace;
                 cv::cvtColor(faceROI, grayFace, cv::COLOR_BGR2GRAY);
                 cv::resize(grayFace, grayFace, cv::Size(100, 100));
                 
                 // Recognize the face
                 int label = -1;
                 double confidence = 0.0;
                 recognizer->predict(grayFace, label, confidence);
                 
                 // Display recognition result
                 std::string name;
                 if (confidence < 70 && nameMap.find(label) != nameMap.end()) {
                     // Known face with good confidence
                     name = nameMap[label] + " (" + std::to_string(int(100 - confidence)) + "%)";
                     cv::putText(frame, name, cv::Point(faceRect.x, faceRect.y - 10),
                                 cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                 } else {
                     // Unknown face or low confidence
                     name = "Unknown";
                     cv::putText(frame, name, cv::Point(faceRect.x, faceRect.y - 10),
                                 cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                 }
             }
         }
         
         // Show system status
         std::string statusMsg = recognizerTrained ? 
                               "Recognition ACTIVE - " + std::to_string(nameMap.size()) + " faces in database" :
                               "Recognition INACTIVE - Add faces to database";
         cv::putText(frame, statusMsg, cv::Point(10, frame.rows - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
         
         // Display the frame
         cv::imshow("Face Detection and Recognition", frame);
         
         // Handle key presses
         int key = cv::waitKey(1);
         if (key == 27 || key == 'q') { // ESC or 'q' to quit
             break;
         } else if (key == 'a') { // 'a' to add new face
             if (addNewFace(cap, faceDetector, recognizer, faceImages, faceLabels, nameMap, nextLabel)) {
                 recognizerTrained = true;
             }
         }
     }
     
     // Step 6: Clean up resources
     cap.release();
     cv::destroyAllWindows();
     
     return 0;
 }
 