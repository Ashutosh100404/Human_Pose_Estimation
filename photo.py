import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Read image
image_path = "pose.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image. Check the file path.")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        print("Pose landmarks detected!")

        # Get image dimensions
        h, w, c = image.shape

        # Draw landmarks on image
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")

        # Draw pose connections
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display results
        cv2.imshow("Pose Landmarks", image)
        cv2.imshow("Pose Drawings", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Release resources
pose.close()
