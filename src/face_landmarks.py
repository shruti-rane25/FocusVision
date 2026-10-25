import cv2
import mediapipe as mp


def detect_facial_landmarks():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access the webcam.")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    drawing = mp.solutions.drawing_utils

    print("FocusVision: Facial landmark detection started.")
    print("Press 'q' to close the window.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION
                )

            cv2.putText(
                frame,
                "Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.imshow("FocusVision - Facial Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()

    print("FocusVision: Facial landmark detection stopped.")


if __name__ == "__main__":
    detect_facial_landmarks()