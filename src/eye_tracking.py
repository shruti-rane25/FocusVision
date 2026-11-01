import cv2
import mediapipe as mp
import numpy as np
import time


def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    return (A + B) / (2.0 * C)


def eye_tracking():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access webcam.")
        return

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    left_eye = [362, 385, 387, 263, 373, 380]
    right_eye = [33, 160, 158, 133, 153, 144]

    EAR_THRESHOLD = 0.21
    CONSEC_FRAMES = 2

    blink_count = 0
    closed_frames = 0
    closure_start = None
    eye_closed_duration = 0.0

    print("FocusVision: Eye Tracking Started.")
    print("Press 'q' to close the camera.")

    while camera.isOpened():
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left = np.array([
                [landmarks[i].x * w, landmarks[i].y * h]
                for i in left_eye
            ])

            right = np.array([
                [landmarks[i].x * w, landmarks[i].y * h]
                for i in right_eye
            ])

            left_ear = calculate_ear(left)
            right_ear = calculate_ear(right)
            avg_ear = (left_ear + right_ear) / 2

            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1

                if closure_start is None:
                    closure_start = time.perf_counter()

                eye_closed_duration = (
                    time.perf_counter() - closure_start
                )

                eye_status = "Closed"

            else:
                eye_status = "Open"

                if closed_frames >= CONSEC_FRAMES:
                    blink_count += 1

                if closure_start is not None:
                    eye_closed_duration = (
                        time.perf_counter() - closure_start
                    )

                closure_start = None
                closed_frames = 0

            cv2.putText(
                frame, f"EAR: {avg_ear:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            cv2.putText(
                frame, f"Eye Status: {eye_status}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )

            cv2.putText(
                frame, f"Blink Count: {blink_count}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2
            )

            cv2.putText(
                frame, f"Closure Duration: {eye_closed_duration:.2f}s",
                (20, 145), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2
            )

            for point in np.concatenate((left, right)):
                cv2.circle(
                    frame, tuple(point.astype(int)),
                    2, (0, 255, 255), -1
                )

        else:
            cv2.putText(
                frame, "No Face Detected",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2
            )

        cv2.imshow(
            "FocusVision - Eye Tracking and Blink Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()

    print("Total Blinks Detected:", blink_count)
    print("Last Eye Closure Duration:", round(eye_closed_duration, 2), "seconds")
    print("FocusVision: Camera Stopped.")


if __name__ == "__main__":
    eye_tracking()