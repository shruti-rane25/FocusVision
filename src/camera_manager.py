import cv2

def start_camera():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("FocusVision: Camera started.")
    print("Press 'q' to close the camera window.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read the camera frame.")
            break

        cv2.imshow("FocusVision - Live Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("FocusVision: Camera stopped.")

if __name__ == "__main__":
    start_camera()
