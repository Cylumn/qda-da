import cv2

from model import AffectRecognitionPipeline


def process_frame(frame,
                  flipped: bool = True):
    if flipped:
        frame = cv2.flip(frame, 1)

    return frame


def init_webcam(id_window: str = 'affect_recognition'):
    cv2.namedWindow(id_window)
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        rval, frame = vc.read()
        frame = process_frame(frame)
    else:
        rval = False

    while rval:
        cv2.imshow(id_window, frame)
        rval, frame = vc.read()

        # Apply transformations to frame
        frame = process_frame(frame)

        # Detect affect
        frame = pipeline(frame)

        # Exit on ESC
        key = cv2.waitKey(20)
        if key == 27:
            break

    vc.release()
    cv2.destroyWindow(id_window)


if __name__ == '__main__':
    pipeline = AffectRecognitionPipeline("haarcascade_frontalface_default.xml")

    init_webcam()
