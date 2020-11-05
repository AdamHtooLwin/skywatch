import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, extract_face
import argparse
import time
import yaml

from models.resnet import ResNet18Classifier


class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn, video):
        self.mtcnn = mtcnn
        self.video = video

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Draw rectangle on frame
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)

                # Show probability
                cv2.putText(frame, str(
                    prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                # cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except:
            pass

        return frame

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """

        cap = cv2.VideoCapture(self.video)

        # Check if video opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(5)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output/output.avi', fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                # draw on frame
                frame = self._draw(frame, boxes, probs, landmarks)

                # Write the frame into the file 'output.avi'
                out.write(frame)

                # # Show the frame
                # cv2.imshow('Face Detection', frame)
            else:
                break

        cap.release()
        out.release()


if __name__ == "__main__":
    default_device = "cuda:0" if torch.cuda.is_available() else None

    parser = argparse.ArgumentParser(
        description='Extract frames from CASIA-FASD videos.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('video', type=str, help='Absolute path to input video.')
    parser.add_argument('-d', '--device', type=str, default=default_device, help='Device to run detector and classifier on.')
    parser.add_argument('-c', '--configs', type=str, required=True, help='Path to config file.')

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        configs = yaml.safe_load(stream)

    device = configs['device'] if torch.cuda.is_available() else None

    print("Starting...")
    start = time.time()

    # Run the app
    mtcnn = MTCNN(device=device)
    fcd = FaceDetector(mtcnn, args.video)
    fcd.run()
    end = time.time()
    elapsed = end - start

    print("Running time: %f ms" % (elapsed * 1000))
