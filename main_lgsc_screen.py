import cv2
import torch
from torchvision.utils import save_image
import numpy as np
from facenet_pytorch import MTCNN, extract_face
import argparse
import time
import yaml
import os
from mss import mss
from PIL import Image

from models.resnet import ResNet18Classifier
from dataset.siwm import get_test_augmentations
from models.lightning.pl_model_siwm import LightningModel


labels_map = {
    1: 'Live',
    0: 'Spoof'
}


class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(
            self,
            mtcnn_model: torch.nn.Module,
            clf_model: torch.nn.Module,
            output_path: str,
            configs: dict,
            device: str
    ):
        self.mtcnn = mtcnn_model
        self.model = clf_model
        self.output_path = output_path
        self.configs = configs
        self.device = device

    def _draw(self, frame, boxes, probs, landmarks, count):
        """
        Draw landmarks and boxes for each face detected
        """

        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # cv2 imwrite no need for RGB conversion
        cv2.imwrite(configs['frames_folder'] + "original" + str(count) + ".png", frame)
        for box, prob, ld in zip(boxes, probs, landmarks):
            cropped_img = extract_face(im_rgb, box, image_size=224,
                                       save_path=configs['frames_folder'] + str(count) + ".png")

            # Draw rectangle on frame
            cv2.rectangle(frame,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0, 0, 255),
                          thickness=2)

            transform = get_test_augmentations()

            # (C, H, W) -> (H, W, C)
            transformed_img = transform(image=np.array(cropped_img).transpose((1, 2, 0)))['image']
            # add batch dim
            transformed_img = transformed_img.unsqueeze(0)

            output = self.model.classify(transformed_img)
            prediction = torch.argmax(output, dim=1).cpu().numpy()

            # Show probability
            cv2.putText(frame,
                        "FDet: " + str(prob), (box[2], int(box[3] - 30.0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame,
                        str(labels_map.get(prediction[0])), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw landmarks
            # cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
            # cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)

        return frame

    def draw_one(self, frame, box, prob, landmark, count):
        """
        Draw landmarks and boxes for only one face detected
        """

        im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # cv2 imwrite no need for RGB conversion
        # cv2.imwrite(configs['frames_folder'] + "original" + str(count) + ".png", frame)

        cropped_img = extract_face(im_rgb, box, image_size=224)

        # Draw rectangle on frame
        cv2.rectangle(frame,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0, 0, 255),
                      thickness=2)

        transform = get_test_augmentations()

        # (C, H, W) -> (H, W, C)
        transformed_img = transform(image=np.array(cropped_img).transpose((1, 2, 0)))['image']
        # add batch dim
        transformed_img = transformed_img.unsqueeze(0)

        # transformed_img = transformed_img.to(device)
        cue = self.model.infer(transformed_img)

        # save cues
        # save_image(cue, configs['frames_folder'] + "cues/" + str(count) + ".png")

        score = cue.mean().cpu().item()

        # Show probability
        cv2.putText(frame,
                    "FDet: " + "{:.3f}".format(prob), (box[0], int(box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    "Spoof Score: " + "{:.6f}".format(score), (box[0], int(box[3] + 30.0)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2, cv2.LINE_AA)

        # Draw landmarks
        # cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
        # cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
        # cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
        # cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
        # cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)

        return frame

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        sct = mss()

        print("Capturing screen...")
        mon = sct.monitors[0]

        count = 1
        print("Starting stream processing...")
        while True:
            frame = sct.grab(mon)

            frame = np.array(frame)
            frame = np.flip(frame[:, :, :3], 2)

            # detect face box, probability and landmarks
            boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)

            # only draw if a face is detected
            if boxes is not None:
                # frame = self._draw(frame, boxes, probs, landmarks, count)
                frame = self.draw_one(frame, boxes[0], probs[0], landmarks[0], count)

                # Write the frame into the file 'output.avi'
            # cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('Face Detection', 600, 600)

            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # # Show the frame
            # cv2.imshow('Face Detection', frame)
            count += 1


if __name__ == "__main__":
    # default_device = "cuda:0" if torch.cuda.is_available() else None
    if not os.path.isdir('output'):
        os.makedirs('output')
    default_output_path = 'output/output.avi'

    parser = argparse.ArgumentParser(
        description='Run MTCNN face detector and vigilant model. Model is a ResNet-18 CNN.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-c', '--configs', type=str, required=True, help='Path to config file.')
    parser.add_argument('-o', '--output_file', type=str, default=default_output_path,  help='Path to output video file.')
    parser.add_argument('-d', '--debug', type=str, default=True,  help='Dump processed frames.')

    args = parser.parse_args()

    with open(args.configs, 'r') as stream:
        configs = yaml.safe_load(stream)

    # create frames folder
    if not os.path.isdir(configs['frames_folder']):
        os.makedirs(configs['frames_folder'])

    # create cues folder
    if not os.path.isdir(configs['frames_folder'] + "cues"):
        os.makedirs(configs['frames_folder'] + "cues")

    device = configs['device'] if torch.cuda.is_available() else None

    print("Starting...")
    start = time.time()

    # sct = mss()
    #
    # mon = sct.monitors[0]
    #
    # while 1:
    #     img = sct.grab(mon)
    #     # img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    #     cv2.imshow('test', np.array(img))
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break

    # Run the app
    mtcnn = MTCNN(image_size=224, post_process=False, device=device)
    model = LightningModel.load_from_checkpoint(configs['weights'])
    model.eval()
    # model.to(device)
    print("Model loaded from ", configs['weights'])

    fcd = FaceDetector(mtcnn, model, args.output_file, configs, device)
    print("Running face detector...")
    fcd.run()
    end = time.time()
    elapsed = end - start

    print("Running time: %f ms" % (elapsed * 1000))
    cv2.destroyAllWindows()
