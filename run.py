
import argparse
from run_utils import *
from norfair import Tracker, Video
from inference import Converter, YoloV5


parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default="models/ball.pt", type=str, help="Path to the model"
)

args = parser.parse_args()

video = Video(input_path=args.video)

player_detector = YoloV5()
ball_detector = YoloV5(model_path=args.model)


j=0
for i, frame in enumerate(video):
    if j==1:
        break
    players_detections = get_player_detections(player_detector, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    detections = ball_detections + players_detections
    
    for detection in detections:
        print(detection)
    j+=1

    
