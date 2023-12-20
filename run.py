
import argparse
from norfair import Tracker, Video
from inference.yolov5 import YoloV5


parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)

args = parser.parse_args()

video = Video(input_path=args.video)

player_detector = YoloV5()


for i, frame in enumerate(video):
    person_df = player_detector.predict(frame)
    person_df = person_df[person_df["name"] == "person"]
    person_df = person_df[person_df["confidence"] > 0.35]
    print(person_df)

    
