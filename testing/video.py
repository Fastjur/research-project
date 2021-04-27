from fer import Video
from fer import FER

video_filename = "./videos/woman2.mp4"
video = Video(video_filename)

# Analyze video, displaying the output
detector = FER(mtcnn=True)
raw_data = video.analyze(detector, display=True)
df = video.to_pandas(raw_data)
