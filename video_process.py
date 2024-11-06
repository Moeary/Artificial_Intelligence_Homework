import ffmpeg
import argparse
def video_to_frame(video_path, frame_path):
    (
        ffmpeg
        .input(video_path)
        .output(frame_path + '/%05d.jpg', r=1, f='image2')
        .run()
    )

def video_to_audio(video_path, audio_path):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path)
        .run()
    )    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='video.mp4')
    parser.add_argument('--frame_path', type=str, default='frames')
    parser.add_argument('--audio_path', type=str, default='audio.mp3')
    python_args = parser.parse_args()
    video_to_frame(python_args.video_path, python_args.frame_path, python_args.audio_path)