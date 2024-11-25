import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('weights/ours.pt') 
    model.predict(source='demo_video/vessel1.mp4',
                  project='runs/detect',
                  name='output',
                  save=True,
                  show_labels = False,
                  show_conf = False,
                  line_width = 3,
                  )