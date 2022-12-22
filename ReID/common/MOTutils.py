import cv2
import random

class MOTline:
    def __init__(self, frame:int, id:int,bb_left:int, bb_top:int, bb_width:int, bb_height:int, 
    visibility:float, class_:int, confidence:float, x:float, y:float, z:float):
        self.frame_ = frame
        self.id_ = id
        self.bb_left_ = bb_left
        self.bb_top_ = bb_top
        self.bb_width_ = bb_width
        self.bb_height_ = bb_height
        self.visibility = visibility,
        self.class_ = class_
        self.confidence_ = confidence
        self.x_ = x
        self.y_ = y
        self.z_ = z

class MOTannotation:
    def __init__(self, path:str):
        annotations = open(path, 'r')
        self.data_ = {}
        for line in annotations:
            l = line.strip().split(',')
            v = MOTline(int(l[0]),int(l[1]),int(l[2]),int(l[3]),int(l[4]),int(l[5]),float(l[6]),int(l[7]),float(l[8]),float(l[9]),float(l[10]),float(l[11]))
            if not v.frame_ in self.data_.keys():
                self.data_[v.frame_] = []
            self.data_[v.frame_].append(v)
        annotations.close()

    def __getitem__(self, frame:int):
        return self.data_[frame]


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    #print(f"c1:{c1}, c2:{c2}")
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)