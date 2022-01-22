import argparse
import cv2
import numpy as np
import onnxruntime

class u2net():
    def __init__(self):
        try:
            cvnet = cv2.dnn.readNet('u2net_portrait.onnx')
        except:
            print('opencv read onnx failed!!!')
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.net = onnxruntime.InferenceSession('u2net_portrait.onnx', so)
        self.input_size = 512
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name
    def detect(self, srcimg):
        img = cv2.resize(srcimg, dsize=(self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        img = (img / 255.0 - self.mean) / self.std
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        outs = self.net.run([self.output_name], {self.input_name: blob})
        # outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})
        
        result = np.array(outs[0]).squeeze()
        result = (1 - result)
        min_value = np.min(result)
        max_value = np.max(result)
        result = (result - min_value) / (max_value - min_value)
        result *= 255
        return result.astype('uint8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='sample.jpg')
    args = parser.parse_args()
    
    mynet = u2net()
    srcimg = cv2.imread(args.imgpath)
    result = mynet.detect(srcimg)
    result = cv2.resize(result, (srcimg.shape[1], srcimg.shape[0]))

    cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
    cv2.imshow('srcimg', srcimg)
    winName = 'Deep learning object detection in onnxruntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()