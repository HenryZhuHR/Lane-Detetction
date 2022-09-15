import time
import cv2
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision import transforms
from model.model import parsingNet


img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def main():
    # NETWORK
    use_aux = False  # we dont need auxiliary segmentation in testing
    griding_num = 200
    backbone = '18'
    cls_num_per_lane = 18
    model_weight = 'culane_18.pth'
    device = torch.device('cuda')
    video_path = 'videos/video-1.mp4'
    culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]


    # ===============================
    #   Load model
    # ===============================
    model = parsingNet(pretrained=False,
                       backbone=backbone,
                       cls_dim=(griding_num+1, cls_num_per_lane, 4),
                       use_aux=use_aux)
    state_dict: dict = torch.load(model_weight, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    model.load_state_dict(compatible_state_dict, strict=False)
    model.to(device)
    model.eval()

    videoCapture = cv2.VideoCapture('videos/video-2.mp4')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('videos/output.mp4', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
    success, frame = videoCapture.read()
    frame: np.ndarray = frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    while success:
        st_time=time.time()
        img_np: np.ndarray = cv2.resize(frame, dsize=(800, 288))
        img_np = img_np/225.
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        img_np = (img_np-mean)/std
        # print(' ----- img_np',torch.Tensor(img_np.transpose(2,0,1)))
        # print(' ----- img_np',img_transforms(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
        img = torch.Tensor(img_np.transpose(2, 0, 1)).unsqueeze(0)
        img=img_transforms(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))).unsqueeze(0)

        with torch.no_grad():
            out: Tensor = model.forward(img.to(device))
            # print(out.size())  # torch.Size([1, 201, 18, 4])
        col_sample = np.linspace(0, 800 - 1, griding_num)
        col_sample_w = col_sample[1] - col_sample[0]
        out_j: np.ndarray = out[0].detach().cpu().numpy()
        out_j = out_j[:, ::-1, :]  # 倒序

        prob: np.ndarray = softmax(out_j[:-1, :, :], axis=0)
        
        idx = np.arange(griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == griding_num] = 0
        out_j = loc
        print(out_j)
        print(out_j.shape)

        for i in range(out_j.shape[1]): # 取列 i
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]): # 取行 
                    if out_j[k, i] > 0:
                        lane_point = (int(out_j[k, i] * col_sample_w * frame.shape[1] / 800) - 1,
                               int(frame.shape[0] * (culane_row_anchor[cls_num_per_lane-1-k]/288)) - 1)
                        cv2.circle(frame, lane_point, 5, (0, 255, 0), -1)

        ed_time=time.time()
        detect_time=ed_time-st_time

        text='FPS:%d time:%.4fms'%(1/detect_time,detect_time*1000)
        
        frame=cv2.putText(frame,text,org=(0, 50),fontFace=font,fontScale=1.2,color=(255, 255, 255),thickness=2)
        # cv2.imwrite('img-src.png', frame)


        # exit()

        cv2.waitKey(int(1000/int(fps))) #延迟
        videoWriter.write(frame) #写视频帧
        success, frame = videoCapture.read()  # 获取下一帧


if __name__ == "__main__":
    main()
