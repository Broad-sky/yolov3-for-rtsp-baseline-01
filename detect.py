import argparse
import time
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import ftplib
import io
import requests
import json
from ftplib import FTP
import datetime
from io import BytesIO

def connectFtp():
    HOST = ""
    port = 0
    user = ""
    passwd = ""
    ftp = ftplib.FTP()  # 实例化FTP对象
    ftp.connect(HOST, port)
    ftp.login(user, passwd)  # 登录
    # ftp.set_pasv(False)  # 如果被动模式由于某种原因失败，请尝试使用活动模式。
    # print(ftp.getwelcome())
    print('已连接到： %s' % HOST)
    # print(ftp.pwd())
    return ftp

def ftpPicture(fp,ftp,picName = 'webcam.jpg'):
    # ftp.mkd("imgnet")
    # fp = open("F:\yoloImages\webcam.jpg", "rb")
    # print(len(fp.read()))
    # ftp.set_debuglevel(2)
    buffer_size = 1024
    ftp.set_pasv(True)
    ftp.storbinary('STOR '+ picName, fp,buffer_size)
    # print(ftp.dir())
    # ftp.set_debuglevel(0)

def detect(save_txt=False, save_img=False, stream_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half = opt.output, opt.source, opt.weights, opt.half
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')

    # Initialize
    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    if os.path.exists(out):
        pass
        # shutil.rmtree(out)  # delete output folder
    else:
        os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size).cuda()

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()
    # Export mode
    # if ONNX_EXPORT:
    #     img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
    #     torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
    #     return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    print("half: ",half)
    if half:
        model.half()
    # Set Dataloader
    vid_path, vid_writer = None, None

    if webcam:
        stream_img = True
        dataset = LoadWebcam(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    #ftp to server
    i = 0
    flg = 0
    ftp = connectFtp()

    # Run inference
    t0 = time.time()
    tftp = time.time()
    for path, img, im0, vid_cap in dataset:
        # post interface
        nowTime = datetime.datetime.now()
        strNowIime = nowTime.strftime("%Y-%m-%d %H:%M:%S")
        res_data = {
            "deviceIp": "",
            "channelID": '0',  # 通道id,对应摄像机通道号
            "presetNum": '32',  # 预置位编号（如获取不到，请传0）
            "alarmTime": strNowIime,  # 告警时间
            "aDAlarmResult": {
                "alarmLevel": 2,  # 告警级别，1，2，3
                "alarmCount": 3,  # 触发报警的目标数量
                "alarmItemList": [
                ]
            },
        }
        #ptf connect anagement
        t00 = time.time() - tftp
        if t00%60>= 30:
            strTest = "keep ftp alive"
            fpb = BytesIO(strTest.encode("UTF-8"))
            buffer_size = 1024
            ftp.storbinary('STOR log.txt', fpb, buffer_size)
            tftp = 0

        t = time.time()
        save_path = str(Path(out) / Path(path).name)
        print("save_path: ",save_path)
        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        pred, _ = model(img)
        det = non_max_suppression(pred.float(), opt.conf_thres, opt.nms_thres)[0]

        s = '%gx%g ' % img.shape[2:]  # print string
        labels = []
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string

            # Write results
            for *xyxy, conf, _, cls in det:
                # if save_txt:  # Write to file
                #     with open(save_path + '.txt', 'a') as file:
                #         file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                if save_img or stream_img:  # Add bbox to image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    clsname = ["car"]#, "truck", "car"
                    if classes[int(cls)] in clsname:
                        alarmItemList = {
                            "targetType": 85,
                            # 81-行人 82-自行车 83-摩托车 84-三轮车 85-小型车 86-面包车 87-卡车 88-大巴车 89-拖拉机 90-挖掘机 91-铲车
                            "alarmImgUrl": "webCam_" + str(i) + ".jpg",
                            "targetRectTop": int(xyxy[0]),  # 目标在图像中的坐标值,距上
                            "targetRectBottom": int(xyxy[1]),  # 目标在图像中的坐标值，距下
                            "targetRectLeft": int(xyxy[2]),  # 目标在图像中的坐标值，距左
                            "targetRectRight": int(xyxy[3]),  # 目标在图像中的坐标值，距右
                        }
                        res_data["aDAlarmResult"]["alarmItemList"].append(alarmItemList)
                        flg = 1
                    labels.append(classes[int(cls)])
                    print("xyxy: ",int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('%sDone. (%.3fs)' % (s, time.time() - t))
        # Stream results

        if flg==1:
            imgRGB = cv2.cvtColor(im0, cv2.IMREAD_COLOR)
            r, buf = cv2.imencode(".jpg", imgRGB)
            bytes_image = Image.fromarray(np.uint8(buf)).tobytes()
            # array转换成二进制
            fp = io.BytesIO(bytes_image)
            # if ftp==None:
            #     ftp = connectFtp()
            ftpPicture(fp, ftp, "webCam_" + str(i) + ".jpg")
            i= i+1
            flg = 0
            fp.close()
            # ftp.close()
            print("ftp success!!")
            res_data["aDAlarmResult"]["alarmCount"] =len(res_data["aDAlarmResult"]["alarmItemList"])
            print(res_data)
            headers = {
                "Content-Type": "application/json",
                "charset": "UTF-8",
                "keep": "alive",
                # "Accept-Encoding":"gzip, deflate"
            }
            data = json.dumps(res_data)
            r = requests.post("",json = data,headers = headers)
            print("r.status_code: ",r.status_code)
            with open("log.txt","w") as fs:
                fs.write(str(res_data))
                fs.write(str(r.status_code))
                fs.write(str(r.content))
            fs.close()
            # break
                # cv2.imwrite(save_path, im0)
        # labels.clear()
        if stream_img:
            # cv2.imwrite(save_path, im0)
            # cv2.resizeWindow("enhanced", 640, 480)
            cv2.imshow("demo", im0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save results (image with detections)
        # if save_img:
        #     if dataset.mode == 'images':
        #         cv2.imwrite(save_path, im0)
        #     else:
        #         if vid_path != save_path:  # new video
        #             vid_path = save_path
        #             if isinstance(vid_writer, cv2.VideoWriter):
        #                 vid_writer.release()  # release previous video writer
        #
        #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
        #             width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #             height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (width, height))
        #         vid_writer.write(im0)

    ftp.close()
    # if save_txt or save_img:
    #     print('Results saved to %s' % os.getcwd() + os.sep + out)
    #     if platform == 'darwin':  # MacOS
    #         os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    url_rtsp = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default=url_rtsp, help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='D:\yolov3-master-ultralytics\output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half',default="True", action='store_true', help='half precision FP16 inference')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(stream_img=True)
