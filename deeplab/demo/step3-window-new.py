import shutil
import PySide6
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *
import threading
import sys
import cv2
import os.path as osp
import time
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os

# DEVICE = 'cuda:0' # 如果您的电脑是GPU的设备，请使用这个
DEVICE = 'cpu'  # 如果您的电脑只有CPU，请使用这个

SYSTEM_INFO = ("以下是一些常见的治疗建议："
               "\n1.药物治疗：详细医疗建议3"
               "\n2.物理因子治疗：详细医疗建议2"
               "\n3.手术治疗：详细医疗建议3")
Window_TITLE = "图像分割系统"
ICON_PATH = "../images/UI/lufei.png"
ZUOZHE = "<a href='http://www.cnsa.gov.cn/'>作者：XXX</a>"
WELCOME_LABEL = '欢迎使用基于深度学习的语义分割系统'


def model_load(model_path, config_path, device):
    # model = init_segmentor(config_path, model_path, device=DEVICE)
    model = init_model(config_path, model_path, device=device)
    return model


def get_result(model, img_path):
    """
    获取图像输出的结果
    Args:
        model:  已经加载好的模型
        img_path: 需要进行分割的图像

    Returns:

    """
    # 通过模型直接进行推理和输出
    result = inference_model(model, img_path)
    result = show_result_pyplot(model, img_path, result, show=False, out_file='tmp/images_tmp.jpg', opacity=1.0)
    return result


class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle(Window_TITLE)
        self.resize(1200, 800)
        self.setWindowIcon(QIcon(ICON_PATH))
        # 图片读取进程
        self.output_size = 480
        # todo 修改为你要加载得模型
        self.device = DEVICE
        self.MODEL_PATH = 'work_dirs/twins_pcpvt-s_uperhead_8xb4-160k_ade20k-512x512/iter_20000.pth'
        self.CONFIG_PATH = 'work_dirs/twins_pcpvt-s_uperhead_8xb4-160k_ade20k-512x512/twins_pcpvt-s_uperhead_8xb4-160k_ade20k-512x512.py'
        self.seg_model = model_load(model_path=self.MODEL_PATH,
                                    config_path=self.CONFIG_PATH, device=self.device)
        self.img2predict = ""

        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.initUI()

    '''
    ***界面初始化***
    '''

    def initUI(self):
        # 图片检测子界面 整体是左右布局
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        img_detection_widget = QWidget()
        img_detection_layout = QHBoxLayout()
        #################### 图片布局 #################################
        right_img_widget = QWidget()
        right_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("../images/UI/up380.jpeg"))
        self.right_img.setPixmap(QPixmap("../images/UI/right380.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        right_img_layout.addWidget(self.left_img)
        right_img_layout.addStretch(0)
        right_img_layout.addWidget(self.right_img)
        right_img_widget.setLayout(right_img_layout)
        left_show_widget = QWidget()
        left_show_layout = QVBoxLayout()
        left_show_title = QLabel("检测结果实时输出信息")
        left_show_title.setFont(font_title)
        left_show_title.setAlignment(Qt.AlignCenter)
        img_info_widget = QWidget()
        img_info_layout = QGridLayout()
        img_info_label0 = QLabel("图片规格：")
        img_info_label2 = QLabel("本图检测耗时：")
        img_info_label3 = QLabel("治疗建议")
        self.img_info_edit0 = QLineEdit("")
        self.img_info_edit1 = QLineEdit("")
        self.img_info_edit2 = QLineEdit("")
        self.img_info_edit3 = QTextEdit("")
        self.img_info_edit3.setText(SYSTEM_INFO)
        # img_info_edit4 = QLineEdit("")
        img_info_layout.addWidget(img_info_label0, 0, 0)
        img_info_layout.addWidget(self.img_info_edit0, 0, 1)
        # img_info_layout.addWidget(img_info_label1)
        # img_info_layout.addWidget(self.img_info_edit1)
        img_info_layout.addWidget(img_info_label2)
        img_info_layout.addWidget(self.img_info_edit2)
        img_info_layout.addWidget(img_info_label3)
        img_info_layout.addWidget(self.img_info_edit3)
        # img_info_layout.addWidget(img_info_label4)
        # img_info_layout.addWidget(img_info_edit4)
        img_info_widget.setLayout(img_info_layout)
        img_info_widget.setFont(font_main)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        left_show_layout.addWidget(left_show_title)
        left_show_layout.addWidget(right_img_widget)
        left_show_layout.addWidget(img_info_widget)
        left_show_layout.addWidget(up_img_button)
        left_show_layout.addWidget(det_img_button)

        self.model_label = QLabel("当前模型：{}".format(self.CONFIG_PATH))
        self.model_label.setFont(font_main)

        change_model_button = QPushButton("切换模型")
        change_model_button.setFont(font_main)
        change_model_button.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(236,99,97);}"
                                          "QPushButton{background-color:rgb(255,99,97)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:5px 5px}"
                                          "QPushButton{margin:5px 5px}")

        left_show_layout.addWidget(self.model_label)
        left_show_layout.addStretch()
        left_show_layout.addWidget(change_model_button)
        left_show_layout.addStretch()

        left_show_widget.setLayout(left_show_layout)
        img_detection_layout.addWidget(left_show_widget)
        # img_detection_layout.addWidget(right_img_widget, alignment=Qt.AlignCenter)
        img_detection_widget.setLayout(img_detection_layout)
        vid_detection_widget = QWidget()
        vid_detection_layout = QHBoxLayout()
        vid_info_widget = QWidget()
        vid_info_layout = QGridLayout()
        vid_info_label0 = QLabel("视频单帧规格：")
        vid_info_label1 = QLabel("目标检测数量：")
        vid_info_label2 = QLabel("目标坐标：")
        vid_info_label3 = QLabel("FPS：")
        vid_info_label4 = QLabel("当前检测总帧数：")
        vid_info_edit0 = QLineEdit("")
        vid_info_edit1 = QLineEdit("")
        vid_info_edit2 = QLineEdit("")
        vid_info_edit3 = QLineEdit("")
        vid_info_edit4 = QLineEdit("")
        vid_info_layout.addWidget(vid_info_label0, 0, 0)
        vid_info_layout.addWidget(vid_info_edit0, 0, 1)
        vid_info_layout.addWidget(vid_info_label1)
        vid_info_layout.addWidget(vid_info_edit1)
        vid_info_layout.addWidget(vid_info_label2)
        vid_info_layout.addWidget(vid_info_edit2)
        vid_info_layout.addWidget(vid_info_label3)
        vid_info_layout.addWidget(vid_info_edit3)
        vid_info_layout.addWidget(vid_info_label4)
        vid_info_layout.addWidget(vid_info_edit4)
        vid_info_widget.setLayout(vid_info_layout)
        vid_info_widget.setFont(font_main)
        vid_info_title = QLabel("检测结果实时输出")
        vid_info_title.setAlignment(Qt.AlignCenter)
        vid_info_title.setFont(font_title)
        self.webcam_detection_btn = QPushButton("摄像头实时监测")
        self.mp4_detection_btn = QPushButton("视频文件检测")
        self.vid_stop_btn = QPushButton("停止检测")
        self.webcam_detection_btn.setFont(font_main)
        self.mp4_detection_btn.setFont(font_main)
        self.vid_stop_btn.setFont(font_main)
        vid_left_widget = QWidget()
        vid_left_layout = QVBoxLayout()
        vid_left_layout.addWidget(vid_info_title)
        vid_left_layout.addWidget(vid_info_widget)
        vid_left_layout.addWidget(self.webcam_detection_btn)
        vid_left_layout.addWidget(self.mp4_detection_btn)
        vid_left_layout.addWidget(self.vid_stop_btn)
        vid_left_widget.setLayout(vid_left_layout)
        self.webcam_detection_btn.clicked.connect(self.open_cam)
        self.mp4_detection_btn.clicked.connect(self.open_mp4)
        self.vid_stop_btn.clicked.connect(self.close_vid)
        self.vid_img = QLabel()
        self.vid_img.setPixmap(QPixmap("../images/UI/up.jpeg"))
        self.vid_img.setAlignment(Qt.AlignCenter)

        vid_detection_layout.addWidget(vid_left_widget)
        vid_detection_layout.addWidget(self.vid_img)
        vid_detection_widget.setLayout(vid_detection_layout)

        # todo 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel(WELCOME_LABEL)  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('../images/UI/logo.jpg'))
        about_img.setAlignment(Qt.AlignCenter)

        label_super = QLabel()  # todo 更换作者信息
        label_super.setText(ZUOZHE)
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        # label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        # about_layout.addWidget(self.model_label)
        # about_layout.addStretch()
        # about_layout.addWidget(change_model_button)
        # about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        change_model_button.clicked.connect(self.change_model)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(about_widget, '关于')
        self.addTab(img_detection_widget, '图片检测')
        # self.addTab(vid_detection_widget, '视频检测')
        self.setTabIcon(0, QIcon('../images/UI/lufei.png'))

        self.setTabPosition(QTabWidget.West)
        # 设置背景颜色
        img_detection_widget.setStyleSheet("background-color:rgb(255,250,205);")
        # self.setTabIcon(1, QIcon('images/UI/lufei.png'))
        # self.setTabIcon(2, QIcon('images/UI/lufei.png'))

    def change_model(self):
        # change_mode
        print("模型切换")
        # fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.pt')
        directory = QFileDialog.getExistingDirectory(self,
                                                     "选取文件夹",
                                                     "./")  # 起始路径
        if directory:
            # 先判断有没有，再进行切换
            tmp_model_path = ""
            tmp_config_path = ""
            file_names = os.listdir(directory)
            for file_name in file_names:
                if file_name.split(".")[-1] == "pth":
                    tmp_model_path = osp.join(directory, file_name)
                if file_name.split(".")[-1] == "py":
                    tmp_config_path = osp.join(directory, file_name)
            if tmp_config_path == "" or tmp_model_path == "":
                QMessageBox.information(self, "切换失败", "请检查是否选择了正确的模型路径")
            else:
                self.MODEL_PATH = tmp_model_path
                self.CONFIG_PATH = tmp_config_path
                self.seg_model = model_load(model_path=self.MODEL_PATH,
                                            config_path=self.CONFIG_PATH, device=self.device)
                self.model_label.setText("当前模型为{}".format(osp.basename(self.CONFIG_PATH)))
                QMessageBox.information(self, "切换成功", "模型已切换")

            # self.model_path = fileName
            # self.model = self.model_load(weights=self.model_path,
            #                              device=self.device)  # todo 指明模型加载的位置的设备
            # QMessageBox.information(self, "成功", "模型切换成功！")
            # self.model_label.setText("当前模型：{}".format(self.model_path))

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("../images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap("../images/UI/right.jpeg"))
            self.img_info_edit0.setText("")
            self.img_info_edit2.setText("")

    '''
    ***检测图片***
    '''

    def detect_img(self):
        start_time = time.time()
        img_size = str(cv2.imread(self.img2predict).shape)
        result = get_result(self.seg_model, img_path=self.img2predict)
        resize_scale = self.output_size / result.shape[0]
        im0 = cv2.resize(result, (0, 0), fx=resize_scale, fy=resize_scale)
        cv2.imwrite("images/tmp/result_show.jpg", im0)
        self.right_img.setPixmap(QPixmap("images/tmp/result_show.jpg"))
        end_time = time.time()
        time_str = str(round(float(end_time - start_time), 3)) + "ms"
        self.img_info_edit0.setText(img_size)
        self.img_info_edit2.setText(time_str)

    '''
    ### 界面关闭事件 ### 
    '''

    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

    '''
    ### 视频关闭事件 ### 
    '''

    def open_cam(self):
        self.webcam_detection_btn.setEnabled(False)
        self.mp4_detection_btn.setEnabled(False)
        self.vid_stop_btn.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### 开启视频文件检测事件 ### 
    '''

    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.webcam_detection_btn.setEnabled(False)
            self.mp4_detection_btn.setEnabled(False)
            # self.vid_stop_btn.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
    ### 视频开启事件 ### 
    '''

    def detect_vid(self):
        print("视频检测逻辑")

    '''
    ### 视频重置事件 ### 
    '''

    def reset_vid(self):
        self.webcam_detection_btn.setEnabled(True)
        self.mp4_detection_btn.setEnabled(True)
        self.vid_img.setPixmap(QPixmap("../images/UI/up.jpeg"))
        self.vid_source = '0'
        self.webcam = True

    '''
    ### 视频重置事件 ### 
    '''

    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
