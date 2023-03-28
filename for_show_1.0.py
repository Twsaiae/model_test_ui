from PyQt5 import QtCore
import sys
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os
from PyQt5.QtCore import pyqtSignal
from models import TRTModule  # isort:skip
import argparse
from pathlib import Path
import time
import json
import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list


class LoginWindow(QWidget):
    def __init__(self, stack_widget):
        # def __init__(self):
        super().__init__()
        self.stack_widget = stack_widget
        self.initUI()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('登录')
        self.setGeometry(200, 200, 200, 100)
        # 创建账号和密码标签和文本框
        self.username_label = QLabel('账号:')
        self.username_edit = QLineEdit()
        self.password_label = QLabel('密码:')
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)

        # 创建登录按钮
        self.login_button = QPushButton('登录')

        # 设置布局
        layout = QGridLayout(self)
        layout.addWidget(self.username_label, 0, 0)
        layout.addWidget(self.username_edit, 0, 1)
        layout.addWidget(self.password_label, 1, 0)
        layout.addWidget(self.password_edit, 1, 1)
        layout.addWidget(self.login_button, 2, 0, 1, 2)

        # # 设置窗口布局
        # self.setLayout(hbox)

        # 绑定登录按钮点击事件
        self.login_button.clicked.connect(self.login)
        shortcut = QShortcut(self)
        shortcut.setKey(Qt.Key_Return)  # 设置回车键与按钮的快捷键为Return
        shortcut.activated.connect(self.login)

    def login(self):
        # 获取账号和密码文本框中的内容
        username = self.username_edit.text()
        password = self.password_edit.text()

        # 检查账号和密码是否正确
        if username == 'Alin' and password == '123':
            # 如果正确，弹出提示框，然后关闭窗口
            # QMessageBox.information(self, '提示', '登录成功！')
            self.stack_widget.setCurrentIndex(1)
        else:
            # 如果错误，弹出提示框，并清空密码文本框
            QMessageBox.warning(self, '提示', '账号或密码错误！')
            self.password_edit.clear()


class MainWindow(QWidget):
    # def __init__(self, stack_widget, show_image):
    output_signal = pyqtSignal(list)

    def __init__(self, stack_widget):
        super().__init__()
        self.stack_widget = stack_widget
        # self.show_image = show_image
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 500, 200)
        self.setWindowTitle('文件检测器')

        # 创建标签和文本框
        self.label = QLabel('地址:', self)
        self.label.move(20, 20)
        self.textbox = QLineEdit(self)
        self.textbox.move(80, 20)
        self.textbox.resize(280, 30)

        # 创建文件选择按钮
        self.file_button = QPushButton('选择文件', self)
        self.file_button.move(380, 20)
        self.file_button.clicked.connect(self.choose_file)

        # 创建检测按钮
        self.detect_button = QPushButton('检测', self)
        self.detect_button.move(200, 80)
        self.detect_button.clicked.connect(self.detect)

    def choose_file(self):
        # 打开文件对话框选择文件或文件夹
        self.filename = QFileDialog.getOpenFileName(self, '选择文件', '',
                                                    'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)')
        # 将选择的文件路径显示在文本框中
        self.textbox.setText(self.filename[0])

    def detect(self):
        if os.path.exists(self.filename[0]):
            # img1 = cv2.imread(self.filename[0])
            cls_list = ['LN', 'RS', 'SC']
            color_list = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
            device = torch.device('cuda:0')
            Engine = TRTModule("best.engine", device)
            H, W = Engine.inp_info[0].shape[-2:]
            Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

            bgr = cv2.imread(self.filename[0])
            # bgr = cv2.imread(str(args.images))
            draw = bgr.copy()
            bgr, ratio, dwdh = letterbox(bgr, (W, H))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=device)
            tensor = torch.asarray(tensor, device=device)
            # inference
            # end1 = time.time()
            data = Engine(tensor)
            # end2 = time.time()
            bboxes, scores, labels = det_postprocess(data)
            bboxes -= dwdh
            bboxes /= ratio

            for (bbox, score, label) in zip(bboxes, scores, labels):
                bbox = bbox.round().int().tolist()
                cls_id = int(label)
                # cls = CLASSES[cls_id]
                # color = COLORS[cls]
                cls = cls_list[cls_id]
                color = color_list[cls_id]
                cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
                cv2.putText(draw,
                            f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, [225, 255, 255],
                            thickness=2)
            img2 = draw
            # self.show_image.output_show(self.filename[0], img1, img2)
            self.output_signal.emit([self.filename[0], bgr, img2])

            self.stack_widget.setCurrentIndex(2)


class ImageDisplay(QWidget):
    # def __init__(self, name, img1, img2, width=300, height=300, padding=20):
    def __init__(self, stack_widget, width=300, height=300, padding=20):
        # super().__init__(stack_widget)
        super().__init__()
        self.stack_widget = stack_widget
        self.w = width
        self.h = height
        self.p = padding
        self.setWindowTitle("检测结果")
        self.setGeometry(100, 100, 2 * width + 3 * padding, height + 3 * padding + 40)

        # 创建标题标签和图像标签
        # self.title_label = QLabel(name, self)
        self.title_label = QLabel(self)
        self.title_label.setGeometry(padding, padding, 2 * width, 20)
        self.title_label.setStyleSheet('font-size: 16px; font-weight: bold;')

        self.img1_label = QLabel(self)
        self.img1_label.setGeometry(padding, 40 + padding, width, height)
        self.img1_label.setAlignment(QtCore.Qt.AlignCenter)

        self.img2_label = QLabel(self)
        self.img2_label.setGeometry(width + 2 * padding, 40 + padding, width, height)
        self.img2_label.setAlignment(QtCore.Qt.AlignCenter)

        # 添加文字注释标签
        self.img1_text_label = QLabel("原图", self)
        self.img1_text_label.setGeometry(padding, height + 2 * padding + 40, width, 20)
        self.img1_text_label.setStyleSheet('font-size: 16px; font-weight: bold;')

        self.img1_text_label.setAlignment(QtCore.Qt.AlignCenter)

        self.img2_text_label = QLabel("结果图", self)
        self.img2_text_label.setGeometry(width + 2 * padding, height + 2 * padding + 40, width, 20)
        self.img2_text_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        self.img2_text_label.setAlignment(QtCore.Qt.AlignCenter)

        # 创建文件选择按钮
        self.return_button = QPushButton('返回', self)
        self.return_button.move(540, 20)
        self.return_button.clicked.connect(self.switch_window)

    def switch_window(self):
        # 切换窗口
        self.stack_widget.setCurrentIndex(1)

    def output_show(self, input_list):
        # 将图像转换为QPixmap并设置到标签上
        self.title_label.setText(input_list[0])
        img1_pixmap = self.cvimage2qpixmap(input_list[1], self.w, self.h)
        img2_pixmap = self.cvimage2qpixmap(input_list[2], self.w, self.h)
        self.img1_label.setPixmap(img1_pixmap)
        self.img2_label.setPixmap(img2_pixmap)

    def cvimage2qpixmap(self, image, width, height):
        """将cv2读入的np格式的图像转换为QPixmap格式并resize到指定大小"""
        image = cv2.resize(image, (width, height))
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimg = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 创建QStackedWidget，并添加两个界面
    stack_widget = QStackedWidget()
    stack_widget.setWindowTitle('ES检测系统')
    login_widget = LoginWindow(stack_widget)
    detection_widget = MainWindow(stack_widget)
    image_show_widget = ImageDisplay(stack_widget)
    detection_widget.output_signal.connect(image_show_widget.output_show)

    stack_widget.addWidget(login_widget)
    stack_widget.addWidget(detection_widget)
    stack_widget.addWidget(image_show_widget)

    stack_widget.show()

    sys.exit(app.exec_())
