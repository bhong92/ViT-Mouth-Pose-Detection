from PyQt5 import QtCore,QtWidgets,QtGui
from main_win.win import Ui_mainWindow

from faceDetection import Detection

from sys import argv, exit
from cv2 import VideoWriter, resize, flip, cvtColor, COLOR_BGR2RGB, imwrite, CAP_PROP_FPS, VideoWriter_fourcc
from os import makedirs, path
from time import strftime, localtime
import time
import eyetracking
import click
from audio import speech_detect
from threading import Thread




class MainWindow():
    def __init__(self):
        # if want to do data analysis: set evaluate=True, ues_eye=True. If want to use app normaly, set them to False
        self.tracker = eyetracking.Tracker(evaluate=False,use_eye=False)
        self.tclick = click.T_click()
        self.bclick = click.Blink()
        time.sleep(6)
        app = QtWidgets.QApplication(argv)
        MainWindow = QtWidgets.QMainWindow()
        self.raw_image = None
        self.ui = Ui_mainWindow()
        self.ui.setupUi(MainWindow)
        flag = self.ui.cap.open(0)
        if flag == False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"no camera",
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.ui.timer_camera.start(20)
        self.action_connect()
        self.enableDet = False
        self.save_fold = './result'
        self.frameCount = 0
        MainWindow.show()
        self.audio = speech_detect.SpeechDetect()
        exit(app.exec_())


    def action_connect(self):

        self.ui.runButton.clicked.connect(self.run_or_continue)
        self.ui.fileButton.clicked.connect(self.train_blink)

        # self.ui.cameraButton.clicked.connect(self.button_open_camera_click)
        # self.ui.resetButton.clicked.connect(self.resetCounter)
        self.ui.timer_camera.timeout.connect(self.show_camera)
        self.ui.saveCheckBox.clicked.connect(self.is_save)


    def train_blink(self):
        # self.bclick.Bclick_train()
        # self.bclick.task1 = Thread(target=self.bclick.Bclick_detect, args=())
        # self.bclick.task1.start()
        pass

    # def showCounter(self):


    def show_camera(self):
        flag, self.camera_image = self.ui.cap.read()
        self.frameCount += 1

        if self.enableDet or self.audio.cursor_enable:
            self.tracker.demo.ok = flag
            self.tracker.demo.frame = self.camera_image
            self.tracker.demo.camstart = True
            self.tracker.curosr_speed_ratio = self.audio.cursor_speed_ratio
            self.tclick.camstart = True
            self.tclick.frame = self.camera_image
            self.bclick.camstart = True
            self.bclick.frame = self.camera_image
        else:
            self.tracker.demo.camstart = False
            self.tclick.camstart = False
            self.bclick.camstart = False
        img_src = self.camera_image
        # -------------TODO: find a way to use eyetracking stream
        # if self.tracker.demo.visualizer.image is not None:
        #     img_src = self.tracker.demo.visualizer.image
        # -------------
        ih, iw, _ = img_src.shape
        w = self.ui.out_video.geometry().width()
        h = self.ui.out_video.geometry().height()

        if iw > ih:
            scal = w / iw
            nw = w
            nh = int(scal * ih)
            img_src_ = resize(img_src, (nw, nh))

        else:
            scal = h / ih
            nw = int(scal * iw)
            nh = h
            img_src_ = resize(img_src, (nw, nh))

        img_src_ = flip(img_src_, 1)

        show = cvtColor(img_src_, COLOR_BGR2RGB)

        self.autoSave(img_src_)

        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], show.shape[2] * show.shape[1], QtGui.QImage.Format_RGB888)
        self.ui.out_video.setPixmap(QtGui.QPixmap.fromImage(showImage))

        # self.showCounter()

    # def resetCounter(self):
    #     self.showCounter()

    def is_save(self):
        if self.ui.saveCheckBox.isChecked():
            self.save_fold = './result'
        else:
            self.save_fold = None

    def autoSave(self, img):

        if self.save_fold:
            makedirs(self.save_fold, exist_ok=True)  
          
            if self.ui.cap is None:
                save_path = path.join(self.save_fold,
                                         strftime('%Y_%m_%d_%H_%M_%S',
                                                       localtime()) + '.jpg')
                imwrite(save_path, img)
            else:
                if self.frameCount == 1: 
                    
                    ori_fps = int(self.ui.cap.get(CAP_PROP_FPS))
                    if ori_fps == 0:
                        ori_fps = 25
                    width, height = img.shape[1], img.shape[0]
                    save_path = path.join(self.save_fold,
                                             strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.mp4')
                    self.out = VideoWriter(save_path, VideoWriter_fourcc(*"mp4v"), ori_fps,
                                               (width, height))
                if self.frameCount > 0:
                    self.out.write(img)

    def button_open_camera_click(self):
        if self.ui.cameraButton.isChecked():
            if self.ui.cap.isOpened():
                self.ui.cap.release()
            if self.ui.timer_camera.isActive():
                self.ui.timer_camera.stop()

            if self.ui.timer_camera.isActive() == False:
                flag = self.ui.cap.open(0)
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"no camera",
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.ui.timer_camera.start(20)
        else:
            self.ui.timer_camera.stop()
            self.ui.cap.release()
            self.ui.out_video.clear()

    def run_or_continue(self):
        if self.ui.runButton.isChecked():
            self.enableDet = True
        else:
            self.enableDet = False


if __name__ == "__main__":
    train_tongue = input("Do you want to train tongue model? enter 'y' or 'n'\n")
    if train_tongue == 'y' or train_tongue == 'Y':
        click.T_click().t_click_train()
    else:
        MainWindow()
