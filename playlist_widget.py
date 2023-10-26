from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QSpinBox, QSlider, QListWidgetItem, QListWidget, QFileDialog, QPushButton, QLineEdit, QComboBox, QStackedWidget, QLabel, QVBoxLayout, QHBoxLayout, QWidget
from video.video_reader import OpenCV_VideoReader
from gui.helper.ndarray_to_qpixmap import NDarray_to_QPixmap
from gui.custom_widgets.labeled_slider_spinbox import LabeledSliderSpinBox
from gui.custom_widgets.labeled_spinbox import LabeledSpinBox
import cv2

class PlaylistWidget(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_reader = OpenCV_VideoReader()
        self.declare_components()
        self.layout_components()

        self.timer = QTimer()
        self.timer.timeout.connect(self.main)
        self.timer.setInterval(16)
        self.timer.start()

    def declare_components(self):

        # add zoom crop controls
        self.rescale = LabeledSliderSpinBox(self)
        self.rescale.setText('rescale (%)')
        self.rescale.setRange(10, 500)
        self.rescale.setValue(100)
        self.rescale.valueChanged.connect(self.on_crop_resize)

        self.left = LabeledSliderSpinBox(self)
        self.left.setText('left')
        self.left.setValue(0)
        self.left.valueChanged.connect(self.on_crop_resize)

        self.bottom = LabeledSliderSpinBox(self)
        self.bottom.setText('bottom')
        self.bottom.setValue(0)
        self.bottom.valueChanged.connect(self.on_crop_resize)

        self.width = LabeledSliderSpinBox(self)
        self.width.setText('width')
        self.width.setValue(100)
        self.width.valueChanged.connect(self.on_crop_resize)

        self.height = LabeledSliderSpinBox(self)
        self.height.setText('height')
        self.height.setValue(100)
        self.height.valueChanged.connect(self.on_crop_resize)

        self.add_button = QPushButton('add', self)
        self.add_button.clicked.connect(self.add_video)

        self.delete_button = QPushButton('delete', self)
        self.delete_button.clicked.connect(self.delete_video)

        self.video_list = QListWidget(self)
        self.video_list.currentRowChanged.connect(self.video_selected)

        self.previous_button = QPushButton('prev', self)
        self.previous_button.clicked.connect(self.previous_video)

        self.next_button = QPushButton('next', self)
        self.next_button.clicked.connect(self.next_video)

        self.zoom = LabeledSpinBox(self)
        self.zoom.setText('zoom (%)')
        self.zoom.setRange(25,500)
        self.zoom.setValue(50)
        self.zoom.setSingleStep(25)

        self.video_label = QLabel(self)

        self.playpause_button = QPushButton('play', self)
        self.playpause_button.setCheckable(True)
        self.playpause_button.clicked.connect(self.playpause_video)

        self.frame_slider = QSlider(Qt.Horizontal, self)
        self.frame_slider.valueChanged.connect(self.frame_changed_slider)

        self.frame_spinbox = QSpinBox(self)
        self.frame_spinbox.valueChanged.connect(self.frame_changed_spinbox)

    def layout_components(self):

        crop_resize = QVBoxLayout()
        crop_resize.addWidget(self.rescale)
        crop_resize.addWidget(self.left)
        crop_resize.addWidget(self.bottom)
        crop_resize.addWidget(self.width)
        crop_resize.addWidget(self.height)
    
        playlist_control0 = QHBoxLayout()
        playlist_control0.addWidget(self.add_button)
        playlist_control0.addWidget(self.delete_button)

        playlist_control1 = QHBoxLayout()
        playlist_control1.addWidget(self.previous_button)
        playlist_control1.addWidget(self.next_button)

        controls = QVBoxLayout()
        controls.addLayout(crop_resize)
        controls.addLayout(playlist_control0)
        controls.addWidget(self.video_list)
        controls.addLayout(playlist_control1)

        video_controls = QHBoxLayout()
        video_controls.addWidget(self.playpause_button)
        video_controls.addWidget(self.frame_slider)
        video_controls.addWidget(self.frame_spinbox)
        
        video_display = QVBoxLayout()
        video_display.addWidget(self.zoom)
        video_display.addWidget(self.video_label)
        video_display.addLayout(video_controls)

        mainlayout = QHBoxLayout(self)
        mainlayout.addLayout(controls)
        mainlayout.addLayout(video_display)

    def frame_changed_slider(self):
        self.frame_spinbox.setValue(self.frame_slider.value())
        self.frame_changed()

    def frame_changed_spinbox(self):
        self.frame_slider.setValue(self.frame_spinbox.value())
        self.frame_changed()

    def frame_changed(self):
        frame = self.frame_slider.value()
        if self.video_reader.is_open():
            self.video_reader.seek_to(frame)

    def add_video(self):
        file_name = QFileDialog.getOpenFileName(self, 'Select file')
        if file_name:
            list_item = QListWidgetItem(file_name[0])
            self.video_list.addItem(list_item)

    def delete_video(self):
        row = self.video_list.currentRow()
        if row:
            self.video_list.takeItem(row)

    def on_crop_resize(self):
        current_item = self.video_list.currentItem()
        if current_item:
            filename = current_item.text()

            resize = self.rescale.value()/100.0
            left = self.left.value()
            bottom = self.bottom.value()
            width = self.width.value()
            height = self.height.value()
            
            self.video_reader.open_file(
                filename, 
                crop = (left, bottom, width, height),
                resize = resize 
            )

            height_max = self.video_reader.get_height_max()
            width_max = self.video_reader.get_width_max()

            self.left.setRange(0, height_max-height)
            self.bottom.setRange(0, width_max-width)
            self.height.setRange(1, height_max-bottom) # TODO maybe -1
            self.width.setRange(1, width_max-left)

    def video_selected(self):
        current_item = self.video_list.currentItem()
        if current_item:
            filename = current_item.text()
            
            self.video_reader.open_file(
                filename,
                resize=0.5,
            )

            num_frames = self.video_reader.get_number_of_frame()
            height_max = self.video_reader.get_height_max()
            width_max = self.video_reader.get_width_max()
            
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(num_frames-1)
            self.frame_spinbox.setRange(0,num_frames-1)

            self.left.setRange(0, height_max-1)
            self.bottom.setRange(0, width_max-1)
            self.height.setRange(1, height_max)
            self.width.setRange(1, width_max)
            
            self.rescale.setValue(100)
            self.left.setValue(0)
            self.bottom.setValue(0)
            self.width.setValue(height_max)
            self.height.setValue(width_max)

    def previous_video(self):
        num_item = self.video_list.count()
        current_row = self.video_list.currentRow()
        previous_row = (current_row - 1) % num_item
        self.video_list.setCurrentRow(previous_row)

    def playpause_video(self):
        if self.playpause_button.isChecked():
            self.playpause_button.setText('pause')
        else:
            self.playpause_button.setText('play')

    def next_video(self):
        num_item = self.video_list.count()
        current_row = self.video_list.currentRow()
        next_row = (current_row + 1) % num_item
        self.video_list.setCurrentRow(next_row)

    def get_video_reader(self):
        return self.video_reader

    def main(self):
        if self.playpause_button.isChecked():
            if self.video_reader.is_open():
                ret, image = self.video_reader.next_frame()
                if ret:
                    frame_index = self.video_reader.get_current_frame_index()
                    self.frame_slider.setValue(frame_index)
                    self.frame_spinbox.setValue(frame_index)

                    scale = self.zoom.value()/100
                    image = cv2.resize(image,None,None,scale,scale,cv2.INTER_NEAREST)
                    self.video_label.setPixmap(NDarray_to_QPixmap(image))


    