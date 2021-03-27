#====== Libraries ======#
import sys;
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget;
from PyQt5.QtGui import QIcon;

# Set up relative DIR for referencing local filepaths
import os;
module_dir = os.path.dirname(os.path.realpath(__file__));

#====== Window Setup ======#
class MainApp(QWidget):

        def __init__(self):
            super().__init__();
            self.initUI();

        def initUI(self):
            # Defines
            WINDOW_SIZE_X = 800;
            WINDOW_SIZE_Y = 600;
            WINDOW_TITLE = "CNN Handwriting Recogniser";
            WINDOW_ICON_DIR = "\\Graphics\\";
            WINDOW_ICON_NAME = "Icon_UoA.png"
            # Code
            self.setWindowTitle(WINDOW_TITLE);
            self.resize(WINDOW_SIZE_X, WINDOW_SIZE_Y);
            self.setWindowIcon(QIcon(module_dir + WINDOW_ICON_DIR + WINDOW_ICON_NAME));
            self.centreWindow();
            
            self.show();

        def centreWindow(self):
            rectangle_frame = self.frameGeometry();
            centre_point = QDesktopWidget().availableGeometry().center();
            rectangle_frame.moveCenter(centre_point);
            self.move(rectangle_frame.topLeft());

            


# If this module is run as main, execute the below:
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainApp()
    sys.exit(app.exec_())