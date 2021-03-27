#====== Libraries ======#
import sys;
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QMainWindow;
from PyQt5.QtWidgets import QMenuBar, QToolBar, QStatusBar;
from PyQt5.QtGui import QIcon;

# Set up relative DIR for referencing local filepaths
import os;
module_dir = os.path.dirname(os.path.realpath(__file__));


#====== Window Setup ======#
class AppMainWindow(QMainWindow):
    # MainWindow provides a framework for building the app's UI.
    # It supports 'QToolbar', 'QMenuBar', and 'QStatus' bars.
    # A MainWindow *must* have a 'central widget' to display.
    # https://doc.qt.io/qt-5/qmainwindow.html


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
        # Core Components
        menubar = self.initMenuBar();
        toolbar = self.initDrawingToolBar();
        statusbar = self.initStatusBar();
        # Central Widget
        main_content = AppMainContent();
        self.setCentralWidget(main_content);

        self.show();

    def centreWindow(self):
        rectangle_frame = self.frameGeometry();
        centre_point = QDesktopWidget().availableGeometry().center();
        rectangle_frame.moveCenter(centre_point);
        self.move(rectangle_frame.topLeft());

    def initMenuBar(self):
        menubar = self.menuBar();

        return menubar;

    def initDrawingToolBar(self):
        toolbar = self.addToolBar("Drawing Tools");

        return toolbar;

     def initStatusBar(self):
        statusbar = self.statusBar();
        statusbar.showMessage("Ready");
        return statusbar;



class AppMainContent(QWidget):
    # Our 'central widget' for the MainWindow frame.
    # Core content goes here.

    def __init__(self):
        super().__init__();


            


# If this module is run as main, execute the below:
if __name__ == '__main__':
    app_window = QApplication(sys.argv)
    ex = AppMainWindow()
    sys.exit(app_window.exec_())