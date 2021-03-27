#====== Libraries ======#
import sys;
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QMainWindow;
from PyQt5.QtWidgets import QMenuBar, QToolBar, QStatusBar, QAction;
from PyQt5.QtWidgets import QTextBrowser, QPushButton, QVBoxLayout;
from PyQt5.QtGui import QIcon;
#from PyQt5.QtCore import Qt;

# Set up relative DIR for referencing local filepaths
# Could replace with Qt resource system, but having issues recognising pyrcc5
# https://doc.qt.io/qt-5/resources.html
import os;
MODULE_DIR = os.path.dirname(os.path.realpath(__file__));
RESOURCES_DIR = MODULE_DIR + "\\resources\\";



#====== Window Setup ======#
# See here for full guide: https://realpython.com/python-menus-toolbars/

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
        WINDOW_ICON_NAME = "icon_robot.png"
        # Code
        self.setWindowTitle(WINDOW_TITLE);
        self.resize(WINDOW_SIZE_X, WINDOW_SIZE_Y);
        self.setWindowIcon(QIcon(RESOURCES_DIR + WINDOW_ICON_NAME));
        self.centreWindow();
        # Core Components
        self.initActions();
        self.menubar = self.initMenuBar();
        self.toolbar = self.initMainToolBar();
        self.statusbar = self.initStatusBar();
        # Central Widget
        main_content = AppMainContent();
        self.setCentralWidget(main_content);

        self.show();

    def centreWindow(self):
        rectangle_frame = self.frameGeometry();
        centre_point = QDesktopWidget().availableGeometry().center();
        rectangle_frame.moveCenter(centre_point);
        self.move(rectangle_frame.topLeft());

    def initActions(self):
        # Actions defined here are owned by AppMainWindow and persist
        # Exit
        self.exitAction = QAction("&Exit", self);
        self.exitAction.setIcon(QIcon(RESOURCES_DIR + "icon_exit.svg"));
        self.exitAction.setShortcut("Ctrl+E");
        self.exitAction.triggered.connect(self.exitApp);
        # Open
        self.openAction = QAction("&Open", self);
        self.openAction.setIcon(QIcon(RESOURCES_DIR + "icon_open.svg"));
        self.openAction.setToolTip("Select new training images");
        self.openAction.setStatusTip("Select new training images");
        self.openAction.setShortcut("Ctrl+O");
        # Help 
        self.helpAction = QAction("Help", self);
        self.helpAction.setIcon(QIcon(RESOURCES_DIR + "icon_help.svg"));
        self.helpAction.triggered.connect(self.helpDialogue);
        # Draw
        self.drawAction = QAction("&Draw", self);
        self.drawAction.setIcon(QIcon(RESOURCES_DIR + "icon_draw.svg"));
        self.drawAction.setToolTip("Start drawing on the canvas");
        self.drawAction.setStatusTip("Start drawing on the canvas");
        self.drawAction.setShortcut("Ctrl+D");
        # View
        self.viewTrainingImagesAction = QAction("View Training Images", self);
        self.viewTestingImagesAction = QAction("View Testing Imaged", self);

        # Note: Add actions to context menus for drawing canvas
        # https://realpython.com/python-menus-toolbars/#creating-context-menus-through-context-menu-policy


    def initMenuBar(self):
        self.menubar = self.menuBar();

        self.fileMenu = self.menubar.addMenu("&File");
        self.fileMenu.addAction(self.openAction);
        self.fileMenu.addAction(self.helpAction);
        self.fileMenu.addSeparator();
        self.fileMenu.addAction(self.exitAction);

        self.viewMenu = self.menubar.addMenu("&View");
        self.viewMenu.addAction(self.viewTrainingImagesAction);
        self.viewMenu.addAction(self.viewTestingImagesAction);

        #submenu = menu.addMenu("name");
        #submenu.addAction(...);

        return self.menubar;

    def initMainToolBar(self):
        self.toolbar = self.addToolBar("Main Tools");
        self.toolbar.addAction(self.openAction);
        self.toolbar.addSeparator();
        self.toolbar.addAction(self.drawAction);

        #self.toolbar.addWidget(...);
        return self.toolbar;

    def initStatusBar(self):
        self.statusbar = self.statusBar();
        self.statusbar.showMessage("Ready");

        #self.formatted_label = QLabel(f"{self.func()} text");
        #self.statusbar.addPermanentWidget(...);
        return self.statusbar;

    def helpDialogue(self):
        self.popup = PopupBox();
        self.popup.assignText("<b>Icon credit to:</b>");
        self.popup.assignText('Icons made by <a href="https://www.freepik.com">Freepik</a> from <a href="https://www.flaticon.com/">www.flaticon.com</a>.');
        self.popup.assignText('Icons made by <a href="https://iconmonstr.com">iconmonstr</a>.');

    def exitApp(self):
        self.close();



class AppMainContent(QWidget):
    # Our 'central widget' for the MainWindow frame.
    # Core content goes here.

    def __init__(self):
        super().__init__();



class PopupBox(QWidget):
    # Text box that pops up in a new windows.
    # Useful for displaying reports and detailed information.
    
    def __init__(self):
        super().__init__();
        self.initPopup();

    def initPopup(self):
        self.setWindowTitle("Popup Box");
        self.setGeometry(400, 300, 400, 400);
        
        self.button = QPushButton("Close", self);
        self.button.clicked.connect(self.exitPopup);
        self.browser = QTextBrowser();
        self.browser.setAcceptRichText(True);
        self.browser.setOpenExternalLinks(True);

        self.vbox = QVBoxLayout();
        self.vbox.addWidget(self.browser);
        self.vbox.addWidget(self.button);

        self.setLayout(self.vbox);
        self.show();

    def assignText(self, message):
        self.browser.append(message);

    def exitPopup(self):
        self.close();


# If this module is run as main, execute the below:
if __name__ == '__main__':
    app = QApplication(sys.argv)
    appwin = AppMainWindow()
    sys.exit(app.exec_())