#====== mainUI.py ======#
# The MainWindow of our app.
# The MainWindow provides a framework for the UI, with our QToolBar,
# QMenuBar, and QStatusBar implemented here. Most actions and
# shortcuts are also initialised here.
# A QMainWindow *must* have a 'central widget' to display, which holds
# the our app's content. Our central widget is under 'maincUI.py'.
# https://doc.qt.io/qt-5/qmainwindow.html


#====== Libraries ======#
import sys;
from PyQt5.QtWidgets import *;
from PyQt5.QtGui import QIcon;
#from PyQt5.QtCore import Qt;

import resources as r;
import maincUI as c;
import peripheralUI;



#====== Window Setup ======#
# See here for full guide: https://realpython.com/python-menus-toolbars/
class AppMainWindow(QMainWindow):

    def __init__(self):
        super().__init__();
        self.initUI();

    def initUI(self):
        # Defines
        WINDOW_SIZE_X = 600;
        WINDOW_SIZE_Y = 400;
        WINDOW_TITLE = "CNN Handwriting Recogniser";
        self.layout = QVBoxLayout
        # Code
        self.setWindowTitle(WINDOW_TITLE);
        self.setFixedSize(WINDOW_SIZE_X, WINDOW_SIZE_Y);
        self.setWindowIcon(QIcon(r.ICON_WINDOW));
        self.centreWindow();
        # Core Components
        self.initActions();
        self.menubar = self.initMenuBar();
        self.toolbar = self.initMainToolBar();
        self.statusbar = self.initStatusBar();

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
        self.exitAction.setIcon(QIcon(r.ICON_EXIT));
        self.exitAction.setShortcut("Ctrl+E");
        self.exitAction.triggered.connect(self.exitApp);
        # Train Model
        self.trainModel = QAction("&Train Model", self);
        self.trainModel.setToolTip("Train handwriting recognition model");
        self.trainModel.setStatusTip("Train handwriting recognition model");
        self.trainModel.triggered.connect(self.modelTraining);
        # Help 
        self.helpAction = QAction("Help", self);
        self.helpAction.setIcon(QIcon(r.ICON_HELP));
        self.helpAction.triggered.connect(self.helpDialogue);
        # Draw
        self.drawAction = QAction("&Draw", self);
        self.drawAction.setIcon(QIcon(r.ICON_DRAW));
        self.drawAction.setToolTip("Start drawing on the canvas");
        self.drawAction.setStatusTip("Start drawing on the canvas");
        self.drawAction.triggered.connect(self.startDrawing)
        self.drawAction.setShortcut("Ctrl+D");
        # View
        self.viewTrainingImagesAction = QAction("View Training Images", self);
        self.viewTestingImagesAction = QAction("View Testing Imaged", self);

        # Note: Add actions to context menus for drawing canvas
        # https://realpython.com/python-menus-toolbars/#creating-context-menus-through-context-menu-policy


    def initMenuBar(self):
        self.menubar = self.menuBar();

        self.fileMenu = self.menubar.addMenu("&File");
        self.fileMenu.addAction(self.trainModel);
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
        self.popup = peripheralUI.PopupBox("Help", r.ICON_HELP);
        self.popup.assignText("<b>Icon credit to:</b>");
        self.popup.assignText('Icons made by <a href="https://www.freepik.com">Freepik</a> from <a href="https://www.flaticon.com/">www.flaticon.com</a>.');
        self.popup.assignText('Icons made by <a href="https://iconmonstr.com">iconmonstr</a>.');

    def modelTraining(self):
        dlg = peripheralUI.createModelDialog()
        dlg.exec_()

    def startDrawing(self):
        #Central Widget
        main_content = c.AppMainContent();
        self.setCentralWidget(main_content);


    def exitApp(self):
        confirm = QMessageBox.question(self, "Warning", "Are you sure you want to quit?",
                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No);
        if (confirm == QMessageBox.Yes):
          self.close();



# If this module is run as main, execute the below:
if __name__ == '__main__':
    app = QApplication(sys.argv)
    appwin = AppMainWindow()
    sys.exit(app.exec_())