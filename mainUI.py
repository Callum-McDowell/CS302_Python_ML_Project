# Callum - 29th March 2021

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
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget, QMainWindow;
from PyQt5.QtWidgets import QMenuBar, QToolBar, QStatusBar, QAction;
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QInputDialog;
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
        WINDOW_SIZE_X = 800;
        WINDOW_SIZE_Y = 600;
        WINDOW_TITLE = "CNN Handwriting Recogniser";
        # Code
        self.setWindowTitle(WINDOW_TITLE);
        self.resize(WINDOW_SIZE_X, WINDOW_SIZE_Y);
        self.setWindowIcon(QIcon(r.ICON_WINDOW));
        self.centreWindow();
        # Core Components
        self.initActions();
        self.menubar = self.initMenuBar();
        self.toolbar = self.initMainToolBar();
        self.statusbar = self.initStatusBar();
        # Central Widget
        main_content = c.AppMainContent();
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
        self.exitAction.setIcon(QIcon(r.ICON_EXIT));
        self.exitAction.setShortcut("Ctrl+E");
        self.exitAction.triggered.connect(self.exitApp);
        # Open
        self.openAction = QAction("&Open", self);
        self.openAction.setIcon(QIcon(r.ICON_OPEN));
        self.openAction.setToolTip("Select new training images");
        self.openAction.setStatusTip("Select new training images");
        self.openAction.setShortcut("Ctrl+O");
        self.openAction.triggered.connect(self.importDataset);
        # Help 
        self.helpAction = QAction("Help", self);
        self.helpAction.setIcon(QIcon(r.ICON_HELP));
        self.helpAction.triggered.connect(self.helpDialogue);
        # Draw
        self.drawAction = QAction("&Draw", self);
        self.drawAction.setIcon(QIcon(r.ICON_DRAW));
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
        self.popup = peripheralUI.PopupBox("Help", r.ICON_HELP);
        self.popup.assignText("<b>Icon credit to:</b>");
        self.popup.assignText('Icons made by <a href="https://www.freepik.com">Freepik</a> from <a href="https://www.flaticon.com/">www.flaticon.com</a>.');
        self.popup.assignText('Icons made by <a href="https://iconmonstr.com">iconmonstr</a>.');

    def importDataset(self):
        file_dir_array = QFileDialog.getOpenFileNames(self, "Open file", "./");
        # Send this data to validation (are pics correct format? accessible?)
        # Generate error warnings for invalid entries
        # Then save valid dirs to TensorFlow NN dataset reference

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