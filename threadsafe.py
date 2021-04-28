#====== ModelManager.py ======#
# HELPER

# Create a flexible QRunnable child class that can be used to thread
# safe nearly any process.
#
# Code adapted from:
# https://www.mfitzp.com/tutorials/multithreading-pyqt-applications-qthreadpool/#improved-qrunnables

import traceback, sys
from PyQt5.QtCore import *;


# Worker
# Thread-ready container for any function.

class Worker(QRunnable):
    '''
    Worker thread
    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.
    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn(*self.args, **self.kwargs)
