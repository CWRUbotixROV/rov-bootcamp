from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk

from os import walk
from os.path import join, relpath

import shutil

from settings import SHAPES

SIZE = 500

class App:
    def __init__(self):
        self.window = Tk()

        self.load_dir = filedialog.askdirectory()

        (_, _, filenames) = next(walk(self.load_dir))
        self.filenames = filenames
        

        self.label = Label(self.window, width=SIZE, height=SIZE)
        self.label.pack()

        button_frame = Frame(self.window)
        button_frame.pack()

        for shape in SHAPES:
            button = Button(button_frame, text=shape, command=lambda shape=shape : self.copy_image(shape))
            button.pack(side=LEFT, padx=10)

        self.update_image()

        self.window.mainloop()
    
    def update_image(self):
        img = Image.open(self.cur_path())
        width_ratio = 500 / img.width
        height_ratio = 500 /img.height
        ratio = min(width_ratio, height_ratio)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        self.image = ImageTk.PhotoImage(img)
        self.label.configure(image=self.image)

    def copy_image(self, shape):
        new_path = join('training', shape, self.filenames[0])
        shutil.copyfile(self.cur_path(), new_path)

        if len(self.filenames) > 1:
            self.filenames.pop(0)
            self.update_image()
        else:
            self.window.destroy()
    
    def cur_path(self):
        return join(self.load_dir, self.filenames[0])

App()