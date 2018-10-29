import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import NumReco
def putimg():
    file_name = filedialog.askopenfilename(title='打开一张图片', filetypes=[('All Files', '*')])
    print(file_name)
    if file_name!="":
        img1 = ImageTk.PhotoImage(file=(file_name))
        Image.config(image=img1)
        Image.image = img1 #keep a reference
        var.set(file_name)
        print(img1)
        if var.get() != "":
            result = NumReco.recognize(var.get())
            print("acc", result)
            result_label.config(text=('result : ' + str(result)))
        else:
            print(var.get())
    else:
        return

root = tk.Tk()
root.title('NumberRec V1.0')
main_fram =  ttk.LabelFrame(root, text=" Monty Python ")
main_fram.pack(padx=10, pady=10,expand =True,fill="both")

ttk.Label(main_fram, text="Chooes a Picture").grid(column=0, row=1)

img_frame = tk.LabelFrame(main_fram,background='#D3D3D3')
img_frame.grid(column=0, row=3, padx=10, pady=10,columnspan=5,rowspan=5)
Image = ttk.Label(img_frame)
Image.grid(column=0, row=3, padx=10, pady=10)
img_frame.columnconfigure (0,weight=10)
img_frame.rowconfigure (0, weight=10)

var = tk.StringVar()
path = ttk.Entry(main_fram,text=var )
path.grid(column=0, row=2,padx=10, pady=10)
open_img = ttk.Button(main_fram,text='打开图片',command= putimg)
open_img.grid(column=1, row=2, padx=10, pady=10)

console =  ttk.LabelFrame(root, text="Console ")
console.pack(padx=10, pady=10,expand =True,fill="both")

result = None
result_label = ttk.Label(console,text=('result : '+str(result)))
result_label.grid(column=0, row=3, columnspan=5,rowspan=5,padx=10, pady=10)



root.mainloop()