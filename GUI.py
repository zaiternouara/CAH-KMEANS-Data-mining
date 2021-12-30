import tkinter as tk
from pandas_ods_reader import read_ods

import partie2


'''Fonction pour la liste des wilaya d'alger'''
def populate(frame):
    '''Put in some data'''
    path = "C:/Users/Z T R/Desktop/TP2/TRYyy/Algeria-Covid19.ods"
    # load a sheet based sheet number
    df = read_ods(path, 1).values.tolist()

    for row in range(len(df)):
        tk.Label(frame, text=df[row][0],font=("Helvetica", 16)).grid(row=row, column=0,)
        tk.Label(frame, text=df[row][1],font=("Helvetica", 16)).grid(row=row, column=1)
        tk.Label(frame, text=df[row][2],font=("Helvetica", 16)).grid(row=row, column=2)
        tk.Label(frame, text=df[row][3],font=("Helvetica", 16)).grid(row=row, column=3)


def onFrameConfigure(canvas):
    '''Reset the scroll region to encompass the inner frame'''
    canvas.configure(scrollregion=canvas.bbox("all"))


'''for call the K-means file and CAH'''
def callback():
    y = partie2.main()
    label = tk.Label(newwindow, text= str(y))
    label.pack()

    
'''Window size and properties'''
root = tk.Tk()
root.geometry('700x400')
root.resizable(0,0)
root.title("GUI")
canvas = tk.Canvas(root, borderwidth=0)
frame = tk.Frame(canvas)
vsb = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=vsb.set)
vsb.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)
canvas.create_window((4,4), window=frame, anchor="nw")

frame.bind("<Configure>", lambda event, canvas=canvas: onFrameConfigure(canvas))
populate(frame)

button = tk.Button(root, text='Display all the information', 
                   width=25, command=callback)
'''Information on data visualization and methods + CAH + k-means'''
button.pack(side=tk.LEFT)
root.mainloop()
