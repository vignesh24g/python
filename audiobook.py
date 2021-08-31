import pyttsx3
import PyPDF2
import tkinter as tk



root=tk.Tk()
root.title("AUDIBLE APPLICATION")
root.geometry("600x400")

x1=tk.StringVar()

def submit():
    name=x1.get()
    book=open(name+".pdf",'rb')
    
    pdf=PyPDF2.PdfFileReader(book)
    pages=pdf.numPages

    speech=pyttsx3.init()
    voices = speech.getProperty('voices')
    speech.setProperty('voice', voices[2].id)

    for n in range(0,pages):
        page=pdf.getPage(n)
        txt=page.extractText()
        speech.say(txt)
        speech.runAndWait()
    

    x1.set("")
    
    
name_l = tk.Label(root, text = ' enter a book from desktop', font=('calibre',10, 'bold'))
name_label = tk.Label(root, text = ' enter the bookname', font=('calibre',10, 'bold'))

name_entry=tk.Entry(root,textvariable = x1,font=('calibre',10, 'normal'))

btn=tk.Button(root,text= 'submit' ,command=submit)

name_label.grid(row=1,column=0)
name_entry.grid(row=1,column=1)
name_l.grid(row=0,column=1)

btn.grid(row=2,column=1)
  
    
'''book=open(name,'rb')
    
pdf=PyPDF2.PdfFileReader(book)
pages=pdf.numPages

speech=pyttsx3.init()
voices = speech.getProperty('voices')
speech.setProperty('voice', voices[2].id)

for n in range(0,pages):
    page=pdf.getPage(n)
    txt=page.extractText()
    speech.say(txt)
    speech.runAndWait()'''

root.mainloop()
