import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
import auth_token
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 
import gc;
import os
from koila import lazy
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:100'
torch.cuda.max_memory_allocated(device='cuda')
app = tk.Tk()
app.geometry("532x632")
app.title("Coffee")
ctk.set_appearance_mode("dark")
prompt = ctk.CTkEntry(master = None, placeholder_text="CTkEntry", height=40,width=512,text_color="black", fg_color="white")
entry = ctk.CTkEntry( master = None,placeholder_text="CTkEntry",width=160,height=20,border_width=2,corner_radius=10)
entry.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
prompt.place(x=10, y=10)
lmain = ctk.CTkLabel(master = None, height=256, width=256)
lmain.place(x=10, y=110)
modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16",torch_dtype=torch.float16,use_auth_token= auth_token)
pipe.to(device)
gc.collect() 
torch.cuda.empty_cache() 
input = torch.randn(8, 28, 28)
label = torch.randn(0, 10, 8)
(input, label) =lazy(input, label, batch=0)
def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=1)["sample"][0]
        image.save('generatedimage.png')
        img = ImageTk.PhotoImage(image)
        lmain.configure(image=img)
        trigger = ctk.CTkButton(master = None, height=40, width=120, font=("Arial", 20),
                                text_color="white", fg_color="blue", command=generate)
        trigger.configure(text="Generate")
        trigger.place(x=206, y=60)
        app.mainloop()