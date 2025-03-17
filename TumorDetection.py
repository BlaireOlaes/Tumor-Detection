import os
import sys
import cv2
import multiprocessing

import tkinter
import ultralytics
import reportlab
import PIL

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer


def resource_path(relative_path):
    """ Get absolute path to resource, works for development and PyInstaller """
    try:
        # PyInstaller creates a temporary folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Path to the YAML file
#yaml_path = resource_path ('.venv/Lib/site-packages/ultralytics/cfg/default.yaml')

#print("Looking for default.yaml at:", yaml_path)

# Verify the file's existence
#try:
 #   with open(yaml_path, 'r') as f:
 #       print("Successfully opened default.yaml")
#except FileNotFoundError:
 #   print("Failed to find default.yaml at:", yaml_path)
#except Exception as e:
 #   print("An error occurred:", str(e))

if __name__ == "__main__":
    multiprocessing.freeze_support()

im0 = None
detected_classes = []

def load_image():
    global im0, imgtk, image_display_width, image_display_height
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if filepath:
        im0 = cv2.imread(filepath)
        im0 = cv2.resize(im0, (image_display_width, image_display_height))
        predict_and_display()

def predict_and_display():
    global im0, imgtk, detected_classes
    results = model.predict(im0)
    annotator = Annotator(im0, line_width=2)
    detected_classes = []

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()  # Get the confidence scores
        masks = results[0].masks.xy
        for mask, cls, conf in zip(masks, clss, confs):
            if conf >= 0.6:  # Only consider detections with confidence score >= 70%
                det_label = f"{names[int(cls)]}: {conf*100:.2f}%"  # Include the confidence level in the label
                annotator.seg_bbox(mask=mask, mask_color=colors(int(cls), True), det_label=det_label)
                if names[int(cls)] not in detected_classes:
                    detected_classes.append((names[int(cls)], conf))  # Include the confidence score in the detected_classes list

    cv2_im = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(cv2_im)
    imgtk = ImageTk.PhotoImage(image=img_pil)

    canvas_image.create_image(0, 0, anchor=NW, image=imgtk)
    update_description()

def update_description():
    global description_label
    descriptions = {
        "tumor_good_chance": """
Common Types of Small-Sized Brain Tumors:\n
Pituitary Adenomas: These benign tumors develop in the pituitary gland, located at the base of the brain. They can affect hormone production and cause visual disturbances.
Acoustic Neuromas: Also known as vestibular schwannomas, these benign tumors grow on the nerve between the inner ear and the brainstem. They can cause hearing loss and balance problems.
Meningiomas: While they can vary in size, smaller meningiomas are often benign and grow slowly. They form on the membranes covering the brain (meninges).


Treatment Options for Small-Sized Brain Tumors:\n
Surgery: For many smaller brain tumors, surgery is the primary treatment option. The goal is to remove the entire tumor while minimizing damage to surrounding brain tissue.
Radiosurgery: A non-invasive procedure that uses focused radiation beams to target and destroy the tumor. It is suitable for small, well-defined tumors.
Radiation Therapy: Delivered over several weeks, it can be used to shrink the tumor or prevent its growth, especially if surgery is not an option.
Medication: For certain types of tumors, medications can be used to manage symptoms or slow tumor growth.
Observation and Monitoring: In some cases, especially with very small or slow-growing tumors, active surveillance may be preferred to watch for changes over time.
""",
        "tumor_moderate_chance": """
Common Types of Moderate-Sized Brain Tumors:\n
Astrocytomas: These are a type of glioma that can vary in aggressiveness. Some astrocytomas may grow slowly over time, making them less immediately life-threatening but still requiring monitoring and treatment.
Oligodendrogliomas: These tumors are often slower-growing and can sometimes be completely removed surgically. They are more likely to respond to certain types of chemotherapy.
Medulloblastomas: Although typically found in children, medulloblastomas can occur in adults and are considered high-grade tumors. Their moderate size might allow for more treatment options, including surgery and radiation therapy.


Treatment Options for Moderate-Sized Brain Tumors:\n
Surgery: If the tumor is operable, surgery is often the first line of treatment to remove as much of the tumor as possible. This can help reduce symptoms and improve prognosis.
Radiation Therapy: May be used after surgery to kill any remaining cancer cells and to control tumor growth. It can be delivered externally or internally.
Chemotherapy: Administered to kill cancer cells throughout the body, especially useful for certain types of brain tumors that are known to spread.
Targeted Therapy and Immunotherapy: These newer treatments aim to target specific genetic abnormalities in the tumor cells or boost the body's immune response against the cancer.
Palliative Care: Focuses on managing symptoms and improving quality of life, especially if the tumor cannot be cured.
""",
        "tumor_less_chance": """
Common Types of Large Brain Tumors:\n
Gliomas: These are the most common type of primary brain tumor, arising from glial cells (supportive tissue in the brain). Glioblastoma multiforme is a particularly aggressive form of glioma.
Meningiomas: Benign tumors that arise from the meninges, the protective layers surrounding the brain. While usually slow-growing, they can become quite large.
Metastatic Brain Tumors: These are cancerous tumors that originate elsewhere in the body and spread to the brain via the bloodstream. They can grow quickly and are often found in multiple locations within the brain.


Treatment Options for Large Brain Tumors:\n
Surgery: The goal is to remove as much of the tumor as safely possible without damaging surrounding brain tissue. For some large tumors, especially metastatic ones, surgery may not be feasible or effective.
Radiation Therapy: Used to shrink the tumor and relieve symptoms. It can be delivered externally (external beam radiation therapy) or internally (brachytherapy).
Chemotherapy: Administered orally or intravenously to kill rapidly dividing cancer cells. It is often used after surgery or alongside radiation therapy.
"""
    }

    description_texts = [f"{cls}: {conf * 100:.2f}% confidence\n{descriptions[cls]}" for cls, conf in detected_classes
                         if cls in descriptions]
    if description_texts:
        description_text.set("\n\n".join(description_texts))
    else:
        description_text.set("No tumors detected.")

    # Adjust font size based on the number of detected objects
    if len(detected_classes) > 1:
        description_label.config(font=("Helvetica", 6))  # Smaller font size
    else:
        description_label.config(font=("Helvetica", 9))  # Default font size
def extract_as_pdf():
    if im0 is None:
        messagebox.showwarning("No Image", "Please insert an image before extracting as PDF.")
        return

    filepath = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if filepath:
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add detailed description
        description_title = Paragraph("SCAN RESULT", styles['Title'])
        story.append(description_title)
        story.append(Spacer(1, 0.2 * inch))

        descriptions = {
            "tumor_good_chance": "Status: Good Chance<br/>Details: Good chance tumors are detected as smaller sizes of brain tumors with less immediate life-threatening potential. Despite their size, these tumors can still significantly disrupt a person’s life due to their location and the specific symptoms they produce. The impact of a smaller brain tumor largely depends on its precise location, the type of tumor, and the individual’s overall health status.",

            "tumor_moderate_chance": "Status: Moderate Chance<br/>Details: Moderate chance tumors are moderate in size. Although they may not exhibit all the severe symptoms associated with larger tumors, they still pose significant challenges to a person’s health and well-being. The prognosis for patients with moderate-sized brain tumors is generally better than for those with larger, more aggressive tumors. Early detection and a multidisciplinary approach to treatment are crucial for managing moderate-sized brain tumors effectively.",

            "tumor_less_chance": "Status: Less Chance<br/>Details: Less chance tumors are larger in size and potentially cancerous, which may necessitate medical interventions such as surgical procedures. Malignant brain tumors, especially larger and aggressive ones, generally have a poorer prognosis compared to benign tumors. However, advances in diagnostic techniques and treatment modalities have improved survival rates and quality of life for many patients. Treatment options for brain tumors depend on various factors, including the type of tumor, its size, and its location within the brain. Common treatments include surgery and radiation therapy."
        }

        description_texts = []

        for cls, conf in detected_classes:
            if cls in descriptions:
                status_and_details = descriptions[cls].split("<br/>")
                status = status_and_details[0]
                details = status_and_details[1]

                story.append(Paragraph(status, styles['BodyText']))
                story.append(Paragraph(details, styles['BodyText']))
                story.append(Spacer(1, 0.2 * inch))

        if description_texts:
            for text in description_texts:
                story.append(Paragraph(text, styles['BodyText']))
                story.append(Spacer(1, 0.2 * inch))

        # Add image
        cv2_im = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(cv2_im)
        temp_image_path = resource_path("temp_image.jpg")
        img_pil.save(temp_image_path)
        story.append(PDFImage(temp_image_path, width=5 * inch, height=4 * inch))

        doc.build(story)
        messagebox.showinfo("Success", "PDF has been successfully created!")
def close_program():
    if messagebox.askokcancel("Quit", "Are you sure you want to close the program?"):
        root.destroy()

def exit_full_screen(event=None):
    root.attributes('-fullscreen', False)

# Load a model
model = YOLO(resource_path("best.pt"))

names = model.model.names

# Initialize Tkinter window
root = Tk()
root.title("Image Segmentation GUI")
root.attributes('-fullscreen', True)

# Bind the Escape key to exit full-screen mode
root.bind("<Escape>", exit_full_screen)

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set image display dimensions
image_display_width = int(screen_width * 0.65)
image_display_height = screen_height

# Set up the layout
frame_left = Frame(root, width=image_display_width, height=image_display_height)
frame_right = Frame(root, width=int(screen_width * 0.35), height=screen_height)

frame_left.pack(side=LEFT, padx=10, pady=10, fill=BOTH, expand=True)
frame_right.pack(side=RIGHT, padx=10, pady=10, fill=BOTH, expand=True)

frame_description = Frame(frame_right, width=int(screen_width * 0.35), height=screen_height)
frame_description.pack(side=TOP, padx=10, pady=10, fill=X)

frame_description_buttons = Frame(frame_right, width=int(screen_width * 0.35), height=screen_height)
frame_description_buttons.pack(side=BOTTOM, padx=10, pady=10, fill=X)


# Create a new frame for the buttons
frame_buttons = Frame(frame_right, width=int(screen_width * 0.35), height=50)
frame_buttons.pack(side=BOTTOM, padx=10, pady=10, fill=X)



# Canvas for displaying image
canvas_image = Canvas(frame_left, width=image_display_width, height=image_display_height)
canvas_image.pack(fill=BOTH, expand=True)

# Description text
description_text = StringVar()
description_label = Label(frame_description, textvariable=description_text, wraplength=int(screen_width * 0.2), justify=LEFT)
description_label.pack(pady=5)

# Create a new frame for the buttons
frame_buttons = Frame(frame_description_buttons, width=int(screen_width * 0.35), height=50)
frame_buttons.pack(side=BOTTOM, padx=10, pady=10, fill=X)

# Buttons
button_load = Button(frame_buttons, text="Insert MRI brain scan", command=load_image, width=20, height=2, bg="lightblue")
button_load.pack(side=LEFT, padx=(0, 10))

button_pdf = Button(frame_buttons, text="Extract data as PDF", command=extract_as_pdf, width=20, height=2, bg="lightgreen")
button_pdf.pack(side=LEFT, padx=(0, 10))

button_close = Button(frame_buttons, text="Close Application", command=close_program, width=20, height=2, bg="red")
button_close.pack(side=LEFT, padx=(0, 10))


# Run the Tkinter loop
root.mainloop()
