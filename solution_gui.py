import os.path
import tkinter as tk
from tkinter import ttk, Menu
from tkinter import filedialog
from tkinter.constants import DISABLED, SUNKEN, E, BOTH, NORMAL
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2

from PIL import ImageTk, Image, ImageOps
import torch
import os


class CADGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Load model
        self.model = torch.hub.load(os.getcwd(),
                                    'custom',
                                    path='best.pt',
                                    source='local',
                                    force_reload=True)

        # Variables
        self.detection_images_file_list = []
        self.mammogram_names_list = []  # List of mammogram names
        self.original_mammograms_list = []
        self.detection_mammogram_list = []
        self.current_position = 0
        self.show_malignant = tk.BooleanVar()
        self.show_malignant.set(True)
        self.show_benign = tk.BooleanVar()
        self.show_benign.set(True)
        self.show_normal = tk.BooleanVar()
        self.show_normal.set(True)
        self.all_frames_df = pd.DataFrame(columns=['Filename', 'Tumour ID', 'Xmin', 'Ymin', 'Xmax', 'Ymax',
                                                             'Confidence', 'Classification'])

        # Root window of GUI
        self.title('CAD Project')
        # self.geometry('1000x1000')
        self.configure(background='black')

        # Create frames which will store GUI elements
        self.original_mammogram_frame = tk.Frame(self, width=400, height=600)
        self.detection_mammogram_frame = tk.Frame(self, width=400, height=600)
        self.buttons_frame = tk.Frame(self, width=800,
                                      height=50)  # Frame for buttons to move to the next/previous image
        self.status_frame = tk.Frame(self, width=1000, height=20)  # Frame for current position as status
        self.mammogram_name_frame = tk.Frame(self, width=800, height=50)  # Frame for the current mammogram's filename
        self.filter_frame = tk.Frame(self, width=200, height=800)  # Frame for the mammogram filtering box
        self.menu_bar = Menu(self)

        # Allocate the frames a grid position on the GUI screen
        self.config(menu=self.menu_bar)
        self.mammogram_name_frame.grid(row=0, column=1, columnspan=2)  # Mammogram name positioned above mammograms
        self.filter_frame.grid(row=1, column=0)  # Filter box on the left of mammograms
        self.original_mammogram_frame.grid(row=1, column=1)  # Original mammogram appears on the left
        self.detection_mammogram_frame.grid(row=1, column=2)  # Mammogram with detections appears on the right
        self.buttons_frame.grid(row=2, column=1, columnspan=2)  # Buttons appear below mammograms
        self.status_frame.grid(row=3, column=0, columnspan=3)  # Status appears at the bottom of gui

        # File menu
        self.file_menu = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Exit", command=self.quit)
        self.file_menu.add_command(label="Load", command=self.load_mammograms)

        # Export menu
        self.export_menu = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(label="Export", menu=self.export_menu)
        self.export_menu.add_command(label="All", command=self.export_all)
        self.export_menu.add_command(label="Current", command=self.export_current)

        # Edit menu
        self.edit_menu = Menu(self.menu_bar, tearoff=False)
        self.menu_bar.add_cascade(label="Remove", menu=self.edit_menu)
        self.edit_menu.add_command(label="All", command=self.remove_all)
        self.edit_menu.add_command(label="Current", command=self.remove_current)

        # Mammogram name
        self.mammogram_name_text = tk.Label(self.mammogram_name_frame, text="", anchor="center")
        self.mammogram_name_text.pack(fill=BOTH)

        # Titles
        self.original_mammogram_title = tk.Label(self.original_mammogram_frame, text="Selected Mammogram")
        self.original_mammogram_title.grid(row=0)
        self.detection_mammogram_title = tk.Label(self.detection_mammogram_frame,
                                                  text="Tumour Detections")
        self.detection_mammogram_title.grid(row=0)
        self.filter_titles = tk.Label(self.filter_frame, text="Filter results")
        self.filter_titles.grid(row=0)

        # Mammogram images
        self.original_mammogram_image = tk.Label(self.original_mammogram_frame, text="No Mammogram's Loaded")
        self.original_mammogram_image.grid(row=1)
        self.detection_mammogram_image = tk.Label(self.detection_mammogram_frame, text="No Detections Made")
        self.detection_mammogram_image.grid(row=1)

        # Movement Buttons
        self.backward_button = tk.Button(self.buttons_frame, text="<<", state=DISABLED,
                                         command=self.previous_mammogram, width=20)
        self.backward_button.grid(row=0, column=0, sticky='e')
        self.forward_button = tk.Button(self.buttons_frame, text=">>", state=DISABLED,
                                        command=self.next_mammogram, width=20)
        self.forward_button.grid(row=0, column=1, sticky='w')

        # Filter Buttons
        self.malignant_checkbutton = tk.Checkbutton(self.filter_frame, text="Malignant",
                                                    onvalue=1, offvalue=0,
                                                    height=2, width=10, variable=self.show_malignant)
        self.malignant_checkbutton.grid(row=1)
        self.benign_checkbutton = tk.Checkbutton(self.filter_frame, text="Benign",
                                                 onvalue=1, offvalue=0,
                                                 height=2, width=10, variable=self.show_benign)
        self.benign_checkbutton.grid(row=2)
        self.normal_checkbutton = tk.Checkbutton(self.filter_frame, text="Normal",
                                                 onvalue=1, offvalue=0,
                                                 height=2, width=10, variable=self.show_normal)
        self.normal_checkbutton.grid(row=3)

        # Status bar
        self.status_bar = tk.Label(self.status_frame, relief=SUNKEN, anchor=E)
        self.status_bar.grid()

        for frame in [self.original_mammogram_frame, self.detection_mammogram_frame,
                      self.status_frame, self.mammogram_name_frame]:
            frame.grid(sticky='nswe')
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            frame.grid_propagate(0)
            frame.config(bg='black')

        for widget in [self.mammogram_name_text, self.original_mammogram_title, self.detection_mammogram_title,
                       self.original_mammogram_image, self.detection_mammogram_image, self.status_bar]:
            widget.grid(sticky='nwe')
            widget.config(background='black', foreground="White")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def load_mammograms(self):

        filepaths = filedialog.askopenfilenames(title="Select Scans")
        raw_images_list = []
        preprocessed_image_list = []

        for file in filepaths:
            _, filename = os.path.split(file)
            self.mammogram_names_list.append(filename)
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            raw_images_list.append(image)

        for image in raw_images_list:
            # Crop borders
            row_count, column_count = image.shape
            x1_crop = int(column_count * 0.01)
            x2_crop = int(column_count * (1 - 0.01))
            y1_crop = int(row_count * 0.04)
            y2_crop = int(row_count * (1 - 0.04))

            preprocessed_image = image[y1_crop:y2_crop, x1_crop:x2_crop]

            blur = cv2.GaussianBlur(preprocessed_image, (5, 5), 0)

            # Binary threshold on the gray scan
            _, mammogram_threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(23, 23))
            mammogram_threshold = cv2.morphologyEx(mammogram_threshold, cv2.MORPH_OPEN, kernel)
            mammogram_threshold = cv2.morphologyEx(mammogram_threshold, cv2.MORPH_DILATE, kernel)

            # Find the largest contour in threshold mammogram
            all_contours, _ = cv2.findContours(mammogram_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            largest_contour = max(all_contours, key=cv2.contourArea)

            # Create binary mark to remove artifacts
            mammogram_mask = np.zeros(mammogram_threshold.shape, np.uint8)

            cv2.drawContours(mammogram_mask, [largest_contour], -1, 255, cv2.FILLED)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Apply mask to original mammogram
            preprocessed_image = preprocessed_image[y:y + h, x:x + w]
            cropped_mammogram_mask = mammogram_mask[y:y + h, x:x + w]

            # Set non breast elements to black
            preprocessed_image[cropped_mammogram_mask == 0] = 0

            # Enhance scans
            clahe_create = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            preprocessed_image = clahe_create.apply(preprocessed_image)
            preprocessed_image_list.append(preprocessed_image)

        model_results = self.model(preprocessed_image_list)
        model_results_df = model_results.pandas().xyxy

        import_frames_list = []

        for scan in model_results_df:
            scan = scan.round(2)
            import_frames_list.append(scan)

        processed_import_frames_df = pd.DataFrame(columns=['Filename', 'Tumour ID', 'Xmin', 'Ymin', 'Xmax', 'Ymax',
                                                             'Confidence', 'Classification'])

        for name, data in zip(self.mammogram_names_list, import_frames_list):
            if data.empty:
                information = [name, '', '', '', '', '', '', 'NORMAL']
                processed_import_frames_df.loc[len(processed_import_frames_df)] = information

            else:
                for index, row in data.iterrows():
                    information = [name, (index + 1), row['xmin'], row['ymin'], row['xmax'], row['ymax'],
                                   "{:.0%}".format(row['confidence']), row['name']]
                    processed_import_frames_df.loc[len(processed_import_frames_df)] = information

        self.all_frames_df = processed_import_frames_df

        for scan in raw_images_list:
            mammogram = Image.fromarray(scan)
            mammogram = ImageOps.contain(mammogram, (400, 580))
            mammogram = ImageTk.PhotoImage(mammogram)
            self.original_mammograms_list.append(mammogram)

        processed_scans = model_results.render()

        for scan in processed_scans:
            mammogram = Image.fromarray(scan)
            self.detection_images_file_list.append(mammogram)

            mammogram2 = ImageOps.contain(mammogram, (400, 580))
            mammogram2 = ImageTk.PhotoImage(mammogram2)

            self.detection_mammogram_list.append(mammogram2)

        self.original_mammogram_image.config(image=self.original_mammograms_list[0])
        self.detection_mammogram_image.config(image=self.detection_mammogram_list[0])
        self.mammogram_name_text.config(text=self.mammogram_names_list[0])

        if len(self.original_mammograms_list) > 0:
            self.forward_button.config(state=NORMAL)

        self.status_bar.config(text=f"Scan {self.current_position + 1} of {len(self.original_mammograms_list)}")

    def next_mammogram(self):
        self.check_filter(1)
        self.update_gui()
        self.backward_button.config(state=NORMAL)
        if self.current_position + 1 == len(self.original_mammograms_list):
            self.forward_button.config(state=DISABLED)

    def previous_mammogram(self):
        self.check_filter(-1)
        self.update_gui()
        self.forward_button.config(state=NORMAL)
        if self.current_position == 0:
            self.backward_button.config(state=DISABLED)

    def check_filter(self, offset):
        filtered_values = []
        if self.show_malignant.get() == 0:
            filtered_values.append("MALIGNANT")

        if self.show_benign.get() == 0:
            filtered_values.append("BENIGN")

        if self.show_normal.get() == 0:
            filtered_values.append("NORMAL")

        proposed = self.current_position + offset

        temp = self.all_frames_df.groupby(['Filename']).agg(tuple).applymap(list).reset_index()

        while any(item in filtered_values for item in temp.iloc[proposed].Classification):
            proposed = proposed + offset
            if proposed < 0 or proposed > len(temp.index):
                proposed = self.current_position
                return

        self.current_position = proposed

    def print_table(self, pdf, df):

        table_cell_width = 23.5
        table_cell_height = 6
        pdf.set_font('Arial', 'B', 10)

        cols = df.columns
        for col in cols:
            pdf.cell(table_cell_width, table_cell_height, col, align='C', border=1)

        pdf.ln(table_cell_height)

        pdf.set_font('Arial', '', 10)

        for index, row in df.iterrows():
            for col in cols:
                value = str(getattr(row, col))
                pdf.cell(table_cell_width, table_cell_height, value, align='C', border=1)

            pdf.ln(table_cell_height)

    def export_all(self):

        directory = filedialog.asksaveasfilename(defaultextension='.pdf', filetypes=[("pdf files", '*.pdf')],
                                                 title="Choose filename")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)

        pdf.cell(40, 10, 'Mammogram Mass Detection Report')
        pdf.ln(10)

        pdf.set_font('Arial', '', 12)
        now = datetime.now()
        date = now.strftime("%B %d, %Y")
        time = now.strftime("%H:%M")
        text = f'x Mammograms were processed on {date} at {time} producing the following results:'
        pdf.cell(40, 10, text)
        pdf.ln(20)

        type_count = {'Malignant': 0, 'Benign': 0, 'Normal': 0}

        temp = self.all_frames_df.groupby(['Filename']).agg(tuple).applymap(list).reset_index()
        for index, row in temp.iterrows():
            if 'MALIGNANT' in row.Classification:
                type_count['Malignant'] = type_count['Malignant'] + 1
            elif 'BENIGN' in row.Classification:
                type_count['Benign'] = type_count['Benign'] + 1
            else:
                type_count['Normal'] = type_count['Normal'] + 1

        keys = type_count.keys()
        values = type_count.values()

        plt.bar(keys, values, color=['red', 'blue', 'green'])
        ticks = range(math.floor(min(values)), math.ceil(max(values)) + 1)
        plt.yticks(ticks)
        plt.xlabel("Classification")
        plt.ylabel("Total Number of Cases")

        plt.savefig('temp.png')
        pdf.image('temp.png', 30, 30, 117.5, 100)
        pdf.ln(100)
        self.print_table(pdf, self.all_frames_df)

        # 3. Output the PDF file
        pdf.output(directory, 'F')

    def export_current(self):
        directory = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("image files", '.png')],
                                                 title="Choose image name")
        self.detection_images_file_list[self.current_position].save(directory)

    def remove_current(self):

        filename = self.mammogram_names_list[self.current_position]

        lists = [self.detection_images_file_list,  self.mammogram_names_list,  self.original_mammograms_list,
                 self.detection_mammogram_list]

        for data_list in lists:
            data_list.pop(self.current_position)

        self.all_frames_df = self.all_frames_df[self.all_frames_df.Filename != filename]
        self.update_gui()

    def remove_all(self):
        lists = [self.detection_images_file_list, self.mammogram_names_list, self.original_mammograms_list,
                 self.detection_mammogram_list]
        for data_list in lists:
            data_list.clear()
        self.all_frames_df = pd.DataFrame(columns=['Filename', 'Tumour ID', 'Xmin', 'Ymin', 'Xmax', 'Ymax',
                                                   'Confidence', 'Classification'])

        self.original_mammogram_image.config(image="", text="No Mammogram's Loaded")
        self.detection_mammogram_image.config(image="", text="No Detections Made")
        self.mammogram_name_text.config(text="")
        self.status_bar.config(text="")
        self.forward_button.config(state=DISABLED)
        self.backward_button.config(state=DISABLED)
        self.current_position = 0

    def update_gui(self):
        self.original_mammogram_image.config(image=self.original_mammograms_list[self.current_position])
        self.detection_mammogram_image.config(image=self.detection_mammogram_list[self.current_position])
        self.mammogram_name_text.config(text=self.mammogram_names_list[self.current_position])
        self.status_bar.config(text=f"Scan{self.current_position + 1} of {len(self.mammogram_names_list)}")


if __name__ == "__main__":
    GUI = CADGUI()
    GUI.mainloop()
