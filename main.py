## Libraries
import pathlib
import shutil
import re
import cv2
import streamlit as st
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from pdf2image import convert_from_path
import pandas as pd
import numpy as np

################################################
##Configuration
ENDPOINT = "https://newocrrecognizer.cognitiveservices.azure.com/"
KEY = "9aLuWK9CcC8fNP0eRjOnkqd1Hr1jvShTnFFzcJDPDpq07aaYY9cJJQQJ99BDACYeBjFXJ3w3AAAFACOGrRYy"
image_analysis_client = ImageAnalysisClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))


########################################################################################################################
##Functions

def load_pdf(pdf):
    """Returns a string and an int
    Converts PDF pages to images and saves them in the 'pages' folder."""
    if pdf is not None:
        # Get the file name
        file_name = pdf.name

        # Define the directory where you want to save the file
        save_directory = "pdfs"  # You can change this to any path

        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Create the full path for the new file
        file_path = os.path.join(save_directory, file_name)

        # Write the file to the specified directory
        try:
            with open(file_path, "wb") as f:
                f.write(pdf.getbuffer())
        except Exception as e:
            st.error(f"Error saving file: {e}")

    document = convert_from_path(file_path, poppler_path="poppler-24.08.0/Library/bin")

    if not os.path.exists("pages"):
        os.makedirs("pages")

    for i, page in enumerate(document):
        page.save(f'pages/page{i}.jpg', 'JPEG')
    return len(document) - 1


def remove_all_files_in_directory_pathlib(directory_name):
    project_dir = pathlib.Path(__file__).parent
    pages_dir = project_dir / directory_name
    if pages_dir.exists():  # Checks if directory exists first
        for item in pages_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def get_date(file):
    file_date = re.sub(".*/(.*)", r"\1", file.name)
    if not re.findall(r".*(\d{2}[A-Z]{3}\d{2}).*", file_date):
        if not re.findall(r".*(-\d{6}).*", file_date):
            file_date = np.nan
        else:
            file_date = re.findall(r".*(\d{6}).*", file_date)[0]
            file_date = file_date[0:2] + "-" + file_date[2:4] + "-" + file_date[4:6]
    else:
        file_date = re.findall(r".*(\d{2}[A-Z]{3}\d{2}).*", file_date)[0]
        months = {"ENE": "/01/", "FEB": "/02/", "MAR": "/03/", "ABR": "/04/", "MAY": "/05/", "JUN": "/06/",
                  "JUL": "/07/", "AGO": "/08/",
                  "SEP": "/09/", "OCT": "/10/", "NOV": "/11/", "DIC": "/12/"}
        for month in months.keys():
            file_date = re.sub(month, months[month], file_date)
        file_date = re.sub("/", "-", file_date)

    return file_date


def crop(fullpage, start_x_rat, start_y_rat, end_x_rat, end_y_rat):
    """ Returns an image
    This function crops a given image"""
    or_height, or_width = int(fullpage.shape[0]), int(fullpage.shape[1])
    start_x, start_y = int(or_height * start_x_rat), int(or_width * start_y_rat)
    end_x, end_y = int(or_height * end_x_rat), int(or_width * end_y_rat)
    cropped = fullpage[start_x:end_x, start_y:end_y]

    return cropped


###################################################################################################################
## Main Functions
def main_ball_function(pdf, provider):
    #try:
        global glob_df
        with st.spinner("Esperando ..."):
            number_of_images = load_pdf(pdf)

            diccionary = {"Fecha": [], "Codigo": [], "Orden de compra": [], "Cantidad": [], "Numero de Lote": [],
                          "Remision": [], "Proveedor": []}

            # for k in range(number_of_images):
            image_path = "pages/page0" + ".jpg"  # + str(k) + ".jpg"

            # Load image to analyze into a 'bytes' object
            with open(image_path, "rb") as f:
                image_data = f.read()

            result = image_analysis_client.analyze(
                image_data=image_data,
                visual_features=[VisualFeatures.READ]
            )

            if result.read is not None and result.read.blocks:
                for block in result.read.blocks:
                    if block.lines is not None:
                        for line in block.lines:
                            if re.match(r"B/L No\. (\d+)", line.text):
                                code = re.sub(r"B/L No\. ", "", line.text)
                                diccionary["Remision"].append(code)

                                page = cv2.imread(image_path)

                                ## Product Code
                                cropped = crop(page, 0.34, 0.07, 0.69, 0.26)
                                cv2.imwrite(image_path[:-4] + "_" + "code" + ".jpeg", cropped)

                                ## Buy Order
                                cropped = crop(page, 0.34, 0.18, 0.69, 0.36)
                                cv2.imwrite(image_path[:-4] + "_" + "order" + ".jpeg", cropped)

                                ## Quantity
                                cropped = crop(page, 0.34, 0.70, 0.69, 0.79)
                                cv2.imwrite(image_path[:-4] + "_" + "quantity" + ".jpeg", cropped)

                                ## Batch Number
                                cropped = crop(page, 0.34, 0.45, 0.69, 0.56)
                                cv2.imwrite(image_path[:-4] + "_" + "batch" + ".jpeg", cropped)

                                with open(image_path[:-4] + "_" + "batch" + ".jpeg", "rb") as f:
                                    image_data = f.read()
                                result = image_analysis_client.analyze(
                                    image_data=image_data,
                                    visual_features=[VisualFeatures.READ])

                                if result.read is not None and result.read.blocks:
                                    for block in result.read.blocks:
                                        if block.lines is not None:
                                            for line in block.lines:
                                                if re.match(r".*\d{10}.*", line.text):
                                                    code = re.sub(r".*(\d{10}).*", r"\1", line.text)
                                                    diccionary["Numero de Lote"].append(code)

                                with open(image_path[:-4] + "_" + "code" + ".jpeg", "rb") as f:
                                    image_data = f.read()
                                result = image_analysis_client.analyze(
                                    image_data=image_data,
                                    visual_features=[VisualFeatures.READ])

                                if result.read is not None and result.read.blocks:
                                    for block in result.read.blocks:
                                        if block.lines is not None:
                                            for line in block.lines:
                                                if re.match(r"^\d{8}/\d{13}\s?$", line.text):
                                                    code = re.sub(r"^\d{8}/", "", line.text)
                                                    diccionary["Codigo"].append(code)

                                with open(image_path[:-4] + "_" + "quantity" + ".jpeg", "rb") as f:
                                    image_data = f.read()
                                result = image_analysis_client.analyze(
                                    image_data=image_data,
                                    visual_features=[VisualFeatures.READ])

                                if result.read is not None and result.read.blocks:
                                    for block in result.read.blocks:
                                        if block.lines is not None:
                                            for line in block.lines:
                                                if re.match(r"\b\d{4}$", line.text):
                                                    code = re.sub(r"^(\d{4}).*", r"\1", line.text)
                                                    diccionary["Cantidad"].append(code)

                                                elif re.match(r"\b\d{5}$", line.text):
                                                    code = re.sub(r"^(\d{5}).*", r"\1", line.text)
                                                    diccionary["Cantidad"].append(code)

                                                elif re.match(r"\b\d{6}$", line.text):
                                                    code = re.sub(r"^(\d{5}).*", r"\1", line.text)
                                                    diccionary["Cantidad"].append(code)

                                with open(image_path[:-4] + "_" + "order" + ".jpeg", "rb") as f:
                                    image_data = f.read()
                                result = image_analysis_client.analyze(
                                    image_data=image_data,
                                    visual_features=[VisualFeatures.READ])

                                if result.read is not None and result.read.blocks:
                                    for block in result.read.blocks:
                                        if block.lines is not None:
                                            for line in block.lines:
                                                if re.match(r"PO#\s*\d+", line.text):
                                                    code = re.sub(r"PO#\s*", "", line.text)
                                                    diccionary["Orden de compra"].append(code)

                                while (len(diccionary["Remision"]) < len(diccionary["Orden de compra"]) or
                                       len(diccionary["Remision"]) < len(diccionary["Codigo"])):
                                    diccionary["Remision"].append(diccionary["Remision"][-1])

                                while len(diccionary["Cantidad"]) > len(diccionary["Codigo"]):
                                    diccionary["Cantidad"].pop()

                                while len(diccionary["Remision"]) > len(diccionary["Codigo"]):
                                    diccionary["Remision"].pop()
                                break

            date = get_date(pdf)
            while len(diccionary["Fecha"]) < len(diccionary["Codigo"]):
                diccionary["Fecha"].append(date)

            while len(diccionary["Proveedor"]) < len(diccionary["Codigo"]):
                diccionary["Proveedor"].append(provider)

        added_df = pd.DataFrame.from_dict(diccionary)
        st.session_state.glob_df = pd.concat([st.session_state.glob_df, added_df], ignore_index=True)

        st.success("Archivo generado exitosamente")
        remove_all_files_in_directory_pathlib("pages/")
        remove_all_files_in_directory_pathlib("pdfs/")

    #except:
        #st.error("Ocurrio un error")

def main_mcc_function(pdf, provider):
    try:
        global glob_df
        with st.spinner("Esperando ..."):
            number_of_images = load_pdf(pdf)
            diccionary = {"Fecha": [], "Codigo": [], "Orden de compra": [], "Cantidad": [], "Numero de Lote": [],
                          "Remision": [], "Proveedor": []}

            for k in range(number_of_images):
                image_path = "pages/page" + str(k) + ".jpg"

                # Load image to analyze into a 'bytes' object
                with open(image_path, "rb") as f:
                    image_data = f.read()

                result = image_analysis_client.analyze(
                    image_data=image_data,
                    visual_features=[VisualFeatures.READ]
                )

                if result.read is not None and result.read.blocks:
                    for block in result.read.blocks:
                        if block.lines is not None:
                            for line in block.lines:
                                if re.search(r"(?i)\bPro\s*-\s*Forma\b", line.text):

                                    page = cv2.imread(image_path)
                                    ## Product Code
                                    cropped = crop(page, 0.2, 0, 0.28, 0.15)
                                    cv2.imwrite(image_path[:-4] + "_" + "code" + ".jpeg", cropped)

                                    ## Buy Order
                                    cropped = crop(page, 0.2, 0.48, 0.28, 0.63)
                                    cv2.imwrite(image_path[:-4] + "_" + "order" + ".jpeg", cropped)

                                    ## Quantity
                                    cropped = crop(page, 0.2, 0.62, 0.28, 0.73)
                                    cv2.imwrite(image_path[:-4] + "_" + "quantity" + ".jpeg", cropped)

                                    ## Batch Number
                                    cropped = crop(page, 0.22, 0.62, 0.285, 0.75)
                                    cv2.imwrite(image_path[:-4] + "_" + "batch" + ".jpeg", cropped)

                                    ## Reference Number
                                    cropped = crop(page, 0.08, 0.78, 0.17, 1)
                                    cv2.imwrite(image_path[:-4] + "_" + "reference" + ".jpeg", cropped)

                                    with open(image_path[:-4] + "_" + "code" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"[a-zA-Z0-9\s]*\d{13}[a-zA-Z0-9\s]*", line.text):
                                                        diccionary["Codigo"].append(line.text)

                                    with open(image_path[:-4] + "_" + "order" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"^\d{10}$", line.text):
                                                        diccionary["Orden de compra"].append(line.text)

                                    with open(image_path[:-4] + "_" + "quantity" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"^\d{1,3}(,\d{3})+$", line.text):
                                                        code = re.sub(",", "", line.text)
                                                        diccionary["Cantidad"].append(code)

                                    with open(image_path[:-4] + "_" + "batch" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"Lote:\s*\d+", line.text):
                                                        code = re.sub(r"Lote:\s*", "", line.text)
                                                        diccionary["Numero de Lote"].append(code)

                                    with open(image_path[:-4] + "_" + "reference" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"No\. BOL: \d+", line.text):
                                                        code = re.sub(r"No\. BOL: ", "", line.text)
                                                        diccionary["Remision"].append(code)

                                    while (len(diccionary["Remision"]) < len(diccionary["Orden de compra"]) or
                                           len(diccionary["Remision"]) < len(diccionary["Codigo"])):
                                        diccionary["Remision"].append(diccionary["Remision"][-1])

                                    while len(diccionary["Cantidad"]) > len(diccionary["Codigo"]):
                                        diccionary["Cantidad"].pop()

                                    while len(diccionary["Remision"]) > len(diccionary["Codigo"]):
                                        diccionary["Remision"].pop()
                                    break

            while len(diccionary["Proveedor"]) < len(diccionary["Codigo"]):
                diccionary["Proveedor"].append(provider)
            date = get_date(pdf)
            while len(diccionary["Fecha"]) < len(diccionary["Codigo"]):
                diccionary["Fecha"].append(date)

        added_df = pd.DataFrame.from_dict(diccionary)
        st.session_state.glob_df = pd.concat([st.session_state.glob_df, added_df], ignore_index=True)

        st.success("Archivo generado exitosamente")
        remove_all_files_in_directory_pathlib("pages/")
        remove_all_files_in_directory_pathlib("pdfs/")

    except:
        st.error("Ocurrio un error")


def main_alpla_function(pdf, provider):
    try:
        global glob_df
        with st.spinner("Esperando ..."):
            diccionary = {"Fecha": [], "Codigo": [], "Orden de compra": [], "Cantidad": [], "Numero de Lote": [],
                          "Remision": [], "Proveedor": []}
            number_of_images = load_pdf(pdf)

            for k in range(number_of_images):
                image_path = "pages/page" + str(k) + ".jpg"

                # Load image to analyze into a 'bytes' object
                with open(image_path, "rb") as f:
                    image_data = f.read()

                result = image_analysis_client.analyze(
                    image_data=image_data,
                    visual_features=[VisualFeatures.READ]
                )

                if result.read is not None and result.read.blocks:
                    for block in result.read.blocks:
                        if block.lines is not None:
                            for line in block.lines:
                                if re.search(r".*\bFacturar\b.*", line.text) or re.search(r".*\bDelivery note\b.*",
                                                                                          line.text):

                                    page = cv2.imread(image_path)

                                    ## Product Code
                                    cropped = crop(page, 0.35, 0.07, 0.64, 0.3)
                                    cv2.imwrite(image_path[:-4] + "_" + "code" + ".jpeg", cropped)

                                    ## Buy Order
                                    cropped = crop(page, 0.3, 0.2, 0.64, 0.38)
                                    cv2.imwrite(image_path[:-4] + "_" + "order" + ".jpeg", cropped)

                                    ## Quantity
                                    cropped = crop(page, 0.36, 0.76, 0.64, 1)
                                    cv2.imwrite(image_path[:-4] + "_" + "quantity" + ".jpeg", cropped)

                                    ## Reference Number
                                    cropped = crop(page, 0.15, 0, 0.28, 0.5)
                                    cv2.imwrite(image_path[:-4] + "_" + "reference" + ".jpeg", cropped)

                                    with open(image_path[:-4] + "_" + "code" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"^\d{13}.*", line.text):
                                                        code = re.sub(r"^.*?(\d{13}).*$|^(?!.*\d{13}).*$", r"\1", line.text)
                                                        diccionary["Codigo"].append(code)

                                    with open(image_path[:-4] + "_" + "order" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{10}[/.].*", line.text):
                                                        code = re.sub(r".*?(\b\d{10}\b).*|.*", r"\1", line.text)
                                                        diccionary["Orden de compra"].append(code)

                                    with open(image_path[:-4] + "_" + "quantity" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{1,3}(,\d{3})*\.\d{2}\sPieza$", line.text):
                                                        code = re.search(r"[\d,]+", re.sub(r"[^\d,]",
                                                                                           "", line.text)).group(
                                                            0) if re.search(r"[\d,]+",
                                                                            re.sub(r"[^\d,]", "", line.text)) else None
                                                        code = re.sub(",", "", code)
                                                        code = int(code)
                                                        code = int(code / 100)
                                                        code = str(code)
                                                        diccionary["Cantidad"].append(code)

                                                    elif re.match(r".*\d{1,3}(\.\d{3})*,\d{2}\sPieza$", line.text):
                                                        code = re.search(r"[\d,]+", re.sub(r"[^\d,]",
                                                                                           "", line.text)).group(0) \
                                                            if re.search(r"[\d,]+", re.sub(r"[^\d,]",
                                                                                           "", line.text)) else None
                                                        code = re.sub(",", "", code)
                                                        code = int(code)
                                                        code = int(code / 100)
                                                        code = str(code)
                                                        diccionary["Cantidad"].append(code)

                                    with open(image_path[:-4] + "_" + "reference" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"Nota de Entrega \d+", line.text) or re.match(
                                                            r"Delivery note \d+", line.text):
                                                        code = re.sub(r"Nota de Entrega ", "", line.text)
                                                        code = re.sub(r"Delivery note ", "", code)
                                                        diccionary["Remision"].append(code)

                                    while (len(diccionary["Remision"]) < len(diccionary["Orden de compra"]) or
                                           len(diccionary["Remision"]) < len(diccionary["Codigo"])):
                                        diccionary["Remision"].append(diccionary["Remision"][-1])

                                    while len(diccionary["Cantidad"]) > len(diccionary["Codigo"]):
                                        diccionary["Cantidad"].pop()

                                    while len(diccionary["Remision"]) > len(diccionary["Codigo"]):
                                        diccionary["Remision"].pop()

                                    while len(diccionary["Orden de compra"]) < len(diccionary["Codigo"]):
                                        diccionary["Orden de compra"].append(diccionary["Orden de compra"][-1])
                                    break


                                elif (re.search(r".*\bCertificado de analisis\b.*", line.text) and
                                      len(diccionary["Numero de Lote"]) < len(diccionary["Codigo"])):

                                    page = cv2.imread(image_path)

                                    ## Batch Number
                                    cropped = crop(page, 0.1, 0.2, 0.2, 0.4)
                                    cv2.imwrite(image_path[:-4] + "_" + "batch" + ".jpeg", cropped)

                                    with open(image_path[:-4] + "_" + "batch" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{6}.*", line.text):
                                                        code = re.sub(r"^.*?(\d{6}).*$|^(?!.*\d{13}).*$", r"\1", line.text)
                                                        diccionary["Numero de Lote"].append(code)

                                    while len(diccionary["Remision"]) < len(diccionary["Numero de Lote"]):
                                        diccionary["Remision"].append(diccionary["Remision"][-1])
                                    break

            while len(diccionary["Numero de Lote"]) < len(diccionary["Codigo"]):
                diccionary["Numero de Lote"].append(np.nan)

            while len(diccionary["Proveedor"]) < len(diccionary["Codigo"]):
                diccionary["Proveedor"].append(provider)

            date = get_date(pdf)
            while len(diccionary["Fecha"]) < len(diccionary["Codigo"]):
                diccionary["Fecha"].append(date)

        added_df = pd.DataFrame.from_dict(diccionary)
        st.session_state.glob_df = pd.concat([st.session_state.glob_df, added_df], ignore_index=True)

        st.success("Archivo generado exitosamente")
        remove_all_files_in_directory_pathlib("pages/")
        remove_all_files_in_directory_pathlib("pdfs/")

    except:
        st.error("Ocurrio un error")


def main_aptar_function(pdf, provider):
    try:
        global glob_df
        with st.spinner("Esperando ..."):
            diccionary = {"Fecha": [], "Codigo": [], "Orden de compra": [], "Cantidad": [], "Numero de Lote": [],
                          "Remision": [], "Proveedor": []}
            number_of_images = load_pdf(pdf)

            for k in range(number_of_images):
                image_path = "pages/page" + str(k) + ".jpg"

                # Load image to analyze into a 'bytes' object
                with open(image_path, "rb") as f:
                    image_data = f.read()

                result = image_analysis_client.analyze(
                    image_data=image_data,
                    visual_features=[VisualFeatures.READ]
                )

                if result.read is not None and result.read.blocks:
                    for block in result.read.blocks:
                        if block.lines is not None:
                            for line in block.lines:
                                if re.search(r".*\bFOLIO FISCAL\b.*", line.text):

                                    page = cv2.imread(image_path)

                                    ## Product Code
                                    cropped = crop(page, 0.3, 0, 0.8, 0.5)
                                    cv2.imwrite(image_path[:-4] + "_" + "code" + ".jpeg", cropped)

                                    ## Buy Order
                                    cropped = crop(page, 0.3, 0, 0.8, 0.5)
                                    cv2.imwrite(image_path[:-4] + "_" + "order" + ".jpeg", cropped)

                                    ## Batch Number
                                    cropped = crop(page, 0.3, 0, 0.8, 0.5)
                                    cv2.imwrite(image_path[:-4] + "_" + "batch" + ".jpeg", cropped)

                                    ## Reference Number
                                    cropped = crop(page, 0.08, 0, 0.18, 0.4)
                                    cv2.imwrite(image_path[:-4] + "_" + "reference" + ".jpeg", cropped)

                                    with open(image_path[:-4] + "_" + "code" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{13}.*", line.text):
                                                        code = re.sub(r"^.*?(\d{13}).*$|^(?!.*\d{13}).*$", r"\1", line.text)
                                                        diccionary["Codigo"].append(code)

                                    with open(image_path[:-4] + "_" + "reference" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"FACTURA:\s*(\d{10})", line.text):
                                                        code = re.sub(r".*?(\b\d{10}\b).*|.*", r"\1", line.text)
                                                        diccionary["Remision"].append(code)

                                    with open(image_path[:-4] + "_" + "order" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"Numero orden de compra:\s*(\d{10})", line.text):
                                                        code = re.sub(r".*?(\b\d{10}\b).*|.*", r"\1", line.text)
                                                        diccionary["Orden de compra"].append(code)

                                    with open(image_path[:-4] + "_" + "batch" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"Lote:\s*(\d+)\s*\(\s*([\d,.]+)\s*PC\s*\)", line.text):
                                                        coden = re.sub(r".*Lote:\s*(\d+).*", r"\1", line.text)
                                                        diccionary["Numero de Lote"].append(coden)

                                                        codeq = re.sub(r".*\(\s*([\d,]+)(?:\.\d+)?\s*PC\s*\).*", r"\1",
                                                                       line.text)
                                                        codeq = re.sub(",", "", codeq)
                                                        diccionary["Cantidad"].append(codeq)

                                    while len(diccionary["Codigo"]) < len(diccionary["Remision"]):
                                        diccionary["Codigo"].append(diccionary["Codigo"][-1])

                                    while len(diccionary["Codigo"]) > len(diccionary["Remision"]):
                                        diccionary["Codigo"].pop()
                                    break

            while len(diccionary["Numero de Lote"]) < len(diccionary["Codigo"]):
                diccionary["Numero de Lote"].append(np.nan)

            while len(diccionary["Proveedor"]) < len(diccionary["Codigo"]):
                diccionary["Proveedor"].append(provider)

            date = get_date(pdf)
            while len(diccionary["Fecha"]) < len(diccionary["Codigo"]):
                diccionary["Fecha"].append(date)

        added_df = pd.DataFrame.from_dict(diccionary)
        st.session_state.glob_df = pd.concat([st.session_state.glob_df, added_df], ignore_index=True)

        st.success("Archivo generado exitosamente")
        remove_all_files_in_directory_pathlib("pages/")
        remove_all_files_in_directory_pathlib("pdfs/")

    except:
        st.error("Ocurrio un error")


def main_graham_function(pdf, provider):
    try:
        global glob_df
        with st.spinner("Esperando ..."):
            diccionary = {"Fecha": [], "Codigo": [], "Orden de compra": [], "Cantidad": [], "Numero de Lote": [],
                          "Remision": [], "Proveedor": []}
            number_of_images = load_pdf(pdf)
            for k in range(number_of_images):
                image_path = "pages/page" + str(k) + ".jpg"

                # Load image to analyze into a 'bytes' object
                with open(image_path, "rb") as f:
                    image_data = f.read()

                result = image_analysis_client.analyze(
                    image_data=image_data,
                    visual_features=[VisualFeatures.READ]
                )

                if result.read is not None and result.read.blocks:
                    for block in result.read.blocks:
                        if block.lines is not None:
                            for line in block.lines:
                                if re.search(r".*\bREMISION\b.*", line.text):

                                    page = cv2.imread(image_path)

                                    ## Product Code
                                    cropped = crop(page, 0.4, 0.4, 0.8, 0.85)
                                    cv2.imwrite(image_path[:-4] + "_" + "code" + ".jpeg", cropped)

                                    ## Quantity
                                    cropped = crop(page, 0.4, 0.08, 0.8, 0.31)
                                    cv2.imwrite(image_path[:-4] + "_" + "quantity" + ".jpeg", cropped)

                                    ## Batch Number
                                    cropped = crop(page, 0.75, 0.25, 0.9, 0.5)
                                    cv2.imwrite(image_path[:-4] + "_" + "batch" + ".jpeg", cropped)

                                    ## Reference Number
                                    cropped = crop(page, 0.01, 0.7, 0.1, 0.95)
                                    cv2.imwrite(image_path[:-4] + "_" + "reference" + ".jpeg", cropped)

                                    with open(image_path[:-4] + "_" + "code" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{13}.*", line.text):
                                                        code = re.sub(r"^.*?(\d{13}).*$|^(?!.*\d{13}).*$", r"\1", line.text)
                                                        diccionary["Codigo"].append(code)

                                                    elif re.match(r"PO#:\s*\d{10}", line.text):
                                                        code = re.sub(r".*?(\d{10}).*?", r"\1", line.text)
                                                        diccionary["Orden de compra"].append(code)

                                    with open(image_path[:-4] + "_" + "reference" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"REMISION No\. \s*(\d{8})", line.text):
                                                        code = re.sub(r".*?(\b\d{8}\b).*|.*", r"\1", line.text)
                                                        diccionary["Remision"].append(code)

                                    with open(image_path[:-4] + "_" + "quantity" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{2}\.\d{3}.*", line.text):
                                                        code = re.sub(r".*?(\d{2}\.\d{3}).*?", r"\1", line.text)
                                                        code = re.sub(r"\.", "", code)
                                                        diccionary["Cantidad"].append(code)

                                            with open(image_path[:-4] + "_" + "batch" + ".jpeg", "rb") as f:
                                                image_data = f.read()
                                            result = image_analysis_client.analyze(
                                                image_data=image_data,
                                                visual_features=[VisualFeatures.READ])

                                            if result.read is not None and result.read.blocks:
                                                for block in result.read.blocks:
                                                    if block.lines is not None:
                                                        for line in block.lines:
                                                            if re.match(r".*Lote:\s[A-Z0-9]+", line.text):
                                                                code = re.sub(r"(?:.*Lote:\s)?([A-Z0-9]+).*", r"\1",
                                                                              line.text)
                                                                diccionary["Numero de Lote"].append(code)
                                            break

            while len(diccionary["Numero de Lote"]) < len(diccionary["Codigo"]):
                diccionary["Numero de Lote"].append(np.nan)

            while len(diccionary["Proveedor"]) < len(diccionary["Codigo"]):
                diccionary["Proveedor"].append(provider)

            date = get_date(pdf)
            while len(diccionary["Fecha"]) < len(diccionary["Codigo"]):
                diccionary["Fecha"].append(date)

        added_df = pd.DataFrame.from_dict(diccionary)
        st.session_state.glob_df = pd.concat([st.session_state.glob_df, added_df], ignore_index=True)

        st.success("Archivo generado exitosamente")
        remove_all_files_in_directory_pathlib("pages/")
        remove_all_files_in_directory_pathlib("pdfs/")

    except:
        st.error("Ocurrio un error")


def main_cuautipack_function(pdf, provider):
    try:
        global glob_df
        with st.spinner("Esperando ..."):
            diccionary = {"Fecha": [], "Codigo": [], "Orden de compra": [], "Cantidad": [], "Numero de Lote": [],
                          "Remision": [], "Proveedor": []}
            number_of_images = load_pdf(pdf)
            for k in range(number_of_images):
                image_path = "pages/page" + str(k) + ".jpg"

                # Load image to analyze into a 'bytes' object
                with open(image_path, "rb") as f:
                    image_data = f.read()

                result = image_analysis_client.analyze(
                    image_data=image_data,
                    visual_features=[VisualFeatures.READ]
                )

                if result.read is not None and result.read.blocks:
                    for block in result.read.blocks:
                        if block.lines is not None:
                            for line in block.lines:
                                if re.search(r".*\bTIPO REMISION\b.*", line.text):

                                    page = cv2.imread(image_path)

                                    ## Buy Order
                                    cropped = crop(page, 0.11, 0.6, 0.18, 0.75)
                                    cv2.imwrite(image_path[:-4] + "_" + "order" + ".jpeg", cropped)

                                    ## Reference Number
                                    cropped = crop(page, 0.08, 0.58, 0.16, 0.78)
                                    cv2.imwrite(image_path[:-4] + "_" + "reference" + ".jpeg", cropped)

                                    with open(image_path[:-4] + "_" + "order" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{10}.*", line.text):
                                                        code = re.sub(r".*?(\b\d{10}\b).*|.*", r"\1", line.text)
                                                        diccionary["Orden de compra"].append(code)

                                    with open(image_path[:-4] + "_" + "reference" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{8}.*", line.text):
                                                        code = re.sub(r".*(\b\d{8}\b).*|.*", r"\1", line.text)
                                                        if code.isnumeric():
                                                            if int(code) > 80000000:
                                                                diccionary["Remision"].append(code)
                                    break

                                elif re.search(r".*\bCERTIFICADO DE CALIDAD\b.*", line.text):
                                    page = cv2.imread(image_path)

                                    ## Product Code
                                    cropped = crop(page, 0.15, 0.1, 0.3, 0.5)
                                    cv2.imwrite(image_path[:-4] + "_" + "code" + ".jpeg", cropped)

                                    ## Quantity
                                    cropped = crop(page, 0.1, 0.6, 0.24, 1)
                                    cv2.imwrite(image_path[:-4] + "_" + "quantity" + ".jpeg", cropped)

                                    ## Batch Number
                                    cropped = crop(page, 0.1, 0.45, 0.26, 0.8)
                                    cv2.imwrite(image_path[:-4] + "_" + "batch" + ".jpeg", cropped)

                                    with open(image_path[:-4] + "_" + "code" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{13}.*", line.text):
                                                        code = re.sub(r".*?(\b\d{13}\b).*?|(.*?)", r"\1", line.text)
                                                        diccionary["Codigo"].append(code)

                                    with open(image_path[:-4] + "_" + "quantity" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"^\s*(\d+(?:,\d+)?)\s*piezas", line.text):
                                                        code = re.sub(r"^\s*(?:(\d+(?:,\d+)?)\s*piezas).*", r"\1",
                                                                      line.text)
                                                        code = re.sub(",", "", code)
                                                        diccionary["Cantidad"].append(code)

                                    with open(image_path[:-4] + "_" + "batch" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    match = re.search(r"(?:\D*|^)\b(\d{10})\b(?:\D*|$)", line.text)
                                                    if match:
                                                        code = match.group(1)
                                                        diccionary["Numero de Lote"].append(code)
                                    break

            while len(diccionary["Orden de compra"]) < len(diccionary["Numero de Lote"]):
                diccionary["Orden de compra"].append(diccionary["Orden de compra"][-1])
            while len(diccionary["Remision"]) < len(diccionary["Numero de Lote"]):
                diccionary["Remision"].append(diccionary["Remision"][-1])

            while len(diccionary["Numero de Lote"]) < len(diccionary["Codigo"]):
                diccionary["Numero de Lote"].append(np.nan)

            while len(diccionary["Proveedor"]) < len(diccionary["Codigo"]):
                diccionary["Proveedor"].append(provider)

            date = get_date(pdf)
            while len(diccionary["Fecha"]) < len(diccionary["Codigo"]):
                diccionary["Fecha"].append(date)

        added_df = pd.DataFrame.from_dict(diccionary)
        st.session_state.glob_df = pd.concat([st.session_state.glob_df, added_df], ignore_index=True)

        st.success("Archivo generado exitosamente")
        remove_all_files_in_directory_pathlib("pages/")
        remove_all_files_in_directory_pathlib("pdfs/")

    except:
        st.error("Ocurrio un error")


def main_lindal_function(pdf, provider):
    try:
        global glob_df
        with st.spinner("Esperando ..."):
            diccionary = {"Fecha": [], "Codigo": [], "Orden de compra": [], "Cantidad": [], "Numero de Lote": [],
                          "Remision": [], "Proveedor": []}

            number_of_images = load_pdf(pdf)
            for k in range(number_of_images):
                image_path = "pages/page" + str(k) + ".jpg"

                # Load image to analyze into a 'bytes' object
                with open(image_path, "rb") as f:
                    image_data = f.read()

                result = image_analysis_client.analyze(
                    image_data=image_data,
                    visual_features=[VisualFeatures.READ]
                )
                if result.read is not None and result.read.blocks:
                    for block in result.read.blocks:
                        if block.lines is not None:
                            for line in block.lines:
                                if re.search(r".*\bDATOS DE FACTURACION\b.*", line.text):

                                    page = cv2.imread(image_path)

                                    ## Product Code
                                    cropped = crop(page, 0.3, 0.2, 0.45, 0.45)
                                    cv2.imwrite(image_path[:-4] + "_" + "code" + ".jpeg", cropped)

                                    ## Buy Order
                                    cropped = crop(page, 0.3, 0.55, 0.45, 0.72)
                                    cv2.imwrite(image_path[:-4] + "_" + "order" + ".jpeg", cropped)

                                    ## Quantity
                                    cropped = crop(page, 0.3, 0.68, 0.45, 0.83)
                                    cv2.imwrite(image_path[:-4] + "_" + "quantity" + ".jpeg", cropped)

                                    ## Batch Number
                                    cropped = crop(page, 0.3, 0, 0.45, 0.23)
                                    cv2.imwrite(image_path[:-4] + "_" + "batch" + ".jpeg", cropped)

                                    ## Reference Number
                                    cropped = crop(page, 0.07, 0.7, 0.15, 0.95)
                                    cv2.imwrite(image_path[:-4] + "_" + "reference" + ".jpeg", cropped)

                                    with open(image_path[:-4] + "_" + "code" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{13}.*", line.text):
                                                        code = re.sub(r".*?(\b\d{13}\b).*?|(.*?)", r"\1", line.text)
                                                        diccionary["Codigo"].append(code)

                                    with open(image_path[:-4] + "_" + "reference" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"^\d{5}$", line.text):
                                                        code = re.sub(r".*?(\b\d{5}\b).*?|(.*?)", r"\1", line.text)
                                                        diccionary["Remision"].append(code)

                                    with open(image_path[:-4] + "_" + "order" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{10}.*", line.text):
                                                        code = re.sub(r".*?(\b\d{10}\b).*?|(.*?)", r"\1", line.text)
                                                        diccionary["Orden de compra"].append(code)

                                    with open(image_path[:-4] + "_" + "quantity" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"^\d+\.\d{2}$", line.text) or re.match(r"^\d+\.\d{3}$",
                                                                                                        line.text):
                                                        code = float(line.text)
                                                        code = code * 1000
                                                        code = int(code)
                                                        code = str(code)
                                                        diccionary["Cantidad"].append(code)

                                    with open(image_path[:-4] + "_" + "batch" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])
                                    batches = []

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    batches.append(line.text)
                                                batchline = "".join(batches)
                                                batchline = batchline.replace(" ", "")
                                                if re.findall(r"VAL2-\d{4}", batchline):
                                                    clean_batches = re.findall(r"VAL2-\d{4}", batchline)
                                                elif re.findall(r"GALA-\d{3}", batchline):
                                                    clean_batches = re.findall(r"GALA-\d{3}", batchline)
                                                else:
                                                    clean_batches = []
                                                for batch in clean_batches:
                                                    if len(diccionary["Numero de Lote"]) < len(diccionary["Codigo"]):
                                                        diccionary["Numero de Lote"].append(batch)

                                    while len(diccionary["Numero de Lote"]) < len(diccionary["Codigo"]):
                                        diccionary["Numero de Lote"].append(np.nan)
                                    break

            while len(diccionary["Numero de Lote"]) < len(diccionary["Codigo"]):
                diccionary["Numero de Lote"].append(np.nan)

            while len(diccionary["Proveedor"]) < len(diccionary["Codigo"]):
                diccionary["Proveedor"].append(provider)

            date = get_date(pdf)
            while len(diccionary["Fecha"]) < len(diccionary["Codigo"]):
                diccionary["Fecha"].append(date)

        added_df = pd.DataFrame.from_dict(diccionary)
        st.session_state.glob_df = pd.concat([st.session_state.glob_df, added_df], ignore_index=True)

        st.success("Archivo generado exitosamente")
        remove_all_files_in_directory_pathlib("pages/")
        remove_all_files_in_directory_pathlib("pdfs/")

    except:
        st.error("Ocurrio un error")


def main_cajaplax_function(pdf, provider):
    try:
        global glob_df
        with st.spinner("Esperando ..."):
            diccionary = {"Fecha": [], "Codigo": [], "Orden de compra": [], "Cantidad": [], "Numero de Lote": [],
                          "Remision": [], "Proveedor": []}
            number_of_images = load_pdf(pdf)
            for k in range(number_of_images):
                image_path = "pages/page" + str(k) + ".jpg"

                # Load image to analyze into a 'bytes' object
                with open(image_path, "rb") as f:
                    image_data = f.read()

                result = image_analysis_client.analyze(
                    image_data=image_data,
                    visual_features=[VisualFeatures.READ]
                )

                if result.read is not None and result.read.blocks:
                    for block in result.read.blocks:
                        if block.lines is not None:
                            for line in block.lines:
                                if re.search(r".*\bFACTURA\b.*", line.text):

                                    page = cv2.imread(image_path)

                                    ## Product Code
                                    cropped = crop(page, 0.25, 0.3, 0.4, 0.53)
                                    cv2.imwrite(image_path[:-4] + "_" + "code" + ".jpeg", cropped)

                                    ## Buy Order
                                    cropped = crop(page, 0.2, 0.6, 0.3, 0.92)
                                    cv2.imwrite(image_path[:-4] + "_" + "order" + ".jpeg", cropped)

                                    ## Quantity
                                    cropped = crop(page, 0.28, 0.6, 0.4, 0.8)
                                    cv2.imwrite(image_path[:-4] + "_" + "quantity" + ".jpeg", cropped)

                                    ## Batch Number
                                    cropped = crop(page, 0.75, 0, 0.85, 0.27)
                                    cv2.imwrite(image_path[:-4] + "_" + "batch" + ".jpeg", cropped)

                                    ## Reference Number
                                    cropped = crop(page, 0.1, 0.6, 0.18, 1)
                                    cv2.imwrite(image_path[:-4] + "_" + "reference" + ".jpeg", cropped)

                                    with open(image_path[:-4] + "_" + "code" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{13}.*", line.text):
                                                        code = re.sub(r".*?(\b\d{13}\b).*?|(.*?)", r"\1", line.text)
                                                        diccionary["Codigo"].append(code)

                                    with open(image_path[:-4] + "_" + "reference" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*NUMERO:.*", line.text):
                                                        code = re.sub("NUMERO: ", "", line.text)
                                                        diccionary["Remision"].append(code)

                                    with open(image_path[:-4] + "_" + "order" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r".*\d{10}.*", line.text):
                                                        code = re.sub(r".*?(\b\d{10}\b).*?|(.*?)", r"\1", line.text)
                                                        diccionary["Orden de compra"].append(code)

                                    with open(image_path[:-4] + "_" + "quantity" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    if re.match(r"^\d{1,3}(,\d{3})$", line.text):
                                                        code = re.sub(r",", r"", line.text)
                                                        diccionary["Cantidad"].append(code)

                                    with open(image_path[:-4] + "_" + "batch" + ".jpeg", "rb") as f:
                                        image_data = f.read()
                                    result = image_analysis_client.analyze(
                                        image_data=image_data,
                                        visual_features=[VisualFeatures.READ])

                                    if result.read is not None and result.read.blocks:
                                        for block in result.read.blocks:
                                            if block.lines is not None:
                                                for line in block.lines:
                                                    print(line.text)
                                                    if re.match(r".*LOTE: .*", line.text) or re.match(r".*LOTE:.*",
                                                                                                      line.text):
                                                        code = re.sub("LOTE: ", "", line.text)
                                                        code = re.sub("LOTE:", "", code)
                                                        diccionary["Numero de Lote"].append(code)
                                    break

                    while len(diccionary["Orden de compra"]) < len(diccionary["Numero de Lote"]):
                        diccionary["Orden de compra"].append(diccionary["Orden de compra"][-1])

                    while len(diccionary["Remision"]) < len(diccionary["Numero de Lote"]):
                        diccionary["Remision"].append(diccionary["Remision"][-1])

                    while len(diccionary["Numero de Lote"]) < len(diccionary["Codigo"]):
                        diccionary["Numero de Lote"].append(np.nan)

            while len(diccionary["Proveedor"]) < len(diccionary["Codigo"]):
                diccionary["Proveedor"].append(provider)

            date = get_date(pdf)
            while len(diccionary["Fecha"]) < len(diccionary["Codigo"]):
                diccionary["Fecha"].append(date)

        added_df = pd.DataFrame.from_dict(diccionary)
        st.session_state.glob_df = pd.concat([st.session_state.glob_df, added_df], ignore_index=True)

        st.success("Archivo generado exitosamente")
        remove_all_files_in_directory_pathlib("pages/")
        remove_all_files_in_directory_pathlib("pdfs/")

    except:
        st.error("Ocurrio un error")

########################################################################################################################
def activate_main(provider, pdf):
    if pdf is not None:
        st.subheader("Generando archivo")
        match provider:
            case "Ball":
                main_ball_function(pdf, provider)
            case "MCC":
                main_mcc_function(pdf, provider)
            case "Alpla":
                main_alpla_function(pdf, provider)
            case "Aptar":
                main_aptar_function(pdf, provider)
            case "Graham":
                main_graham_function(pdf, provider)
            case "Cuautipack":
                main_cuautipack_function(pdf, provider)
            case "Lindal":
                main_lindal_function(pdf, provider)
            case "Cajaplax":
                main_cajaplax_function(pdf, provider)
            case _:
                st.write("Selecciona un proveedor válido")
    else:
        st.write("Selecciona un archivo")

if 'glob_df' not in st.session_state:
    st.session_state.glob_df = pd.DataFrame({"Fecha": [], "Codigo": [], "Orden de compra": [], "Cantidad": [], "Numero de Lote": [],
                      "Remision": [], "Proveedor": []})

app_mode = st.sidebar.selectbox("Selecciona una pagina", ["Ingresar archivos", "Archivo generado"])
if app_mode == "Ingresar archivos":
    up_container = st.container()
    down_container = st.container()

    up_container.title("Ingresos")
    provider = up_container.selectbox("Selecciona el Proveedor",["Ball", "Alpla", "Aptar", "Graham", "Cuautipack", "Lindal", "MCC", "Cajaplax"])
    pdf = up_container.file_uploader("Insertar PDF", type="pdf")
    generate_button = up_container.button("Generar Excel", on_click=activate_main, args = (provider, pdf))

if app_mode == "Archivo generado":
    st.title("Archivo Generado")
    st.dataframe(st.session_state.glob_df)

