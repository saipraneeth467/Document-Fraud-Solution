import fitz
import re
from PyPDF2 import PdfReader
import pandas as pd
from datetime import datetime
from io import BytesIO
from dfs.commons import constants
from dfs.commons.ioutils.datastore_utils import read_from_data_lake
from pathlib import Path
import tempfile
import os
from PIL import Image, ImageDraw
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_vgg16_bn', pretrained = True)
import cv2
import math
import pandas as pd
import numpy as np
from spellchecker import SpellChecker
from dfs.commons.ioutils.datastore_utils import read_from_data_lake, write_to_data_lake

class MetadataAnalyzer:
    meta_error_list = []
    @staticmethod
    def get_expected_value(bank_name, meta_repo):
        
        df = meta_repo
        try:
            expected_value = [df.loc[bank_name]['Author'],df.loc[bank_name]['Creator'],'','',df.loc[bank_name]['Producer']]
            return expected_value
        except:
            return ''
        
    @classmethod   
    def datefun(cls, date1_str,date2_str):
        
        try:
            # from datetime import datetime
            if date1_str.startswith("D:"):
                try:
                    # Remove the single quotes around the timezone offset
                    date1_str = date1_str.replace("'", "")
                    date2_str = date2_str.replace("'", "")

                    # Convert the date strings to datetime objects
                    date1 = datetime.strptime(date1_str, 'D:%Y%m%d%H%M%S%z')
                    date2 = datetime.strptime(date2_str, 'D:%Y%m%d%H%M%S%z')
                except:
                    # Add a new condition for the "D:YYYYMMDDHHMMSS" format
                    # Convert the date strings to datetime objects
                    date1 = datetime.strptime(date1_str, 'D:%Y%m%d%H%M%S')
                    date2 = datetime.strptime(date2_str, 'D:%Y%m%d%H%M%S')

            elif '/' in date1_str:
                # Convert the date strings to datetime objects
                date_format = "%m/%d/%Y %H:%M:%S"
                date1 = datetime.strptime(date1_str, date_format)
                date2 = datetime.strptime(date2_str, date_format)

            else:
                # Remove the single quotes around the timezone offset
                date1_str = date1_str.replace("'", "")
                date2_str = date2_str.replace("'", "")

                # Convert the date strings to datetime objects
                date_format = "%Y%m%d%H%M%S%z"
                date1 = datetime.strptime(date1_str, date_format)
                date2 = datetime.strptime(date2_str, date_format)

            # Calculate the time difference
            time_difference = date1 - date2 

            # Extract the number of days, hours, and minutes
            total_seconds = time_difference.total_seconds()
            days, remainder = divmod(total_seconds, 24 * 3600)
            hours, remainder = divmod(remainder, 3600)
            minutes, _ = divmod(remainder, 60)
            
            return f"Time difference: {int(days)} days, {int(hours)} hours, and {int(minutes)} minutes"

        except:
            # Handle any exceptions (e.g., invalid date formats)
            print("Metadata: New Date format found")
            # return f"Error: {str(e)}"
            cls.meta_error_list.append(f"Metadata : New Date format found {date1_str} and {date2_str}")

        # return cls.meta_error_list
    @staticmethod
    def remove_version(input_string):
        # Use regex to match the version number
        pattern = r'\s*[\d\.\-\(\)\/\,\:\|\s]+'
        result = re.sub(pattern, '', input_string).strip()

        # Remove extra spaces
        result = re.sub(r'\s{2,}', ' ', result)
        # result = re.sub(r'\s+', ' ', result)
        return result
    
    @classmethod
    def metadata_main(cls, filename, bank_names, meta_repo, excel_df, PDF_type, four_corner_df2):
        
        bytes_stream = BytesIO(filename)
        try:
            file = PdfReader(bytes_stream)
            print('pdf file is detected')
        except Exception as e:
            print('Metadata Analysis failed -- Please make sure a pdf file is uploaded')
            return
        
        if type(file.metadata) == None.__class__:
            metadata = {'/Producer': '', '/CreationDate': '', '/ModDate': '', '/Author': '', '/Creator': '', '/Keywords': '', '/Subject': '', '/Title': ''}
        else:
            metadata = file.metadata
        
        metadata_list = ['/Author','/Creator','/CreationDate','/ModDate','/Producer']
        metadata_names = []
        metadata_values = []
        metadata_dic = {}
        bank_name = ''
        
        #try:
        for m in metadata_list:
            metadata_names.append(m)
            if m in metadata.keys():
                repo_list = ['/Creator', '/Producer']
                if m in repo_list:
                    str1 = cls.remove_version(str(metadata[m]))
                    metadata_values.append(str1)
                else:   
                    metadata_values.append(metadata[m])
            else:
                metadata_values.append('')
        df = pd.DataFrame(list(zip(metadata_names, metadata_values)), columns = ['Metadata', 'Value'])

        metadata_dic = dict(zip(df['Metadata'], df['Value']))
        expected_values = bank_names
        if expected_values != '':
            if len(expected_values) == 1:
                df['Expected_Value'] = cls.get_expected_value(expected_values[0],meta_repo)
                bank_name = expected_values[0]
            elif len(expected_values) > 1:
                real_bank = ''
                matching_num = 0
                for each_value in expected_values:
                    df['Expected_Value'] = cls.get_expected_value(each_value,meta_repo)
                    for i in range(len(df)):
                        temp = 0
                    if df['Expected_Value'].iloc[i] in df['Value'].iloc[i]:
                        temp += 1
                    if temp > matching_num:
                        matching_num = temp
                        real_bank = each_value
                df['Expected_Value'] = cls.get_expected_value(real_bank,meta_repo)
                bank_name = real_bank
            
            matches = []
            for i in range(len(df)):
                if df['Expected_Value'].iloc[i] != '' and df['Value'].iloc[i] != '':
                    if df['Value'].iloc[i].rstrip() in df['Expected_Value'].iloc[i]:
                        matches.append('yes')
                    else:
                        matches.append('no')
                else:
                    matches.append('')

            df['Match'] = matches
            # doc = fitz.open(stream = filename, filetype = 'pdf')
            # for page_Number in range(doc.page_count):
            #     page = doc.load_page(page_Number) #load page
            #     text = page.get_text()
            #     if text.strip():
            #         pdf_type = "Input PDF is text based"
            #         break
            #     elif page.get_pixmap():
            #         pdf_type = "Input PDF is image based"
            #         break
            # doc.close()
        #     pdf_type_1 = pdf_type
        #     df['doc_type'] = pdf_type_1
        # excel_df.loc[len(excel_df.index)] = [df, '1A.Metadata_Report']
        # return metadata_dic, bank_name, excel_df
            df.loc[5,['Metadata', 'Value']]=['Doc_Type',PDF_type]
            creation_date = metadata_dic['/CreationDate'] 
            modification_date = metadata_dic['/ModDate']
            if creation_date!="" and modification_date!="":
                diff = cls.datefun(modification_date,creation_date)
            elif creation_date!="" and modification_date== "":
                diff = "Mod_Date is not present"
            elif creation_date=="" and modification_date!= "":
                diff = "Creation_Date is not present"
            else:
                diff = "Creation_Date and Mod_Date is not present in doc"

            df.loc[6,['Metadata', 'Value']]=['Difference b/w Mod_Date and Creation_Date',diff]
            
        df = pd.concat([df,four_corner_df2])
        # print('----------------------------------------------------------')
        # display(df)
        excel_df.loc[len(excel_df.index)] = [df, '1A.Metadata_Raw_Output']
        return metadata_dic,bank_name,excel_df

    
    @staticmethod
    def metadata_rules(metadata, meta_repo, pdf_software_repo, scanner_repo, final_report_insight, excel_df, four_corner_df, PDF_type):

        if metadata == {}:
            print('Metadata Analysis failed -- No metadata detected from file')
            return

        # initiating final output columns
        Insight = ['']*14
        num_of_alerts = 0
        Severity_Level = ['']*14
        Flag = [0]*14


        df = meta_repo
        producer_list = list(df['Producer'])
        producer_list = [n for n in producer_list if n != '']
        
        creator_list = list(df['Creator'])
        creator_list = [n for n in creator_list if n != '']

        author_list = list(df['Author'])
        author_list = [n for n in author_list if n != '']

        # scanners list
        df1 = scanner_repo
        scanner_list = list(df1['Scanner_List'])
        scanner_list = [n for n in scanner_list if n != '']

        # editing software list
        df2 = pdf_software_repo
        pdf_software = list(df2['Editing_software'])
        pdf_software = [n for n in pdf_software if n != '']

        # rule 1: producer isn't in the list of scanner brands & producer is not an engine
        in_list = False
        for scanner in scanner_list:
            if scanner in metadata['/Producer']:
                in_list = True

        for producer in producer_list:
            if metadata['/Producer'] in producer:
                in_list = True

        if in_list == False:
            num_of_alerts += 1
            Flag[0] = 1
            if PDF_type == 'Digital':
                Severity_Level[0] = 'HIGH'
            else:
                Severity_Level[0] = 'MEDIUM'
            Insight[0] = 'Suspicious producer found in metadata'


        # rule 2: producer is in the list of commonly used pdf editing software
        for software in pdf_software:
            if software in metadata['/Producer']:
                num_of_alerts += 1
                Flag[1] = 1
                Severity_Level[1] = 'HIGH'
                Insight[1] = 'Editing software as producer detected in metadata'


        # rule 3: producer is blank
        if metadata['/Producer'] == '':
            num_of_alerts += 1
            Flag[2] = 1
            Severity_Level[2] = 'MEDIUM'
            Insight[2] = 'Producer missing in metadata'


        # rule 4: Creator is in the list of commonly used pdf editing software
        for software in pdf_software:
            if software in metadata['/Creator']:
                num_of_alerts += 1
                Flag[3] = 1
                Severity_Level[3] = 'HIGH'
                Insight[3] = 'Editing software as creator detected in metadata'

        # rule 5: creator is blank
        if metadata['/Creator'] == '':
            num_of_alerts += 1
            Flag[4] = 1
            Severity_Level[4] = 'MEDIUM'
            Insight[4] = 'Creator missing in metadata'


        # rule 6: Author is in the list of commonly used pdf editing software
        for software in pdf_software:
            if software in metadata['/Author']:
                num_of_alerts += 1
                Flag[5] = 1
                Severity_Level[5] = 'HIGH'
                Insight[5] = 'Editing software as author detected in metadata'
                
        # rule 7: creation date != modification date
        if metadata['/CreationDate'] and metadata['/ModDate']:
            if metadata['/CreationDate'] != metadata['/ModDate']:
                num_of_alerts += 1
                Flag[6] = 1
                #Severity_Level[6] = 'LOW'
                Severity_Level[6] = 'HIGH'
                Insight[6] = 'The file was modified after creation'


        today_datetime = datetime.now()
        # rules regarding creation date
        if metadata['/CreationDate'] != '' and len(metadata['/CreationDate']) > 6:
            current_dateTime = datetime.now()
            year = str(current_dateTime.year)
            if current_dateTime.month < 10:
                month = '0' + str(current_dateTime.month)
            else:
                month = str(current_dateTime.month)
            if current_dateTime.day < 10:
                day = '0' + str(current_dateTime.day)
            else:
                day = str(current_dateTime.day)


            creationdate = metadata['/CreationDate']
            # get the date string
            match = re.search(r'\d+', creationdate)
            if match:
                creationdate = creationdate[match.start():]

            if creationdate.startswith("D:"):
                created_year = int(creationdate[2:6])
                created_month = int(creationdate[6:8])
                created_day = int(creationdate[8:10])
                creationdate = f"{created_year:04d}-{created_month:02d}-{created_day:02d}"
                creationdate = pd.to_datetime(creationdate)
            elif '/' in creationdate:
                date_format = "%m/%d/%Y %H:%M:%S"
                creationdate = datetime.strptime(creationdate,date_format)
                creationdate = creationdate.strftime("%Y-%m-%d")  
                created_year = int(creationdate[:4])
                created_month = int(creationdate[5:7])
                created_day = int(creationdate[8:10])
                creationdate = pd.to_datetime(creationdate)
            else:
                created_year = int(creationdate[:4])
                created_month = int(creationdate[4:6])
                created_day = int(creationdate[6:8])
                creationdate = f"{created_year:04d}-{created_month:02d}-{created_day:02d}"
                creationdate = pd.to_datetime(creationdate)


            created_datetime = datetime(int(created_year),int(created_month),int(created_day))
            # rule 8: today's date - creation date > 3 months
            if (today_datetime-created_datetime).days > 90:
                num_of_alerts += 1
                Flag[7] = 1
                #Severity_Level[7] = 'LOW'
                Severity_Level[7] = 'LOW'
                Insight[7] = 'The file was created more than three months ago'

            # rule 10: today's date - mod date < 3 days
            if (today_datetime-created_datetime).days < 3:
                num_of_alerts += 1
                Flag[9] = 1
                #Severity_Level[9] = 'LOW'
                Severity_Level[9] = 'HIGH'
                Insight[9] = 'The file was last created within three days'


        # rules regarding modification date
        if metadata['/ModDate'] != '':

            moddate = metadata['/ModDate']
            match1 = re.search(r'\d+', moddate)
            if match1:
                moddate = moddate[match1.start():]

            if moddate.startswith("D:"):
                mod_year = int(moddate[2:6])
                mod_month = int(moddate[6:8])
                mod_day = int(moddate[8:10])
                moddate = f"{mod_year:04d}-{mod_month:02d}-{mod_day:02d}"
                moddate = pd.to_datetime(moddate)
            elif '/' in moddate:
                date_format = "%m/%d/%Y %H:%M:%S"
                moddate = datetime.strptime(moddate,date_format)
                moddate = moddate.strftime("%Y-%m-%d")  
                mod_year = int(moddate[:4])
                mod_month = int(moddate[5:7])
                mod_day = int(moddate[8:10])
                moddate = pd.to_datetime(moddate)
            else:
                mod_year = int(moddate[:4])
                mod_month = int(moddate[4:6])
                mod_day = int(moddate[6:8])
                moddate = f"{mod_year:04d}-{mod_month:02d}-{mod_day:02d}"
                moddate = pd.to_datetime(moddate)

            mod_datetime = datetime(int(mod_year),int(mod_month),int(mod_day))

            # rule 9: today's date - mod date > 3 months
            if (today_datetime-mod_datetime).days > 90:
                num_of_alerts += 1
                Flag[8] = 1
                #Severity_Level[8] = 'LOW'
                Severity_Level[8] = 'LOW'
                Insight[8] = 'The file was last modified more than three months ago'

            # rule 11: today's date - mod date < 3 days
            if (today_datetime-mod_datetime).days < 3:
                num_of_alerts += 1
                Flag[10] = 1
                #Severity_Level[10] = 'LOW'
                Severity_Level[10] = 'HIGH'
                Insight[10] = 'The file was last modified within three days'



        # rule 11: creation date is blank
        if metadata['/CreationDate'] == '' :
            num_of_alerts += 1
            Flag[11] = 1
            #Severity_Level[11] = 'LOW'
            Severity_Level[11] = 'HIGH'
            Insight[11] = 'Creation date not detected in metadata'

        # rule 12: modification date is blank
        if metadata['/ModDate'] == '' :
            num_of_alerts += 1
            Flag[12] = 1
            Severity_Level[12] = 'LOW'
            Insight[12] = 'Modification date not detected in metadata'

        # rule 13: producer is in the list of image to pdf convertor
        convertors = ['Wondershare PDFelement','FM-PDF','TalkHelper','JPG to PDF','Atop','PDFlite','Framework Team','SuperGeek']
        for convertor in convertors:
            if convertor in metadata['/Producer']:
                num_of_alerts += 1
                Flag[13] = 1
                Severity_Level[13] = 'HIGH'
                Insight[13] = 'Image to pdf convertor was found in metadata'


        rule = ['Producer_not_engine_or_scanner','Producer_editing_software','Producer_missing',
                'Creator_editing_software','Creator_missing','Author_editing_software',
                'Creation_dt_different_from_modification_dt','Created_more_than_3M',
                'Modified_more_than_3M','Created_within_last_3D','Modified_within_last_3D','Creation_dt_missing',
                'Modification_dt_missing','Producer_image_to_PDF_converter']

        #generate final report for metadata analysis
        df = pd.DataFrame(list(zip(rule, Flag, Severity_Level, Insight)), columns = ['Rule', 'Flag','Severity_Level','Insight'])

        #generate json file for alerts
        df2 = df[df['Flag'] == 1][['Insight','Severity_Level']]
        df2['Module'] = 'Metadata'
        df2.columns = ['Alerts','Severity', 'Module']
        df2 = df2[['Module','Alerts', 'Severity']]
        final_report_insight = pd.concat([final_report_insight,df2])
        df = pd.concat([df,four_corner_df], ignore_index=True).fillna('')
        # print('----------------------------------------------------------')
        # display(df)
        excel_df.loc[len(excel_df.index)] = [df, '1B.Metadata_Report']
        return final_report_insight,excel_df
    
    @staticmethod
    def meta_repo_value(bank_name, pdf_filename, pdf_filename_file):
        df_final = pd.DataFrame()
        bytes_stream = BytesIO(pdf_filename)
        
        try:
            file = PdfReader(bytes_stream)
            # print('pdf file is detected')
        except Exception as e:
            # print('Metadata Repo Cannot be stored')
            return
        if type(file.metadata) == None.__class__:
            metadata = {'/Producer': '', '/CreationDate': '', '/ModDate': '', '/Author': '', '/Creator': '', '/Keywords': '', '/Subject': '', '/Title': ''}
        else:
            metadata = file.metadata
        metadata_list = ['/Author','/Creator','/CreationDate','/ModDate','/Producer']
        metadata_names = []
        metadata_values = []
    
        for m in metadata_list:
            metadata_names.append(m) 
            if m in metadata.keys():
                metadata_values.append(metadata[m])
            else:
                metadata_values.append('')
        df = pd.DataFrame(list(zip(metadata_names, metadata_values)), columns = ['Metadata', 'Value'])
        df['bank'] = bank_name
        res = df.pivot(index='bank',columns='Metadata', values='Value')
        df_final = pd.concat([res, df_final], axis=0) 
        df_final['file_name'] = pdf_filename_file    
        return df_final
    
    
    # 4 corner analysis
#     @staticmethod
#     def doc_extraction(image_path):
#         image = Image.open(image_path)
#         # image = read_from_data_lake(image_path)
#         image = image.convert('RGB')
#         width,height = image.size[:2]
#         image_shape = [width, height]
        
#         doc = DocumentFile.from_images(image_path)
#         result = model(doc)
#         output = result.export()
#         return image_shape, output
    
    @staticmethod
    def doctr_check(image_path):

        image = Image.open(image_path)
        # image = read_from_data_lake(image_path)
        image = image.convert('RGB')
        width,height = image.size[:2]
        image_shape = [width, height]

        doc = DocumentFile.from_images(image_path)
        # img1 = read_from_data_lake(image_path)
        # with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_path)[-1]) as temp_file:
        #     img1.save(temp_file, format=img1.format)
        #     temp_file_path = temp_file.name   
        # doc = DocumentFile.from_images(temp_file_path)
        
        result = model(doc)
        output = result.export()
        word_count_area = 0
        word_count = 0

        # Calculate the word count area and word count
        for page in output['pages']:
            for block in page['blocks']:
                for line in block['lines']:
                    for word in line['words']:
                        # Extract coordinates and convert to absolute values
                        (x1, y1), (x2, y2) = word['geometry']
                        abs_x1 = int(x1 * image_shape[1])
                        abs_y1 = int(y1 * image_shape[0])
                        abs_x2 = int(x2 * image_shape[1])
                        abs_y2 = int(y2 * image_shape[0])

                        # Calculate the area of the word bounding box
                        word_area = (abs_x2 - abs_x1) * (abs_y2 - abs_y1)
                        word_count_area += word_area

                        # Count each word
                        word_count += len(word['value'].split())

        # Calculate the page area
        page_area = image_shape[0] * image_shape[1]
        ratio = word_count_area / page_area

        def convert_coordinates(geometry, page_dim):
            len_x = page_dim[1]
            len_y = page_dim[0]
            (x_min, y_min) = geometry[0]
            (x_max, y_max) = geometry[1]
            x_min = math.floor(x_min * len_x)
            x_max = math.ceil(x_max * len_x)
            y_min = math.floor(y_min * len_y)
            y_max = math.ceil(y_max * len_y)
            return [x_min, y_min, x_max,y_max]

        def get_coordinates(output, image_shape):
            page_dim = output['pages'][0]["dimensions"]
            text_coordinates = []
            left_crop = top_crop = right_crop = bottom_crop = False
            for obj1 in output['pages'][0]["blocks"]:
                for obj2 in obj1["lines"]:
                    for obj3 in obj2["words"]:                
                        converted_coordinates = convert_coordinates(
                                                   obj3["geometry"],page_dim
                                                  )
                        # print("{}: {}".format(converted_coordinates,
                        #                       obj3["value"]
                        #                       )
                        #      )
                        text_coordinates.append([converted_coordinates,obj3["value"]])
            return text_coordinates
        def get_line_word_count(text_coordinates):
            first_line_y = text_coordinates[0][0][1]
            last_line_y = text_coordinates[-1][0][1]
            first_line_words = [i[1] for i in text_coordinates if i[0][1] == first_line_y]
            last_line_words = [i[1] for i in text_coordinates if i[0][1] == last_line_y]

            if any(i[0][2] > 0.9 * width for i in text_coordinates if i[0][1] == first_line_y):
                first_line_corner_check  = True
            else:
                first_line_corner_check = False
            if any(i[0][2] > 0.9 * width for i in text_coordinates if i[0][1] == last_line_y):
                last_line_corner_check  = True
            else:
                last_line_corner_check = False
            return len(first_line_words), len(last_line_words), first_line_y, last_line_y, first_line_corner_check, last_line_corner_check

        spell=SpellChecker()
        text_coordinates = get_coordinates(output, image.size)
        first_line_word_count, last_line_word_count, first_line_y, last_line_y, first_line_corner_check, last_line_corner_check = get_line_word_count(text_coordinates)
        if first_line_corner_check and first_line_word_count < 3:
            text_coordinates = [i for i in text_coordinates if i[0][1] != first_line_y]
        if last_line_corner_check and last_line_word_count <3:
            text_coordinates = [i for i in text_coordinates if i[0][1] != last_line_y]

        if text_coordinates:
            top = min([i[0][1] for i in text_coordinates])
            left = min([i[0][0] for i in text_coordinates])
            bottom = max([i[0][3] for i in text_coordinates])
            right = max([i[0][2] for i in text_coordinates])
        else:
            top = 0
            bottom = image.size[1]
            left = 0
            right = image.size[0]

        doc_check = 0

        margin_diffs = [round(top/height,3), round(left/width,3), round(1-(bottom/height),3), round(1-(right/width),3)]
        count = sum(diff>0.02 for diff in margin_diffs)
        if (round(top/height,3) < 0.02) or (round(left/width,3) < 0.02) or (round(1-(bottom/height),3) < 0.02) or (round(1-(right/width),3) < 0.02):
            doc_check = 0
        else:
            doc_check = 1

        if doc_check == 0:

            if (round(top/height,3) < 0.02):
                edge_text = []
                spelling_check_true = []
                for i in text_coordinates:
                    if abs(i[0][1] - top) < 5:
                        edge_text.append(i[1])
                if len(edge_text) > 0:
                    for word in edge_text:
                        if word in spell:
                            spelling_check_true.append(word)
                    if len(edge_text) > 0:
                        if len(spelling_check_true)/len(edge_text) > 0.9:
                            doc_check = 1

            if (round(left/width,3) < 0.02):
                edge_text = []
                spelling_check_true = []
                for i in text_coordinates:
                    if abs(i[0][0] - left) < 5:
                        edge_text.append(i[1])
                if len(edge_text) > 0:
                    for word in edge_text:
                        if word in spell:
                            spelling_check_true.append(word)
                    if len(edge_text) > 0:
                        if len(spelling_check_true)/len(edge_text) > 0.9:
                            doc_check = 1


            if (round(bottom/height,3) < 0.02):
                edge_text = []
                spelling_check_true = []
                for i in text_coordinates:
                    if abs(i[0][3] - bottom) < 5:
                        edge_text.append(i[1])
                if len(edge_text) > 0:
                    for word in edge_text:
                        if word in spell:
                            spelling_check_true.append(word)
                    if len(edge_text) > 0:
                        if len(spelling_check_true)/len(edge_text) > 0.9:
                            doc_check = 1


            if (round(right/width,3) < 0.02):
                edge_text = []
                spelling_check_true = []
                for i in text_coordinates:
                    if abs(i[0][2] - right) < 5:
                        edge_text.append(i[1])
                if len(edge_text) > 0:
                    for word in edge_text:
                        if word in spell:
                            spelling_check_true.append(word)
                    if len(edge_text) > 0:
                        if len(spelling_check_true)/len(edge_text) > 0.9:
                            doc_check = 1


        # if doc_check:
        #     print("DB boundary is fully enclosed by the image boundary.")
        # else:
        #     print("DB boundary is not fully enclosed by the image boundary - less than the minimum permissible distance from the image boundary.")

        # draw = ImageDraw.Draw(image)
        margin = [round(top/height,3),round(left/width,3),round(1-(bottom/height),3),round(1-(right/width),3)]
        # print("Margin differences: ",round(top/height,3),round(left/width,3),round(1-(bottom/height),3),round(1-(right/width),3))
        # print("Count :", count)
        # draw.rectangle([left, top, right, bottom], outline="green", width=3)
        # print('DB Border Dimensions:', bottom-top, 'x', right-left)
        return image, count, [top,left,bottom,right], word_count, ratio, margin

    @staticmethod
    def detect_AB_border(image, percentage_threshold=0.1):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get image dimensions
        h, w = gray.shape

        # Compute the sum of pixel values along rows and columns
        row_sums = np.sum(gray, axis=1)
        col_sums = np.sum(gray, axis=0)

        # Initialize border coordinates
        top, bottom, left, right = 0, h, 0, w

        # Detect top border
        for i in range(h - 1):
            if abs(row_sums[i] - row_sums[i + 1]) / row_sums[i] > percentage_threshold:
                top = i + 1
                break

        # Detect bottom border
        for i in range(h - 1, 0, -1):
            if abs(row_sums[i] - row_sums[i - 1]) / row_sums[i] > percentage_threshold:
                bottom = i
                break

        # Detect left border
        for j in range(w - 1):
            if abs(col_sums[j] - col_sums[j + 1]) / col_sums[j] > percentage_threshold:
                left = j + 1
                break

        # Detect right border
        for j in range(w - 1, 0, -1):
            if abs(col_sums[j] - col_sums[j - 1]) / col_sums[j] > percentage_threshold:
                right = j
                break

        return left, top, right, bottom

    @staticmethod
    def highlight_border(image, top, bottom, left, right):
        highlighted_image = image.copy()
        cv2.rectangle(highlighted_image, (left, top), (right, bottom), (0, 0, 255), 3)
        return highlighted_image

    @staticmethod
    def bg_complexity(image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        edges=cv2.Canny(gray,100,200)
        variance=np.var(edges)
        return variance
    
    @staticmethod
    def detect_cropped_image_using_doctr(image_path, image_shape, margin=10):
        image = Image.open(image_path)
        # image = read_from_data_lake(image_path)
        width,height = image.size[:2]
        image_shape = [width, height]
        
        # with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_path)[-1]) as temp_file:
        #     image.save(temp_file, format=image.format)
        #     temp_file_path = temp_file.name   
        # doc = DocumentFile.from_images(temp_file_path)
        
        doc = DocumentFile.from_images(image_path)
        result = model(doc)
        output = result.export()

        # Extract text elements and bounding boxes
        text_data = output['pages'][0]["blocks"]

        left_crop = right_crop = top_crop = bottom_crop = False
        count=0
        aliases = []
        for block in text_data:
            for line in block["lines"]:
                for word in line["words"]:
                    geometry=word["geometry"]
                    #print(len(geometry))
                    if len(geometry)==2:
                        (x0,y0),(x1,y1)=geometry
                    #else:
                        #x0,y0,x1,y1=0,0,1,1
                    #(x0, y0, x1, y1) = word["geometry"]
                    x = int(x0 * image_shape[1])
                    y = int(y0 * image_shape[0])
                    w = int((x1 - x0) * image_shape[1])
                    h = int((y1 - y0) * image_shape[0])
                    #print(x,y,w,h)
                    # Check if text is within the margin from the edges
                    
                    if x <= margin and left_crop is False:
                        left_crop = True
                        aliases.append('LEFT')
                    if y <= margin and top_crop is False:
                        top_crop = True
                        aliases.append('TOP')
                    if x + w >= image_shape[1] - margin and right_crop is False:
                        right_crop = True
                        aliases.append('RIGHT')
                    if y + h >= image_shape[0] - margin and bottom_crop is False:
                        bottom_crop = True
                        aliases.append('BOTTOM')
                        
        doctr_crop = ""
        severity = "No Cropping Detected"
        if left_crop or right_crop or top_crop or bottom_crop:
            severity = "LOW"
            doctr_crop = "Cropped"
        elif (left_crop and right_crop) or (top_crop and bottom_crop):
            severity = "MEDIUM"
            doctr_crop = "Cropped"
        elif left_crop and right_crop and top_crop and bottom_crop:
            severity = "HIGH"
            doctr_crop = "Cropped"
        else:
            doctr_crop = "Not Cropped"
    
        return severity, aliases, doctr_crop
    
    @staticmethod
    def detect_cropped_image_Sobel(image_path, mild_threshold=30, moderate_threshold=60, severe_threshold=90):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or unable to read.")
            
        # if constants.IS_S3_PATH:
        #     raw_image_data = read_from_data_lake(image_path, reader='cv2').getvalue()
        # else:    
        #     raw_image_data = read_from_data_lake(image_path, reader='cv2')
            
        # nparr = np.frombuffer(raw_image_data, np.uint8)
        # bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        # gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        

        # Calculate the gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate the magnitude of the gradients
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Calculate mean magnitude near the borders
        border_width = 10
        top_border = magnitude[:border_width, :]
        bottom_border = magnitude[-border_width:, :]
        left_border = magnitude[:, :border_width]
        right_border = magnitude[:, -border_width:]

        mean_top = np.mean(top_border)
        mean_bottom = np.mean(bottom_border)
        mean_left = np.mean(left_border)
        mean_right = np.mean(right_border)

        # Determine the severity of cropping
        max_mean = max(mean_top, mean_bottom, mean_left, mean_right)
        # print(max_mean)
        mean_values = {"TOP" : mean_top, "BOTTOM" : mean_bottom, "LEFT" : mean_left, "RIGHT": mean_right}
        aliases = [alias for alias, value in mean_values.items() if value > 30]
        sobel_crop = ""
        if not aliases :
            aliases = []
        if max_mean > severe_threshold:
            severity = "HIGH"
            sobel_crop = "Cropped"
        elif max_mean > moderate_threshold:
            severity = "MEDIUM"
            sobel_crop = "Cropped"
        elif max_mean > mild_threshold:
            severity = "LOW"
            sobel_crop = "Cropped"
        else:
            severity = "No Cropping Detected"
            sobel_crop = "Not Cropped"

        return severity, aliases, sobel_crop
    
    @classmethod
    def handle_cropping_detection(cls, image_path, variance):
        image = Image.open(image_path)
        # image = read_from_data_lake(image_path)
        width,height = image.size[:2]
        image_shape = [width, height]
        if variance < 4000:
            return cls.detect_cropped_image_Sobel(image_path)
        else:
            return cls.detect_cropped_image_using_doctr(image_path, image_shape)
   

                
    @classmethod            
    def common_proc(cls, image_path):
        # image = read_from_data_lake(image_path)
        image = Image.open(image_path)
        width,height = image.size[:2]
        image_shape = [width, height]
        final_crop = ""
        _,m,result_S= cls.detect_cropped_image_Sobel(image_path)
        _,m,result = cls.detect_cropped_image_using_doctr(image_path, image_shape, margin=100)
        if result_S == "Cropped" and result == "Cropped":
            final_crop = "Cropped"
        if result_S == "Not Cropped" and result == "Not Cropped":
            final_crop = "Not Cropped" 
        if result_S == "Not Cropped" and result == "Cropped":
            final_crop = "Not Cropped"
        if result_S == "Cropped" and result == "Not Cropped":
            if doc_check==1:
                final_crop = "Not Cropped"
            else:
                final_crop = "Cropped"
        return final_crop
    
    @staticmethod
    def calculate_margin(doctr_coords, top, left, bottom, right):
        top_margin = abs(doctr_coords[0]-top)
        left_margin = abs(doctr_coords[1]-left)
        bottom_margin = abs(doctr_coords[2]-bottom)/bottom
        right_margin = abs(doctr_coords[3]-right)
        
        # print("Margin differences: ", top_margin, left_margin, bottom_margin, right_margin)
        margin= [top_margin, left_margin, bottom_margin, right_margin]
        count = sum(diff>10 for diff in margin)
        # print("Count :", count)
        return count, margin

    @staticmethod
    def create_excel_file(count, severity, final_crop, page_number, DB_border_coor, sides):
        
        severity_four_corner = ""
        
        if count == 4:
            severity_four_corner = "No Alerts"
        elif count == 3:
            severity_four_corner = "LOW"
        elif count == 2:
            severity_four_corner = "MEDIUM"
        else:
            severity_four_corner = "HIGH"
            
        try:
            insight = f"{count}_Corners detected"
        except:
            insight = ""
            

        insight_crop = ""
        if severity == "HIGH":
            count_edges_crop = 4
        elif severity == "MEDIUM":
            count_edges_crop = 2
        elif severity == "LOW":
            count_edges_crop = 1
        else:
            count_edges_crop = 0
            
        try:
            insight_crop = f"{count_edges_crop}_Sides cropped"
        except:
            insight_crop = ""

           
        flag_four_corner = ""
        if count != 4:
            flag_four_corner = 1
        else: 
            flag_four_corner = 0

        flag_crop = ""
        if severity == "No Cropping Detected":
            severity = 'No Alerts'
            flag_crop = 0
        else:
            flag_crop = 1
                
        df1 = pd.DataFrame({"Rule": (f"Four_Corners_Page_Num_{page_number}", f"Image_Cropping_Page_Num_{page_number}"), "Flag": (flag_four_corner, flag_crop), "Severity_Level": (severity_four_corner, severity), "Insight": (insight, insight_crop)})
        df2 = pd.DataFrame({"Metadata": (f"Four_Corners_Page_Num_{page_number}", f"Image_Cropping_Page_Num_{page_number}"), "Value": (insight, insight_crop), "Expected_Value": (DB_border_coor, sides)})
        
        # df1.to_excel(excel_filename, index=False)
        # print(f'Excel file created as {excel_filename}')
        
        return df1, df2, count_edges_crop 
                
    @classmethod
    def process_pdfs(cls, input_next_module):
        # pdf_name = os.path.basename(input_next_module)[:-4]
        # pdf_output_folder = os.path.join(output_folder, pdf_name)
        # os.makedirs(pdf_output_folder, exist_ok = True)
        pdf_document = read_from_data_lake(input_next_module)
        final_df1 = pd.DataFrame()
        final_df2 = pd.DataFrame()
        # final_df2 = pd.DataFrame()
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap()
            image_path = 'page_image.png'
            pix.save(image_path)   
            # image_path = os.path.join(pdf_output_folder, f"{pdf_name}_page_{page_number +1}.png")
            # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image_file:
            #     temp_image_path = temp_image_file.name
            #     pix.save(temp_image_path)
#             with open(temp_image_path,'rb') as temp_image_file:
#                 image_bytes = temp_image_file.read()
                
#             temp_image_path = os.path.join(json_output_dir, 'temp.png')
#             img_bytes = io.BytesIO()
#             pix.save(img_bytes)
#             img_bytes.seek(0)
#             write_to_data_lake(temp_image,temp_image_path)
            # pix.save(image_path)
            
        
            pil_image, count, DB_border_coor, word_count, ratio, margin = cls.doctr_check(image_path)
            # pil_image = pil_image.convert('RGB')
            open_cv_image = np.array(pil_image)

            # Convert RGB to BGR
            image = open_cv_image[:, :, ::-1].copy()
            top, bottom, left, right = cls.detect_AB_border(image)
            AB_border_coor = [left, top, right, bottom]

            # Highlight the border
            highlighted_image = cls.highlight_border(image, top, bottom, left, right)
            # Get dimensions of the images
            highlighted_dim = highlighted_image.shape[:2]
            # img_dim = str([highlighted_dim[0], highlighted_dim[1]])
            img_area = highlighted_dim[0] * highlighted_dim[1]
            if img_area > 1.6*((bottom-top)*(right-left)):
                count, margin = cls.calculate_margin(DB_border_coor, top, left, bottom, right)
            # print(f'Image Dimensions: {highlighted_dim[0]} x {highlighted_dim[1]}')
            # print('AB Border Dimensions:', bottom-top, 'x', right-left)
            # base = os.path.splitext(os.path.basename(image_path))[0]
            # new_filename = f"{base}_AB-DB.jpg"

            # Display the image
            # plt.figure(figsize=(10, 5))
            # plt.title('Highlighted AB & DB Borders')
            # plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
            # highlighted_image = cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(new_filename, highlighted_image)
            # print(f"image saved as {new_filename}")

            # plt.show()
            # variance = cls.bg_complexity(np.array(pil_image))
            variance = cls.bg_complexity(open_cv_image)

            severity, sides, crop =  cls.handle_cropping_detection(image_path, variance)
                # _, aliases = detect_cropped_image_Sobel(image_path)

            final_crop = cls.common_proc(image_path)
            df1_pdf, df2_pdf, count_edges_crop = cls.create_excel_file(count, severity, final_crop, page_number+1, DB_border_coor, sides)
            final_df1 = pd.concat([final_df1, df1_pdf], ignore_index=True)
            final_df2 = pd.concat([final_df2, df2_pdf], ignore_index=True)
            # display(final_df)
            # print('count_edges_crop----------',count_edges_crop)
            # print('count----------',count)
            # print('DB_border_coor----------',DB_border_coor)
            # print('sides----------',sides)
            # print('AB_border_coor----------',AB_border_coor)
            


        return final_df1, final_df2, count_edges_crop, count, DB_border_coor, sides, AB_border_coor

    
    @classmethod
    def metadata_entry(cls, input_next_module, bank_names, pdf_filename_file, four_corner_run, pdf_type):
        excel_df = pd.DataFrame(columns=['tabs_df', 'tab_name'])
        try:
            
            if four_corner_run == 1:
                four_corner_df1, four_corner_df2, count_edges_crop, count, DB_border_coor, sides, AB_border_coor = cls.process_pdfs(input_next_module)
            else:
                four_corner_df1 = pd.DataFrame()
                four_corner_df2 = pd.DataFrame()

            # display(four_corner_df)
            df_final_meta = pd.DataFrame()
            final_report_insight = pd.DataFrame(columns = ['Alerts','Severity'])
            excel_df = pd.DataFrame(columns = ['tabs_df','tab_name'])

            doc = read_from_data_lake(input_next_module)
            PDF_type = pdf_type
            for page in doc:
                page.set_rotation(0)
                page.wrap_contents()
            pdf_final = doc.write()

            try:
                meta_repo = read_from_data_lake(constants.METADATA_REPOSITORY_PATH, sheet_name='repo')
                pdf_software_repo = read_from_data_lake(constants.METADATA_REPOSITORY_PATH, sheet_name='Editing_software')
                scanner_repo = read_from_data_lake(constants.METADATA_REPOSITORY_PATH, sheet_name='Scanner_List')

                meta_repo.set_index('Bank',inplace = True)
                meta_repo = meta_repo.fillna('')
                pdf_software_repo = pdf_software_repo.fillna('')
                scanner_repo = scanner_repo.fillna('')
            except:
                cls.meta_error_list.append("Metadata: Repository path missing")
                meta_repo = pd.DataFrame(columns=['Bank','Producer','Creator','Author'])
                pdf_software_repo = pd.DataFrame(columns=['Editing_software'])
                scanner_repo = pd.DataFrame(columns=['Scanner_List'])
                # print('meta_error_list-------',cls.meta_error_list)

            # bank_name = bank_names[0]

            df_final = cls.meta_repo_value(bank_names, pdf_final, pdf_filename_file)
            df_final_meta = pd.concat([df_final_meta, df_final], axis =0)
            metadata, bank_name, excel_df = cls.metadata_main(pdf_final, [bank_names], meta_repo, excel_df, PDF_type, four_corner_df2)
            final_report_insight, excel_df = cls.metadata_rules(metadata, meta_repo, pdf_software_repo, scanner_repo, final_report_insight, excel_df, four_corner_df1, PDF_type)
        
        except Exception as e:
            print(f"Error in Metadata module: {e}")
            cls.meta_error_list.append(f"Metadata: Error {e}")
            final_report_insight = pd.DataFrame(columns=['Module', 'Alerts', 'Severity'])
            excel_df = pd.DataFrame(columns=['Metadata', 'Value', 'Expected_Value', 'Match'])
            df_final_meta = pd.DataFrame(columns=['Rule', 'Flag', 'Severity_Level', 'Insight'])
            excel_df.loc[len(excel_df.index)] = [excel_df, '1A. Metadata_Raw_Output']
            excel_df.loc[len(excel_df.index)] = [df_final_meta, '1B. Metadata_Report']
            
        return final_report_insight, excel_df, df_final_meta, cls.meta_error_list

if __name__ == '__main__':
    
    # excel_df.loc[len(excel_df.index)] = [df, '1B.Metadata_']
    # excel_df1['tabs_df'].loc[1]
    result = pd.concat([df1, df2], ignore_index=True).fillna('')

    # import os
    input_next_module = 'core_v2/Wells Fargo Tampered Input.pdf'

    # input_next_module = os.path.join(constants.S3_DATA_PREFIX, 'input_data', 'Wells Fargo Tampered Input.pdf')

    bank_names = ['wells fargo']
    pdf_filename_file = "wells fargo input.pdf"
    
    final_report_insight, excel_df, _ = MetadataAnalyzer.metadata_entry(input_next_module, bank_names, pdf_filename_file)
    print(final_report_insight)
    print(excel_df)
    
    