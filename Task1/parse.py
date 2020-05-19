import fnmatch
import os
import io
import re
import string
import urllib
from urllib.request import urlopen
from tika import parser
from lxml import etree as ET
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from xml.dom import minidom
from docx import Document
from bs4 import BeautifulSoup, NavigableString, Tag


def pretty(elem):
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml("  ")


def write_to_xml(text, xmlPath):
    element = ET.Element("doc")
    for line in text:
        p = ET.SubElement(element, "block")
        p.text = line
        element.append(p)
    # tree = ET.ElementTree(element)
    # tree.write(xmlPath)
    file = open(xmlPath, "w")
    file.write(pretty(element))
    file.close()


def write_to_xml_with_headers(lines, xmlPath):
    # appt = ET.Element("block", name=line.value)
    element = ET.Element("doc")
    for line in lines:
        if line.count(" ") < 10:
            parent = ET.Element("block")
            parent.set("header", line)
            element.append(parent)
        else:
            child = ET.SubElement(parent, "text")
            child.text = line

    # tree = ET.ElementTree(element)
    # tree.write(xmlPath)
    file = open(xmlPath, "w")
    file.write(pretty(element))
    file.close()


def create_xml(list, xmlPath):
    root = ET.Element("doc")

    for line in list:
        # Block
        if line[1] == "block":
            appt = ET.Element("block", header=line[0])
            root.append(appt)
        # Text
        if line[1] == "text":
            begin = ET.SubElement(appt, "text")
            begin.text = line[0]

    file = open(xmlPath, "w")
    file.write(pretty(root))
    file.close()


class Parse:


    def read_txt(input_path, output_path):
        file = open(input_path, "r")
        lines = file.readlines()
        file.close()
        list = [x for x in lines if x is not "\n"]
        write_to_xml_with_headers(list, output_path)

    def read_docx(input_path, output_path):
        document = Document(input_path)
        lines = []
        for para in document.paragraphs:
            if para is not "\n":
                lines.append(para.text)
        # list = [x for x in lines if x is not "\n"]
        write_to_xml_with_headers(lines, output_path)

    def read_pdf(input_path, output_path):
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams)
        fp = open(input_path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages, password, caching, True):
            interpreter.process_page(page)
        text = retstr.getvalue()
        fp.close()
        device.close()
        retstr.close()
        text = text.replace("", "").split("  ")
        write_to_xml_with_headers(text, output_path)

    def read_html(url, xmlPath):
        list = []
        rs = urlopen(url)
        root = BeautifulSoup(rs, 'html.parser')
        pattern = '<.*?>'
        regex = re.compile(r'{}'.format(pattern))
        title = root.find_all('title')
        title = regex.sub('', str(title))
        list.append((title, "block"))

        for h3 in root.find_all('h3'):
            h3 = regex.sub('', str(h3))

            list.append((h3, "block"))
        for p in root.find_all('p'):
            p = regex.sub('', str(p))
            if p!='':
                list.append((p, "text"))
            create_xml(list, xmlPath)
        # soup = BeautifulSoup(html, "lxml")
        #
        # # kill all script and style elements
        # for script in soup(["script", "style"]):
        #     script.extract()
        # text = soup.get_text()
        # lines = (line.strip() for line in text.splitlines())
        # chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # list = [x for x in chunks if x is not "\n"]
        # for line in list:
        #     line.replace("\r","\n")
        # return list
