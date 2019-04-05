
# coding: utf-8

# In[1]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import datetime
import json
import re
import pandas as pd
from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open

from datetime import datetime


# In[2]:


def read_pdf_file(pdfFile):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    process_pdf(rsrcmgr, device, pdfFile)
    device.close()
    content = retstr.getvalue()
    retstr.close()
    return content


# In[ ]:


file_header = 'C:/Users/user/Downloads/project/data/bond/'

for n in range(0,48): #pages range MAX 48
    
    pdf_url = 'https://www.myasset.com/myasset/research/rs_list/rs_list.cmd?cd006=&cd007=RF09&cd008=&rel_jongmok_list=&searchKeyGubun=1&keyword=&jongMok_keyword=&keyword_in=&startCalendar=2005%2F01%2F01&endCalendar=2017%2F12%2F31&pgCnt=50&page={}#spot'.format(n+1)
    open_url = urlopen(pdf_url)
    read_url = BeautifulSoup(open_url, 'html.parser', from_encoding='utf-8')
    list = read_url.findAll('td', {'class': 'js-chkBlank'})
    date = read_url.findAll('tr', {'class': 'js-moveRS'})
    
    for x, y in zip(list, date):
        
        try:
            i = x.find('a')
            pdf_num = i.attrs['data-seq']
            web_pdf=urlopen("http://file.myasset.com/sitemanager/upload/" + pdf_num)
            read_pdf = read_pdf_file(web_pdf)
            print(read_pdf)

            
            pdf_num = pdf_num.replace('/', '').replace('.', '').replace('pdf', '.txt')
            
            mdate = pdf_num[0:8]                    #pdf_num[:]
            rdate = str(y.find('td'))               #js-moveRs
            rdate = rdate[4:14].replace('/','')

            print(file_header+mdate+rdate)
            with open(file_header+mdate+"_"+rdate+".txt", 'w', encoding='utf-8') as f:   
                #file_header+mdade+rdate+"txt"
                f.write(read_pdf)
           
          
        except :
            print("파일이 없습니다.")

                    

