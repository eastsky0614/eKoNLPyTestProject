
# coding: utf-8

# In[56]:


import os
import sys
import platform
import pandas as pd
import re
from datetime import datetime, timedelta


# In[57]:


import requests
from bs4 import BeautifulSoup
import html2text
from datetime import datetime


# In[1]:


# get pdf download link, meeting date, registration date
# from_date : start date
def get_minutes_list(from_date='20050101'):
    prefix_addr = "https://www.bok.or.kr"
    from_date = datetime.strptime(from_date, '%Y%m%d')

#     page index is from 1 to 30
    for pageIndex in range(1, 31):
        url = 'https://www.bok.or.kr/portal/bbs/B0000245/list.do?menuNo=200761&pageIndex={}'.format(pageIndex)
        user_agent = 'Mozilla/5.0'
        headers ={'User-Agent' : user_agent}
        page = requests.get(url, headers=headers)

        try:
            soup = BeautifulSoup(page.content, 'html.parser')
#             get minutes board list
            brdList = soup.find_all('span', class_='col m10 s10 x9 ctBx')
#             get data information
            datainfo = soup.find_all('div', class_='col s12 dataInfo')

            for post, data in zip(brdList, datainfo):
#                 get minutes pdf download link
                guid = prefix_addr+post.a['href'][:-12]

#                 get description
                desPage = requests.get(guid)
                desSoup = BeautifulSoup(desPage.content, 'html.parser')   
                description = desSoup.find('div', class_='dbData').get_text().strip()
                if description.replace(' ','').find('통화정책방향') >= 0:
#                     get title
                    title = post.find('span', class_='titlesub').get_text().strip()
        
#                     get meeting date of minutes
                    mdate = title[title.find(')(') + 2:-1]            
                    if mdate[-1] == '.':
                        mdate = mdate[:-1]
                    mdate = datetime.strptime(mdate, '%Y.%m.%d')

#                     get data until from_date
                    if mdate < from_date:
                        break

#                     get registration date of minutes
                    rdate = data.find('span', class_='date').get_text().strip()
                    rdate = datetime.strptime(rdate[3:], '%Y.%m.%d')    

                    get_minutes_file(guid, mdate, rdate)
        except:
            print("get url.content error and pass page{} it".format(pageIndex))


# In[59]:


# download minutes pdf file
def get_minutes_file(page_addr, mdate, rdate):
#     save file path
    file_header = 'data/minutes/pdf/KO_'
    prefix_addr = "http://bok.or.kr"

    page = requests.get(page_addr)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    try:
        links = soup.find('div', class_='addfile').find_all('a')

        for link in links:
            filename = link.get_text()
            filename = filename.replace('\r', '').replace('\t', '').replace('\n', '')

            if filename[-3:] == 'pdf':
                filename = mdate.strftime('%Y%m%d') + "_" + rdate.strftime('%Y%m%d')+ '.pdf'
                file_addr = prefix_addr + link["href"]
                file_res = requests.get(file_addr)
                filepath = file_header + filename

#                 save file
                with open(filepath, 'wb') as f:
                    f.write(file_res.content)

                print('save file name : ')
                print(filename)
    except:
        print("get file failed and pass it")


# In[62]:


# download and save minutes pdf file
get_minutes_list()

