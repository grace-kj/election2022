import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import datetime as dt
from datetime import timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.tools import argparser
from apscheduler.schedulers.background import BackgroundScheduler
import time

pd.options.display.float_format = "{:.2f}".format

#developer key
DEVELOPER_KEY = "AIzaSyBWjQG7t6H-LFX79MVn3y51NSVuZDCWiec"
YOUTUBE_API_SERVICE_NAME="youtube"
YOUTUBE_API_VERSION="v3"
youtube = build(YOUTUBE_API_SERVICE_NAME,YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)


def get_video_info(search_response):
  result_json = {}
  idx =0
  for item in search_response['items']:
    if item['id']['kind'] == 'youtube#video':
      result_json[idx] = info_to_dict(item['id']['videoId'], item['snippet']['title'], item['snippet']['description'], item['snippet']['thumbnails']['medium']['url'])
      idx += 1
  return result_json

def info_to_dict(videoId, title, description, url):
  result = {
      "videoId": videoId,
      "title": title,
      "description": description,
      "url": url
  }
  return result


def get_statistics_response(youtube, Id):
  search_response = youtube.videos().list(
    part = ["statistics"],
    id=Id,
    maxResults = 10
    ).execute()
  return search_response


def get_search_response_old(youtube, query, when):
  search_response = youtube.search().list(
    q = query,
    publishedAfter= when,
    order = "relevance",
    part = [ "snippet"],
    maxResults = 50
    ).execute()
  return search_response


def jsontodataframe(result):
  videoID= []
  title=[]
  description=[]

  for key, elem in result.items():
    elem['title']= elem['title'].replace('&quot;','\'')
    elem['description']= elem['description'].replace('&quot;','\'')
  
    videoID.append(elem['videoId'])
    title.append(elem['title'])
    description.append(elem['description'])

  df= pd.DataFrame({'ID': videoID, 'title': title, 'description': description})
  return df

def getstatistics(df):
  statistics= []

  for id in df['ID']:
    search_result= get_statistics_response(youtube, id)
    statistics.append(search_result['items'][0]['statistics'])

  viewcount= []
  likecount=[]
  commentcount=[]

  for counts in statistics:
    if counts['viewCount']:
      view= counts['viewCount']
    else:
      view= 0

    if 'likeCount' in counts.keys():
      like= counts['likeCount']
    else:
      like= 0

    if 'commentCount' in counts.keys():
      comment= counts['commentCount']
    else:
      comment= 0
    
    viewcount.append(view)
    likecount.append(like)
    commentcount.append(comment)
      
      
  df['viewcount']= viewcount
  df['likecount']= likecount
  df['commentcount']= commentcount
  
  return df

def maketest(df, DATE, PERSON):
  df['document']= df['title'] + df['description']
  df['label']= [0 for _ in range(len(df))]
  df_test= df[['ID','document','label']]
  df_test.columns=['id','document','label']
  df_test.to_csv(os.path.join('/content/drive/MyDrive/Colab Notebooks/선거인단 koelectra/data/BoolQ', DATE + PERSON+ '.tsv'), index= False, sep= '\t')


def score(DATE, PERSON):
  result= pd.read_csv(os.path.join('/content/drive/MyDrive/Colab Notebooks/선거인단 koelectra/koelectra results', DATE + PERSON +'.tsv'),header= None, delimiter= ',')
  result=result.transpose()
  df= pd.read_csv(os.path.join('/content/drive/MyDrive/Colab Notebooks/선거인단 koelectra/old', DATE + PERSON +'.tsv'), delimiter= '\t')
  df['label']=result
  df['label']= df['label'].apply(lambda x: x if x==1 else -1)
  #define scores
  df['score']= ((df['viewcount']+df['commentcount']*10+df['likecount']*5))*df['label']

  return df


#main
def main():
  print('function started')
  DATELIST= []

  #ex) DATE= '1223'
  x = dt.datetime.now()
  DATE= dt.date.today().strftime('%y%m%d')[2:]
   
  #get old data
  when= str(x.year)+"-"+ str(x.month)+"-"+str(x.day)+ "T00:00:00Z" #(since)
  search_result_yoon =get_search_response_old(youtube,'윤석열', when)
  search_result_lee= get_search_response_old(youtube, '이재명', when)
  search_result_ahn= get_search_response_old(youtube, '안철수', when)

  #get data's info
  result_yoon=get_video_info(search_result_yoon)
  result_lee=get_video_info(search_result_lee)
  result_ahn=get_video_info(search_result_ahn)

  #json to dataframe
  df_yoon= jsontodataframe(result_yoon)
  df_lee= jsontodataframe(result_lee)
  df_ahn= jsontodataframe(result_ahn)

  #append data's statistics
  df_yoon_final= getstatistics(df_yoon)
  df_lee_final= getstatistics(df_lee)
  df_ahn_final= getstatistics(df_ahn)

  #save dataframe for later weight calculations
  df_yoon_final.to_csv(os.path.join('/content/drive/MyDrive/Colab Notebooks/선거인단 koelectra/old', DATE + '윤석열.tsv'), index= False, sep= '\t')
  df_lee_final.to_csv(os.path.join('/content/drive/MyDrive/Colab Notebooks/선거인단 koelectra/old', DATE + '이재명.tsv'), index= False, sep= '\t')
  df_ahn_final.to_csv(os.path.join('/content/drive/MyDrive/Colab Notebooks/선거인단 koelectra/old', DATE + '안철수.tsv'), index= False, sep= '\t')

  #make test data
  maketest(df_yoon_final, DATE, '윤석열')
  maketest(df_lee_final, DATE, '이재명')
  maketest(df_ahn_final, DATE, '안철수')

  #do classification
  b="2"
  l="0.00002"

  os.system('chmod 755 run_BoolQ.sh')
  os.system('python run_seq_cls.py --task BoolQ --config_file koelectra-base-v3.json --data {} --train {} --bs {} --lr {}'.format("test.tsv", False, b, l))
  os.system('python run_seq_cls.py --task BoolQ --config_file koelectra-base-v3.json --data {} --train {} --bs {} --lr {}'.format( DATE+"윤석열.tsv", False, b, l))
  os.system('python run_seq_cls.py --task BoolQ --config_file koelectra-base-v3.json --data {} --train {} --bs {} --lr {}'.format( DATE+"이재명.tsv", False, b, l))
  os.system('python run_seq_cls.py --task BoolQ --config_file koelectra-base-v3.json --data {} --train {} --bs {} --lr {}'.format( DATE+"안철수.tsv", False, b, l))
  
  '''
  #concat score
  DATELIST.append(DATE)
  for date in DATELIST:
    yoon= score(date, '윤석열')
    lee= score(date,'이재명')     
    ahn= score(date,'안철수') 
    if date== DATELIST[0]:
      score_yoon=yoon
      score_lee= lee
      score_ahn= ahn
    else:
      score_yoon= pd.concat([score_yoon, yoon])
      score_lee= pd.concat([score_lee, lee])
      score_ahn= pd.concat([score_ahn, ahn])
                         
  score_yoon= score_yoon.reset_index()
  score_lee= score_lee.reset_index()
  score_ahn= score_ahn.reset_index()

  #rolling
  roll= len(score_yoon['score'])
  yoon_rolled=[]
  storage= 0
  for i in range(len(score_yoon['score'])):
    storage+= score_yoon['score'][i]
    if i%roll==0:
      yoon_rolled.append(storage)
      storage=0
  lee_rolled=[]
  storage= 0
  for i in range(len(score_lee['score'])):
    storage+= score_lee['score'][i]
    if i%roll==0:
      lee_rolled.append(storage)
      storage=0
  ahn_rolled=[]
  storage= 0
  for i in range(len(score_ahn['score'])):
    storage+= score_ahn['score'][i]
    if i%roll==0:
      ahn_rolled.append(storage)
      storage=0

  #visualize
  plt.plot(yoon_rolled, label='yoon')
  plt.plot(lee_rolled, label='lee')
  plt.plot(ahn_rolled, label='ahn')
  plt.legend()
  plt.savefig(DATE+ '.png')
  '''

if __name__ == '__main__':

  scheduler = BackgroundScheduler()
  scheduler.start()
  scheduler.add_job(main, 'interval', hours=24)
  
  print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

  try:
      # This is here to simulate application activity (which keeps the main thread alive).
      while True:
          time.sleep(2)
  except (KeyboardInterrupt, SystemExit):
      # Not strictly necessary if daemonic mode is enabled but should be done if possible
      scheduler.shutdown()



