

import pandas as pd
import os
import datetime
import dateutil.parser
from minet.twitter import TwitterAPIScraper
import numpy as np
scraper = TwitterAPIScraper()
from sklearn.preprocessing import normalize


df = pd.read_csv('data/data_ready.csv',dtype={'user_id':'str'})



try:
    os.mkdir('data/last_tweets')
except:
    pass
h=0
for id in list(df.screen_name)[:]:
    h+=1
    fname='data/last_tweets/'+id+'.csv'
    #print(h,fname,end='\t')
    if os.path.isfile(fname):
        pass
    else:
        tweets=[]
        for tweet in scraper.search('from:'+str(id),limit=20):
            tweets.append(tweet)
        pd.DataFrame(tweets).to_csv(fname)

tweets_content=[]
for id in list(df.screen_name)[:]:    
    fname='data/last_tweets/'+id+'.csv'
    try:
        tweets_df=pd.read_csv(fname,usecols=['local_time','lang'])
        frcount=tweets_df.lang.value_counts()['fr']
        most_recent_date=dateutil.parser.parse(tweets_df['local_time'].iloc[0])
        least_recent_date=dateutil.parser.parse(tweets_df['local_time'].iloc[-1])
        freshness = (datetime.datetime.today() - dateutil.parser.parse(tweets_df['local_time'].iloc[0])).days
        rythm = (most_recent_date-least_recent_date).days
        tweets_content.append([int(freshness/365*12),int(len(tweets_df)*365/rythm),frcount/len(tweets_df)])
        
    except:
        tweets_content.append([100,0,0])
        print (id,'out',end=' ')
np.save("embeddings/tweets_content.npy", normalize(np.array(tweets_content)))



