

#################################### NLP ############################################

'''Problem Statement: -
In the era of widespread internet use, it is necessary for businesses to understand 
what the consumers think of their products. If they can understand what the consumers 
like or dislike about their products, they can improve them and thereby increase their 
profits by keeping their customers happy. For this reason, they analyze the reviews of 
their products on websites such as Amazon or Snapdeal by using text mining and sentiment 
analysis techniques. 

Task 1:
Extract reviews of any product from e-commerce website Amazon.
Perform sentiment analysis on this extracted data and build a unigram and bigram word cloud. 
'''

'''Business Objective:
    Perform extracting reviews from any e-commerce website and perform sentiment analysis
    on the data in order to understand need of user and reduce churn rate.'''
# Amazon
#TV 
import bs4
from bs4 import BeautifulSoup as bs
import requests

link = "https://www.amazon.in/dp/B0C82ZHYQ8/ref=sspa_dk_hqp_detail_aax_0?sp_csd=d2lkZ2V0TmFtZT1zcF9ocXBfc2hhcmVk&th=1"
#link= "https://www.amazon.in/Xiaomi-inches-Smart-Google-L43MA-SIN/dp/B0DC6XZJ2D?ref_=ast_sto_dp&th=1"
page=requests.get(link)
page
page.content

soup=bs(page.content,'html.parser')
print(soup.prettify())


## now let us scrap the contents
names=soup.find_all('span',class_="a-profile-name")
names
### but the data contains with html tags,let us extract names from html tags
cust_names=[]
for i in range(0,len(names)):
    cust_names.append(names[i].get_text())
    
cust_names
len(cust_names)
cust_names.pop(-1)
cust_names.pop(-1)
cust_names.append(1)
cust_names.append(1)

len(cust_names)

# 3
### There are total 3 users names 
#Now let us try to identify the titles of reviews

title_rate=soup.find_all('a',class_='review-title')
tr_list = [x.text.strip() for x in title_rate]
tr_list
len(tr_list)
ratings = []
reviews = []

# Process each entry in tr_list
for i in tr_list:
    rating, review = i.split('\n', 1)
    ratings.append(rating)
    reviews.append(review)
ratings 
reviews 
rate = [int(i[0]) for i in ratings]
print(rate)
len(rate )
len(reviews )

## now let us scrap review body
reviews=soup.find_all("div",class_="a-row a-spacing-small review-data")
reviews
review_body=[]
for i in range(0,len(reviews)):
    review_body.append(reviews[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
review_body
len(review_body)

########
#convert to csv file
import pandas as pd
df=pd.DataFrame()
df['customer_names']=cust_names
df['review_title']=reviews
df['rate']=rate
df['review_body']=review_body
df
df.to_csv('C:/8-text_Mining/text_mining/tv_reviews.csv',index=True)

######
#sentiment analysis
import pandas as pd
from textblob import TextBlob
df=pd.read_csv("C:/8-text_Mining/text_mining/tv_reviews.csv")
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity'] 

###############################################################################

'''Task 2:
Extract reviews for any movie from IMDB and perform sentiment analysis.
'''
'''Business Objective:
    Analyzing the sentiment of reviews can help suggest movies that match a 
    user's taste, based on their previous ratings and review patterns.'''
# IMDB
import bs4
from bs4 import BeautifulSoup as bs
import requests
link= "https://www.imdb.com/title/tt1400986/reviews/?ref_=tt_ov_ql_2"
# link = "https://www.gsmarena.com/honor_200-13050.php"
#link="https://www.imdb.com/title/tt0371746/reviews/?ref_=tt_ql_2"
page=requests.get(link)
page
page.content


soup=bs(page.content,'html.parser')
print(soup.prettify())


title=soup.find_all('a',class_='title')
title

review_titles=[]
for i in range(0,len(title)):
    review_titles.append(title[i].get_text())
review_titles


review_titles[:]=[ title.strip('\n')for title in review_titles]
review_titles
len(review_titles)
#Got 24 review titles



rating=soup.find_all('span',class_='rating-other-user-rating')
rating

rate=[]
for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate
rate= [r.strip().split('/')[0] for r in rate]
rate
len(rate)
rate.append('')
rate.append('')
len(rate)



review=soup.find_all('div',class_='text show-more__control')
review

review_body=[]
for i in range(0,len(review)):
    review_body.append(review[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
len(review_body)

###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['review_title']=review_titles
df['rate']=rate
df['review_body']=review_body
df
df.to_csv("C:/8-text_mining/text_mining/IDBM_reviews.csv",index=True)
######### 
#sentiment analysis
import pandas as pd
from textblob import TextBlob
df=pd.read_csv("C:/8-text_mining/text_mining/IDBM_reviews.csv")
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']

#############################################################################

'''Task 3: 
Choose any other website on the internet and do some research on how to extract 
text and perform sentiment analysis
'''

'''Business Objective:
    Perform extracting reviews from any e-commerce website and perform sentiment analysis
    on the data in order to understand need of user and reduce churn rate.'''
    
# Flipkart
#Boat Product Watch
import bs4
from bs4 import BeautifulSoup as bs
import requests
link = "https://www.boat-lifestyle.com/products/enigma-r32-round-tft-display-women-smartwatch"
#link="https://www.boat-lifestyle.com/products/airdopes-alpha-true-wireless-earbuds?_gl=1*1asuzs9*_up*MQ..&gclid=CjwKCAjw0aS3BhA3EiwAKaD2ZUE1Z2Y1zY8L8vqS0r_ZywCCySZGLKmMMvpC4FI_5fZjG4ZTTBcMOBoCbO4QAvD_BwE"
page=requests.get(link)
page
page.content
## now let us parse the html page
soup=bs(page.content,'html.parser')
print(soup.prettify())

## now let us scrap the contents
names=soup.find_all('span',class_="jdgm-rev__author")
names
### but the data contains with html tags,let us extract names from html tags
cust_names=[]
for i in range(0,len(names)):
    cust_names.append(names[i].get_text())
    
cust_names
len(cust_names)
#cust_names.pop(-1)
#len(cust_names)


### There are total 6 users names 
#Now let us try to identify the titles of reviews

title=soup.find_all('b',class_="jdgm-rev__title")
title
# when you will extract the web page got to all reviews rather top revews.when you
# click arrow icon and the total reviews ,there you will find span has no class
# you will have to go to parent icon i.e.a
#now let us extract the data
review_titles=[]
for i in range(0,len(title)):
    review_titles.append(title[i].get_text())
review_titles

len(review_titles)
##now let us scrap ratings
rating=soup.find_all('span',class_="jdgm-rev__rating")
rating
###we got the data
ratings = [int(span['data-score']) for span in soup.find_all('span', {'class': 'jdgm-rev__rating'})]

# Print the ratings
print(ratings)
len(ratings)

## now let us scrap review body
reviews=soup.find_all("div",class_="jdgm-rev__body")
reviews
review_body=[]
for i in range(0,len(reviews)):
    review_body.append(reviews[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
review_body
len(review_body)

####
###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['customer_names']=cust_names
df['review_title']=review_titles
df['rate']=ratings
df['review_body']=review_body
df
df.to_csv('C:/8-text_mining/text_mining/Boat_Watch_reviews.csv',index=True)

######
#sentiment analysis
import pandas as pd
from textblob import TextBlob
df=pd.read_csv("C:/8-text_mining/text_mining/Boat_Watch_reviews.csv")
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity'] 


