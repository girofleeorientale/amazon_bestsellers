from audioop import reverse
from bs4 import BeautifulSoup
import requests
from soup import is_bestseller
import pandas as pd 

# find stars
def get_rating(soup):

	try:
		rating = soup.find("i", attrs={'class':'a-icon a-icon-star a-star-4-5'}).string.strip()
		
	except AttributeError:
		
		try:
			rating = soup.find("span", attrs={'class':'a-icon-alt'}).string.strip()
		except:
			rating = ""	

	return rating

# find the number of reviews
def get_review_count(soup):
	try:
		review_count = soup.find("span", attrs={'id':'acrCustomerReviewText'}).string.strip()
		
	except AttributeError:
		review_count = ""	

	return review_count

# define if the product is a bestseller (i.e. top10)
def is_bestseller(soup):

    string = " Best Sellers Rank "

    try:

        el = soup.find("th", string=string).parent
        print("element : ", el)

        string_found = str(el)
        string_split = string_found.split()
        res = ''

        for i, elt in enumerate(string_split):
            if (elt == 'in') and ("span" in string_split[i-1]):
                res = ''
                for x in string_split[i-1]:
                    if x.isdigit() == True:
                        res+=x

        res = int(res)  
    except AttributeError:
        res = 100	
	
    return (res <= 10)

def is_bestseller_tr(soup):
      try:
            tr_elements = soup.find_all('tr')
            print('RES', tr_elements)
	
            res = []

            for i, tr in enumerate(tr_elements):
                if 'Best Sellers Rank' in str(tr):
                    res.append(str(tr))
			
            string_split = res[0].split()
            res = ''

            for i, elt in enumerate(string_split):
                    if (elt == 'in') and ("span" in string_split[i-1]):
                        res = ''
                    for x in string_split[i-1]:
                        if x.isdigit() == True:
                            res+=x

            res = int(res)
      except AttributeError:
            res = 100	
	
      return (res <= 10)

# update one row in a chunk
def update_one_row(id, chunk, url):
    df1 = pd.read_csv('split_csv_pandas/chunk0.csv') 
    
    # get to the page
    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Vasya/20100101 Firefox/82.0' } 
    page = requests.get(url, headers=headers)
    html_contents = page.text

    # make soup
    soup = BeautifulSoup(html_contents, 'lxml')
	
    stars = get_rating(soup=soup)
    reviews = get_review_count(soup=soup)

    print("html", html_contents)

    is_best = is_bestseller_tr(soup=soup)
	
    
    print('is B: ', is_best)
    df1.at[id,'isBestSeller']=is_best
    print(df1[['isBestSeller']])
    print(df1.loc[[23]]['isBestSeller'])

	

#print(df1.head(5))


#update_one_row(23, 3, 'https://www.amazon.co.uk/dp/B09HGRXXTM')

#  *****  LOOK AT 20 FIRST IN DF1  ******

df1 = pd.read_csv('split_csv_pandas/chunk2.csv') 

#print(df1[['productURL', 'isBestSeller']].head(10))

#df_storage= df1[df1['categoryName'] == 'Storage & Home Organisation']
#df_storage_good_review = df_storage[df_storage['reviews'] > 500]
#print(df_storage_good_review[['productURL', 'isBestSeller']].head())

df_string_instr = df1[df1['categoryName'] == 'String Instruments']
df_instr_good_review = df_string_instr[df_string_instr['reviews'] > 50]
print(df_instr_good_review[['productURL', 'isBestSeller']].head(20))
#print(df_instr_good_review)

#string_instr_bestsellers = df_string_instr.loc[df_string_instr['isBestSeller'] == True]
#print(string_instr_bestsellers)

update_one_row(24526, 4, 'https://www.amazon.co.uk/dp/B087M27985')

print(df1.at[24526, 'isBestSeller'])
