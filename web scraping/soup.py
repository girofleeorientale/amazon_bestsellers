from bs4 import BeautifulSoup
import re
import urllib.request
import requests

def is_bestseller(headers, url):
    headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Vasya/20100101 Firefox/82.0' } 
    #page = requests.get("https://www.amazon.co.uk/dp/B08LD4VXGL", headers=headers)
    #html_contents = page.text

    page = requests.get(url, headers=headers)
    html_contents = page.text

    soup = BeautifulSoup(html_contents, 'lxml')

    string = " Best Sellers Rank "

    el = soup.find("th", string=string).parent

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

    #print('elt: ', el)
    print('string: ', string_found)
    print('res ', res)

    return (res <= 10)